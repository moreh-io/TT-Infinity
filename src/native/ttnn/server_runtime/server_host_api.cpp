// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#include "server_runtime/server_runtime_core.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <thread>
#include <tuple>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>

#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/operations/kvcache_manager/common/config.hpp"
#include "ttnn/operations/kvcache_manager/common/fabric_helpers.hpp"
#include "ttnn/operations/kvcache_manager/common/protocol/layout.hpp"
#include "ttnn/operations/kvcache_manager/common/protocol/packet.hpp"

namespace kvcache_manager::detail::server_runtime {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace {
// Common deadline for host-side completion waits that do not copy value data
// through the host.
constexpr auto kHostCompletionTimeout = std::chrono::seconds(5);

// The non-mapped write-through path double-reads every occupied DRAM bank
// through targeted PCIe copies before publishing a durable PUT. A remote T3K
// chip can legitimately need more than five seconds even for a modest
// multi-bank value, while the NoC-mapped path has no such host copy.
constexpr auto kTargetedPcieCommitDrainTimeout = std::chrono::seconds(30);

constexpr auto put_commit_drain_timeout(bool stage_noc_ok) {
  return stage_noc_ok ? kHostCompletionTimeout
                      : kTargetedPcieCommitDrainTimeout;
}

// Max whole-directory re-reads before ServerRuntimeCore::get concludes a true
// miss. A host-direct debug read can overlap an asynchronous device-side
// directory update; bounded retries distinguish that visibility window from a
// stable miss.
constexpr uint32_t kDirReadMaxRetries = 16;
} // namespace

ServerRuntimeCore::ScalarGetMode
ServerRuntimeCore::scalar_get_mode() const noexcept {
  const bool mapped_completion =
      scalar_get_pipeline_.mapped_completion_enabled &&
      get_completion_host_ptr_ != nullptr;
  const bool mapped_ingress = scalar_get_pipeline_.ingress.enabled &&
                              mapped_completion &&
                              get_ingress_host_ptr_ != nullptr &&
                              scalar_get_pipeline_.ingress.l1 != nullptr;
  return {.mapped_completion = mapped_completion,
          .mapped_ingress = mapped_ingress};
}

bool ServerRuntimeCore::test_mapped_get_ingress_enabled() const noexcept {
  return scalar_get_mode().mapped_ingress;
}

void ServerRuntimeCore::prepare_mapped_get_completion() {
  TT_ASSERT(scalar_get_mode().mapped_completion);
  *get_completion_host_ptr_ = 0;
  __builtin_ia32_sfence();
}

uint32_t ServerRuntimeCore::wait_for_mapped_get_completion() {
  TT_ASSERT(scalar_get_mode().mapped_completion);
  const auto deadline =
      std::chrono::steady_clock::now() + kHostCompletionTimeout;
  uint32_t spins = 0;
  while (true) {
    __builtin_ia32_lfence();
    const uint32_t raw_status = *get_completion_host_ptr_;
    if (raw_status != 0) {
      TT_FATAL(raw_status == 1u || raw_status == 2u || raw_status == 4u,
               "mapped GET completion has invalid status encoding {}",
               raw_status);
      return raw_status - 1u;
    }
    __builtin_ia32_pause();
    if ((++spins & 0xFFu) == 0) {
      throw_if_tiering_worker_failed();
      if (std::chrono::steady_clock::now() >= deadline) {
        break;
      }
    }
  }
  TT_THROW("mapped GET completion did not arrive within timeout");
}

void ServerRuntimeCore::submit_mapped_get_ingress(
    uint32_t ring_slot, uint32_t client_token, uint32_t seq, uint32_t key_len,
    const uint32_t *key_words, uint32_t client_dram_base,
    uint32_t logical_page_size, uint32_t output_capacity_bytes,
    uint32_t completion_l1_addr) {
  TT_ASSERT(scalar_get_mode().mapped_ingress);
  TT_ASSERT(key_words != nullptr);
  TT_ASSERT(ring_slot < ring_slots_);
  const uint32_t client_id = protocol::client_id_from_token(client_token);
  TT_ASSERT(client_id < protocol::kMaxClients);
  TT_ASSERT(protocol::client_generation_from_token(client_token) != 0);
  TT_ASSERT(ring_slots_ % protocol::kMaxClients == 0);
  const uint32_t ring_slots_per_client = ring_slots_ / protocol::kMaxClients;
  TT_ASSERT(ring_slots_per_client > 0);
  TT_ASSERT(ring_slot / ring_slots_per_client == client_id);
  TT_ASSERT(key_len >= 1 && key_len <= max_key_len_);
  TT_ASSERT(completion_l1_addr != 0 &&
            completion_l1_addr % sizeof(uint32_t) == 0);

  protocol::GetIngressDescriptor descriptor{};
  const uint32_t generation = ++get_ingress_generation_;
  descriptor.generation_begin = generation;
  descriptor.ring_slot = ring_slot;
  descriptor.packet_bytes =
      (sizeof(protocol::PacketHeader) + sizeof(protocol::GetChunkReqHeader) +
       key_len + 15u) &
      ~15u;

  auto *packet = reinterpret_cast<protocol::PacketHeader *>(descriptor.packet);
  *packet =
      protocol::make_get_ingress_packet_header(seq, key_len, client_token);

  auto *request = reinterpret_cast<protocol::GetChunkReqHeader *>(
      descriptor.packet + sizeof(protocol::PacketHeader));
  request->stream_id_seq = seq;
  request->req_chunk_idx = 0;
  request->key_len = key_len;
  request->client_dram_base = client_dram_base;
  request->logical_page_size = logical_page_size;
  request->output_capacity_bytes = output_capacity_bytes;
  request->completion_l1_addr = completion_l1_addr;
  request->completion_mode = protocol::kGetCompletionCumulativeV2;
  std::memcpy(descriptor.packet + sizeof(protocol::PacketHeader) +
                  sizeof(protocol::GetChunkReqHeader),
              key_words, protocol::kMaxPutKeyBytes);
  descriptor.generation_end = generation;

  const uint32_t descriptor_slot =
      generation % protocol::kGetIngressDescriptorSlots;
  std::memcpy(get_ingress_host_ptr_ +
                  descriptor_slot * sizeof(protocol::GetIngressDescriptor),
              &descriptor, sizeof(descriptor));
  __builtin_ia32_sfence();

  std::array<uint8_t, protocol::kGetIngressDoorbellSize> doorbell{};
  std::memcpy(doorbell.data(), &generation, sizeof(generation));
  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_ASSERT(device != nullptr);
  tt::tt_metal::detail::WriteToDeviceL1(
      device, kGetIngressCore,
      scalar_get_pipeline_.ingress.l1_base +
          protocol::kGetIngressDoorbellOffset,
      std::span<const uint8_t>(doorbell.data(), doorbell.size()),
      tt::CoreType::WORKER);
}

std::pair<uint32_t, uint32_t> ServerRuntimeCore::put_drain_progress() {
  check_ready();
  throw_if_tiering_worker_failed();

  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr,
           "put_drain_progress: server_submesh has no chip at (0,0)");
  std::array<uint8_t, protocol::kPutCommitCounterRegionSize> counters{};
  tt::tt_metal::detail::ReadFromDeviceL1(
      device, kServerCore, server_l1_base_ + commit_issued_offset_,
      std::span<uint8_t>(counters.data(), counters.size()),
      tt::CoreType::WORKER,
      /*barrier_scoped_to_core=*/true);

  uint32_t issued = 0;
  uint32_t committed = 0;
  std::memcpy(&issued, counters.data(), sizeof(issued));
  std::memcpy(&committed, counters.data() + protocol::kPutCommitIssuedSemSize,
              sizeof(committed));
  return {issued, committed};
}

ServerRuntimeCore::GetAdmissionProgress
ServerRuntimeCore::test_get_admission_progress() const {
  check_ready();
  throw_if_tiering_worker_failed();

  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr,
           "test_get_admission_progress: server_submesh has no chip at (0,0)");

  std::array<uint32_t, protocol::kGetCreditSemSize / sizeof(uint32_t)>
      progress{};
  tt::tt_metal::detail::ReadFromDeviceL1(
      device, kServerCore, server_l1_base_ + get_credit_sem_offset_,
      std::span<uint8_t>(reinterpret_cast<uint8_t *>(progress.data()),
                         sizeof(progress)),
      tt::CoreType::WORKER,
      /*barrier_scoped_to_core=*/true);

  uint32_t ready = test_initial_counter_;
  if (scalar_get_pipeline_.admission.enabled) {
    TT_ASSERT(scalar_get_pipeline_.admission.l1 != nullptr);
    std::array<uint32_t, protocol::kGetAdmissionReadySize / sizeof(uint32_t)>
        ready_words{};
    tt::tt_metal::detail::ReadFromDeviceL1(
        device, kGetAdmissionCore,
        scalar_get_pipeline_.admission.l1_base +
            protocol::kGetAdmissionReadyOffset,
        std::span<uint8_t>(reinterpret_cast<uint8_t *>(ready_words.data()),
                           sizeof(ready_words)),
        tt::CoreType::WORKER,
        /*barrier_scoped_to_core=*/true);
    ready = ready_words[0];
  }

  return GetAdmissionProgress{
      .enabled = scalar_get_pipeline_.admission.enabled,
      .ready = ready,
      .source_done = progress[protocol::kGetAdmissionSourceDoneWord],
      .durable_done = progress[protocol::kGetAdmissionWriteDoneWord],
  };
}

// Snapshot the device-owned issued frontier, then wait until BRISC's committed
// counter reaches it. This is ServerRuntimeCore-internal host/device
// synchronization used by debug APIs and teardown; Client completion uses the
// Fabric COMMIT_BARRIER opcode instead.
void ServerRuntimeCore::drain_inflight_puts() {
  if (!server_workload_running_.load(std::memory_order_acquire)) {
    return;
  }
  throw_if_tiering_worker_failed();
  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr,
           "drain_inflight_puts: server_submesh has no chip at (0,0)");
  const uint32_t commit_issued_addr = server_l1_base_ + commit_issued_offset_;
  const uint32_t commit_done_addr = server_l1_base_ + commit_done_offset_;
  std::array<uint8_t, protocol::kPutCommitIssuedSemSize> issued_buf{};
  std::array<uint8_t, protocol::kPutCommitDoneSemSize> sem_buf{};

  tt::tt_metal::detail::ReadFromDeviceL1(
      device, kServerCore, commit_issued_addr,
      std::span<uint8_t>(issued_buf.data(), issued_buf.size()),
      tt::CoreType::WORKER,
      /*barrier_scoped_to_core=*/true);
  uint32_t target = 0;
  std::memcpy(&target, issued_buf.data(), sizeof(target));

  const auto t0 = std::chrono::steady_clock::now();
  const bool profile_wait = kvcache_profile_;
  const auto deadline = std::chrono::steady_clock::now() +
                        put_commit_drain_timeout(stage_noc_ok_);
  bool drained = false;
  bool waited = false;
  uint32_t observed = 0;
  while (true) {
    throw_if_tiering_worker_failed();
    // Scoped barrier: kServerCore-only read (server writes L1 only on
    // kServerCore). See Cluster::l1_barrier(tt_cxy_pair).
    tt::tt_metal::detail::ReadFromDeviceL1(
        device, kServerCore, commit_done_addr,
        std::span<uint8_t>(sem_buf.data(), sem_buf.size()),
        tt::CoreType::WORKER,
        /*barrier_scoped_to_core=*/true);
    observed = *reinterpret_cast<const uint32_t *>(sem_buf.data());
    if (observed == target ||
        protocol::cumulative_counter_is_ahead(observed, target)) {
      drained = true;
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    waited = true;
    __builtin_ia32_pause();
  }
  if (profile_wait && waited) {
    const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
    drain_wait_ns_.fetch_add(ns, std::memory_order_relaxed);
    drain_wait_count_.fetch_add(1, std::memory_order_relaxed);
  }
  TT_FATAL(drained,
           "drain_inflight_puts: commit_done did not reach {} within timeout "
           "(BRISC stalled?)",
           target);
}

// Host-direct debug/test read: resolve catalog metadata from the device
// directory, then return bytes from the recorded tier. Not a production path
// (the wire Client::get is). See
// ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §6.
std::optional<std::vector<uint8_t>>
ServerRuntimeCore::get(const std::string &key) {
  auto operation_lock = acquire_host_operation();
  check_ready();
  drain_inflight_puts(); // every issued PUT's directory write has landed after
                         // this

  // Retry not-found snapshots so a host debug read overlapping an asynchronous
  // device update does not report a transient miss. Callers must not run this
  // diagnostic concurrently with requests.
  auto *dir_dev = dram_directory_buf_->get_device_buffer(MeshCoordinate(0, 0));
  TT_FATAL(dir_dev != nullptr, "ServerRuntimeCore::get: directory MeshBuffer "
                               "has no device buffer at (0,0)");
  const size_t dir_stride = static_cast<size_t>(dir_dev->aligned_page_size());
  std::vector<uint8_t> dir_all(static_cast<size_t>(dir_dev->size()), 0);
  int found_slot = -1;
  uint32_t total_len = 0;
  uint32_t value_byte_off =
      0; // sub-page packing: the value's byte offset in the heap
  uint32_t location = protocol::kLocationDevice;
  for (uint32_t attempt = 0;; ++attempt) {
    tt::tt_metal::detail::ReadFromBuffer(*dir_dev, dir_all.data());
    found_slot = -1;
    for (uint32_t i = 0; i < max_keys_; ++i) {
      const uint8_t *e = dir_all.data() + dir_stride * i;
      const uint32_t valid = *reinterpret_cast<const uint32_t *>(e + 0);
      const uint32_t key_len = *reinterpret_cast<const uint32_t *>(e + 4);
      if (valid != 1 || key_len != key.size()) {
        continue;
      }
      if (std::memcmp(e + protocol::kDirEntryHeaderBytes, key.data(),
                      key_len) != 0) {
        continue;
      }
      found_slot = static_cast<int>(i);
      total_len = *reinterpret_cast<const uint32_t *>(e + 8);
      value_byte_off = *reinterpret_cast<const uint32_t *>(
          e + protocol::kEntryDeviceValueByteOffset);
      location = *reinterpret_cast<const uint32_t *>(
          e + protocol::kEntryLocationByteOffset);
      break;
    }
    if (found_slot >= 0 || attempt >= kDirReadMaxRetries) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(poll_us_));
  }
  if (found_slot < 0) {
    return std::nullopt;
  }
  if (location == protocol::kLocationHost) {
    // Host-tier value: return the host RAM copy directly, no device read. A
    // missing host copy (mid-move / orphan) reads as a miss.
    std::lock_guard<std::mutex> lk(host_values_mutex_);
    auto it = host_values_.find(static_cast<uint32_t>(found_slot));
    if (it == host_values_.end()) {
      return std::nullopt;
    }
    const uint8_t *bytes = host_value_bytes(it->second);
    return std::vector<uint8_t>(bytes, bytes + it->second.len);
  }

  // Device-tier value: targeted read of just this value's heap units (see
  // read_value_units).
  return read_value_units(value_byte_off, total_len);
}

// Host-direct debug/test presence check: found iff a live directory entry holds
// this key.
bool ServerRuntimeCore::exists(const std::string &key) {
  auto operation_lock = acquire_host_operation();
  check_ready();
  drain_inflight_puts();
  auto *dir_dev = dram_directory_buf_->get_device_buffer(MeshCoordinate(0, 0));
  TT_FATAL(dir_dev != nullptr, "ServerRuntimeCore::exists: directory "
                               "MeshBuffer has no device buffer at (0,0)");
  const size_t dir_stride = static_cast<size_t>(dir_dev->aligned_page_size());
  std::vector<uint8_t> dir_all(static_cast<size_t>(dir_dev->size()), 0);
  tt::tt_metal::detail::ReadFromBuffer(*dir_dev, dir_all.data());
  for (uint32_t i = 0; i < max_keys_; ++i) {
    const uint8_t *e = dir_all.data() + dir_stride * i;
    if (*reinterpret_cast<const uint32_t *>(e + 0) != 1) {
      continue;
    }
    if (*reinterpret_cast<const uint32_t *>(e + 4) != key.size()) {
      continue;
    }
    if (std::memcmp(e + protocol::kDirEntryHeaderBytes, key.data(),
                    key.size()) == 0) {
      return true;
    }
  }
  return false;
}

CacheState ServerRuntimeCore::cache_state(const std::string &key) {
  auto operation_lock = acquire_host_operation();
  check_ready();
  drain_inflight_puts();

  auto *dir_dev = dram_directory_buf_->get_device_buffer(MeshCoordinate(0, 0));
  TT_FATAL(dir_dev != nullptr, "ServerRuntimeCore::cache_state: directory "
                               "MeshBuffer has no device buffer at (0,0)");
  const size_t dir_stride = static_cast<size_t>(dir_dev->aligned_page_size());
  std::vector<uint8_t> dir_all(static_cast<size_t>(dir_dev->size()), 0);
  tt::tt_metal::detail::ReadFromBuffer(*dir_dev, dir_all.data());

  uint32_t slot = 0;
  uint32_t total_len = 0;
  uint32_t location = protocol::kLocationHost;
  bool found = false;
  for (uint32_t i = 0; i < max_keys_; ++i) {
    const uint8_t *entry = dir_all.data() + dir_stride * i;
    if (*reinterpret_cast<const uint32_t *>(entry) != 1 ||
        *reinterpret_cast<const uint32_t *>(entry + 4) != key.size()) {
      continue;
    }
    if (std::memcmp(entry + protocol::kDirEntryHeaderBytes, key.data(),
                    key.size()) == 0) {
      found = true;
      slot = i;
      total_len = *reinterpret_cast<const uint32_t *>(entry + 8);
      location = *reinterpret_cast<const uint32_t *>(
          entry + protocol::kEntryLocationByteOffset);
      break;
    }
  }
  if (!found) {
    return {};
  }

  std::lock_guard<std::mutex> lk(host_values_mutex_);
  const auto host_it = host_values_.find(slot);
  return CacheState{
      .found = true,
      .device_resident = location != protocol::kLocationHost,
      .host_backed =
          host_it != host_values_.end() && host_it->second.len == total_len,
  };
}

// Host-direct debug/test remove uses the same BRISC-owned directory path as
// wire REMOVE so the hash index and buddy allocator stay synchronized with the
// authoritative directory entry.
bool ServerRuntimeCore::remove(const std::string &key) {
  auto operation_lock = acquire_host_operation();
  check_ready();
  drain_inflight_puts();
  TT_FATAL(!key.empty() && key.size() <= max_key_len_,
           "ServerRuntimeCore::remove key length is out of range");

  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr,
           "ServerRuntimeCore::remove: server_submesh has no chip at (0,0)");

  std::vector<uint8_t> request(protocol::kDirReqMailboxSize, 0);
  const uint32_t op = static_cast<uint32_t>(protocol::OpCode::REMOVE);
  const uint32_t key_len = static_cast<uint32_t>(key.size());
  std::memcpy(request.data(), &op, sizeof(op));
  std::memcpy(request.data() + sizeof(uint32_t), &key_len, sizeof(key_len));
  std::memcpy(request.data() + protocol::kDirReqMailboxHeaderBytes, key.data(),
              key.size());

  std::array<uint8_t, protocol::kDirReqSemSize> req_sem{};
  tt::tt_metal::detail::ReadFromDeviceL1(
      device, kServerCore, server_l1_base_ + dir_req_sem_offset_,
      std::span<uint8_t>(req_sem.data(), req_sem.size()), tt::CoreType::WORKER,
      /*barrier_scoped_to_core=*/true);
  uint32_t issued = 0;
  std::memcpy(&issued, req_sem.data(), sizeof(issued));
  ++issued;
  std::memcpy(req_sem.data(), &issued, sizeof(issued));

  tt::tt_metal::detail::WriteToDeviceL1(
      device, kServerCore, server_l1_base_ + dir_req_mailbox_offset_,
      std::span<const uint8_t>(request.data(), request.size()),
      tt::CoreType::WORKER);
  tt::tt_metal::detail::WriteToDeviceL1(
      device, kServerCore, server_l1_base_ + dir_req_sem_offset_,
      std::span<const uint8_t>(req_sem.data(), req_sem.size()),
      tt::CoreType::WORKER);

  std::array<uint8_t, protocol::kDirDoneSemSize> done_sem{};
  const auto deadline =
      std::chrono::steady_clock::now() + kHostCompletionTimeout;
  uint32_t completed = 0;
  do {
    throw_if_tiering_worker_failed();
    tt::tt_metal::detail::ReadFromDeviceL1(
        device, kServerCore, server_l1_base_ + dir_done_sem_offset_,
        std::span<uint8_t>(done_sem.data(), done_sem.size()),
        tt::CoreType::WORKER,
        /*barrier_scoped_to_core=*/true);
    std::memcpy(&completed, done_sem.data(), sizeof(completed));
    if (completed == issued) {
      break;
    }
    TT_FATAL(std::chrono::steady_clock::now() < deadline,
             "ServerRuntimeCore::remove timed out waiting for directory "
             "request {} (observed {})",
             issued, completed);
    std::this_thread::sleep_for(std::chrono::microseconds(poll_us_));
  } while (true);

  std::array<uint8_t, protocol::kDirRespMailboxSize> response{};
  tt::tt_metal::detail::ReadFromDeviceL1(
      device, kServerCore, server_l1_base_ + dir_resp_mailbox_offset_,
      std::span<uint8_t>(response.data(), response.size()),
      tt::CoreType::WORKER,
      /*barrier_scoped_to_core=*/true);
  uint32_t status = 0;
  std::memcpy(&status, response.data(), sizeof(status));
  const bool removed = (status & protocol::kStatusFlagRemoved) != 0;
  return removed;
}

uint64_t ServerRuntimeCore::moves_completed() const {
  throw_if_tiering_worker_failed();
  return moves_completed_.load(std::memory_order_acquire);
}

MoveProfile ServerRuntimeCore::move_profile() const {
  return MoveProfile{
      .to_host_count = move_to_host_count_.load(std::memory_order_relaxed),
      .to_host_worker_ns =
          move_to_host_worker_ns_.load(std::memory_order_relaxed),
      .to_host_complete_ns =
          move_to_host_complete_ns_.load(std::memory_order_relaxed),
      .to_host_read_calls =
          move_to_host_read_calls_.load(std::memory_order_relaxed),
      .to_host_read_bytes =
          move_to_host_read_bytes_.load(std::memory_order_relaxed),
      .to_host_read_passes =
          move_to_host_read_passes_.load(std::memory_order_relaxed),
      .to_host_read_retries =
          move_to_host_read_retries_.load(std::memory_order_relaxed),
      .to_host_read_io_ns =
          move_to_host_read_io_ns_.load(std::memory_order_relaxed),
      .to_host_scatter_ns =
          move_to_host_scatter_ns_.load(std::memory_order_relaxed),
      .to_host_device_pushes =
          move_to_host_device_pushes_.load(std::memory_order_relaxed),
      .to_device_count = move_to_device_count_.load(std::memory_order_relaxed),
      .to_device_worker_ns =
          move_to_device_worker_ns_.load(std::memory_order_relaxed),
      .to_device_complete_ns =
          move_to_device_complete_ns_.load(std::memory_order_relaxed),
  };
}

AllocatorProfile ServerRuntimeCore::allocator_profile() const {
  check_ready();
  throw_if_tiering_worker_failed();

  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr,
           "allocator_profile: server_submesh has no chip at (0,0)");
  std::array<uint32_t, protocol::kAllocatorProfileWords> words{};
  tt::tt_metal::detail::ReadFromDeviceL1(
      device, kServerCore, server_l1_base_ + allocator_profile_offset_,
      std::span<uint8_t>(reinterpret_cast<uint8_t *>(words.data()),
                         sizeof(words)),
      tt::CoreType::WORKER,
      /*barrier_scoped_to_core=*/true);

  const auto pair_u64 = [&words](uint32_t low_word, uint32_t high_word) {
    return static_cast<uint64_t>(words[low_word]) |
           (static_cast<uint64_t>(words[high_word]) << 32);
  };
  const int clock_mhz = server_submesh_->get_clock_rate_mhz();
  TT_FATAL(
      clock_mhz > 0,
      "allocator_profile: server_submesh AI clock must be positive, got {} MHz",
      clock_mhz);

  return AllocatorProfile{
      .allocation_calls = words[protocol::kAllocatorProfileAllocationCallsWord],
      .allocation_successes =
          words[protocol::kAllocatorProfileAllocationSuccessesWord],
      .allocation_failures =
          words[protocol::kAllocatorProfileAllocationFailuresWord],
      .victim_scans = words[protocol::kAllocatorProfileVictimScansWord],
      .victims_evicted = words[protocol::kAllocatorProfileVictimsEvictedWord],
      .lease_calls = words[protocol::kAllocatorProfileLeaseCallsWord],
      .lease_successes = words[protocol::kAllocatorProfileLeaseSuccessesWord],
      .lease_failures = words[protocol::kAllocatorProfileLeaseFailuresWord],
      .lease_victim_scans =
          words[protocol::kAllocatorProfileLeaseVictimScansWord],
      .lease_victims_evicted =
          words[protocol::kAllocatorProfileLeaseVictimsEvictedWord],
      .scan_cycles = pair_u64(protocol::kAllocatorProfileScanCyclesLowWord,
                              protocol::kAllocatorProfileScanCyclesHighWord),
      .lease_scan_cycles =
          pair_u64(protocol::kAllocatorProfileLeaseScanCyclesLowWord,
                   protocol::kAllocatorProfileLeaseScanCyclesHighWord),
      .allocation_cycles =
          pair_u64(protocol::kAllocatorProfileAllocationCyclesLowWord,
                   protocol::kAllocatorProfileAllocationCyclesHighWord),
      .clock_mhz = static_cast<uint32_t>(clock_mhz),
  };
}

std::pair<uint32_t, uint32_t> ServerRuntimeCore::pool_stats() {
  std::lock_guard<std::mutex> lk(host_values_mutex_);
  uint32_t free_units = 0;
  for (const auto &[unit, units] : pool_free_) {
    (void)unit;
    free_units += units;
  }
  uint32_t resident = 0;
  for (const auto &[slot, hv] : host_values_) {
    (void)slot;
    resident += (hv.pool_units > 0) ? 1u : 0u;
  }
  return {free_units, resident};
}

void ServerRuntimeCore::flush() {
  auto operation_lock = acquire_host_operation();
  check_ready();
  drain_inflight_puts();
}

} // namespace kvcache_manager::detail::server_runtime
