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

void ServerRuntimeCore::stage_value_device_to_host(uint32_t slot,
                                                   uint32_t value_byte_off,
                                                   uint32_t total_len) {
  // Fallback write-through copies the value's heap units into host memory.
  const uint32_t U = protocol::kMaxChunkValueBytes;
  const uint32_t units = protocol::chunk_count_for(total_len);

  std::optional<uint32_t> pool_unit;
  {
    std::lock_guard<std::mutex> lk(host_values_mutex_);
    drop_host_value(slot);
    pool_unit = pool_alloc(units);
  }

  // Read outside the lock, as the heap path always has: the reserved extent is
  // nobody else's and the map entry is not published until the bytes are in, so
  // a concurrent host-direct read either misses the slot or sees the finished
  // value -- never a partial one.
  ValueReadProfile read_profile;
  ValueReadProfile *profile = kvcache_profile_ ? &read_profile : nullptr;
  std::vector<uint8_t> bytes;
  if (pool_unit.has_value()) {
    uint8_t *dst = pool_host_ptr_ + static_cast<size_t>(*pool_unit) * U;
    read_value_units_into(
        std::span<uint8_t>(dst, static_cast<size_t>(units) * U), value_byte_off,
        total_len, profile);
    // The device reads these bytes over PCIe; order them before anything
    // publishes the eviction.
    __builtin_ia32_sfence();
  } else {
    bytes.resize(static_cast<size_t>(units) * U);
    read_value_units_into(std::span<uint8_t>(bytes), value_byte_off, total_len,
                          profile);
    bytes.resize(total_len);
  }
  if (profile != nullptr) {
    move_to_host_read_calls_.fetch_add(read_profile.calls,
                                       std::memory_order_relaxed);
    move_to_host_read_bytes_.fetch_add(read_profile.bytes,
                                       std::memory_order_relaxed);
    move_to_host_read_passes_.fetch_add(read_profile.passes,
                                        std::memory_order_relaxed);
    move_to_host_read_retries_.fetch_add(read_profile.retries,
                                         std::memory_order_relaxed);
    move_to_host_read_io_ns_.fetch_add(read_profile.io_ns,
                                       std::memory_order_relaxed);
    move_to_host_scatter_ns_.fetch_add(read_profile.scatter_ns,
                                       std::memory_order_relaxed);
  }

  std::lock_guard<std::mutex> lk(host_values_mutex_);
  HostValue &hv = host_values_[slot];
  hv.len = total_len;
  hv.pool_unit = pool_unit.value_or(0);
  hv.pool_units = pool_unit.has_value() ? units : 0;
  hv.generation = 0;
  hv.heap = std::move(bytes);
}

ServerRuntimeCore::DevicePushDestination
ServerRuntimeCore::reserve_device_push_destination(uint32_t slot,
                                                   uint32_t total_len) {
  const uint32_t U = protocol::kMaxChunkValueBytes;
  const uint32_t units = protocol::chunk_count_for(total_len);
  std::lock_guard<std::mutex> lk(host_values_mutex_);
  drop_host_value(slot);
  DevicePushDestination destination;
  destination.pool_unit = pool_alloc(units);
  if (destination.pool_unit.has_value()) {
    destination.read_off = pool_read_off_ + *destination.pool_unit * U;
    destination.generation =
        publish_pool_meta(slot, destination.read_off, total_len);
  }
  return destination;
}

void ServerRuntimeCore::publish_device_pushed_value(
    uint32_t slot, uint32_t total_len,
    const DevicePushDestination &destination) {
  const uint32_t units = protocol::chunk_count_for(total_len);
  std::vector<uint8_t> bytes;
  if (!destination.pool_unit.has_value()) {
    bytes.assign(stage_host_ptr_, stage_host_ptr_ + total_len);
  }

  std::lock_guard<std::mutex> lk(host_values_mutex_);
  HostValue &hv = host_values_[slot];
  hv.len = total_len;
  hv.pool_unit = destination.pool_unit.value_or(0);
  hv.pool_units = destination.pool_unit.has_value() ? units : 0;
  hv.generation = destination.generation;
  hv.heap = std::move(bytes);
}

// Publish the count of value chunks staged so far, which is the BRISC
// host-stream loop's gate.
void ServerRuntimeCore::publish_stage_progress(IDevice *device,
                                               uint32_t stage_progress_addr,
                                               uint32_t staged_chunks) {
  std::array<uint8_t, protocol::kStageProgressSize> buf{};
  std::memcpy(buf.data(), &staged_chunks, sizeof(uint32_t));
  tt::tt_metal::detail::WriteToDeviceL1(
      device, kServerCore, stage_progress_addr,
      std::span<const uint8_t>(buf.data(), buf.size()), tt::CoreType::WORKER);
}

// Return an in-place mapped offset, or zero when the value must use the staging
// window.
uint32_t ServerRuntimeCore::promote_read_off(uint32_t slot) {
  std::lock_guard<std::mutex> lk(host_values_mutex_);
  auto it = host_values_.find(slot);
  if (it == host_values_.end() || it->second.pool_units == 0) {
    return 0;
  }
  return pool_read_off_ + it->second.pool_unit * protocol::kMaxChunkValueBytes;
}

// Prepare host backing for explicit admission or a host-served GET. The mapped
// path exposes a host offset; the fallback coalesces host-to-device writes by
// DRAM bank. See ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §6.
void ServerRuntimeCore::stage_value_host_to_device(uint32_t slot,
                                                   uint32_t value_byte_off,
                                                   uint32_t total_len,
                                                   uint32_t stage_read_off) {
  if (stage_noc_ok_) {
    // value_byte_off is unused on this path — the server derives the heap
    // location from commit_req. A pool-resident value is already NoC-readable
    // in place; its extent remains reserved until promote_done. Heap-resident
    // values still need one contiguous staging copy.
    (void)value_byte_off;
    if (stage_read_off != 0) {
      return;
    }
    const auto tg0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lk(host_values_mutex_);
      auto it = host_values_.find(slot);
      TT_FATAL(it != host_values_.end(),
               "stage_value_host_to_device: slot {} not in host_values_", slot);
      TT_FATAL(it->second.len == total_len,
               "stage_value_host_to_device: slot {} host_values_ size {} != "
               "total_len {}",
               slot, it->second.len, total_len);
      TT_FATAL(total_len <= stage_bytes_,
               "stage_value_host_to_device: total_len {} exceeds staging "
               "capacity {}",
               total_len, stage_bytes_);
      std::memcpy(stage_host_ptr_, host_value_bytes(it->second), total_len);
      // Order the staged bytes before the commit_req that releases the server
      // to read them.
      __builtin_ia32_sfence();
    }
    if (kvcache_profile_) {
      const auto tg1 = std::chrono::steady_clock::now();
      // Book the staging copy under promote_dma_ns_ (the copy phase).
      promote_dma_ns_.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(tg1 - tg0)
              .count(),
          std::memory_order_relaxed);
    }
    return;
  }
  // Without a usable NoC mapping, copy host_values_[slot] into the device heap
  // at value_byte_off, coalesced into one PCIe write per DRAM bank.
  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr,
           "stage_value_host_to_device: server_submesh has no chip at (0,0)");
  auto *value_dev = dram_value_buf_->get_device_buffer(MeshCoordinate(0, 0));
  TT_FATAL(value_dev != nullptr, "stage_value_host_to_device: value MeshBuffer "
                                 "has no device buffer at (0,0)");
  const uint32_t U = protocol::kMaxChunkValueBytes;
  const uint32_t num_banks =
      device->allocator()->get_num_banks(BufferType::DRAM);
  const uint32_t aligned_page =
      static_cast<uint32_t>(value_dev->aligned_page_size());
  const uint32_t base = static_cast<uint32_t>(value_dev->address());
  const uint32_t base_unit = value_byte_off / U;
  const uint32_t units = protocol::chunk_count_for(total_len);
  std::vector<uint8_t> bytes;
  {
    std::lock_guard<std::mutex> lk(host_values_mutex_);
    auto it = host_values_.find(slot);
    TT_FATAL(it != host_values_.end(),
             "stage_value_host_to_device: slot {} not in host_values_", slot);
    // Copy out so the per-unit PCIe writes run without holding the lock.
    // Without a NoC-mapped host pool, the value is always heap-resident here.
    const HostValue &hv = it->second;
    bytes.assign(host_value_bytes(hv), host_value_bytes(hv) + hv.len);
  }
  TT_FATAL(bytes.size() == total_len,
           "stage_value_host_to_device: slot {} host_values_ size {} != "
           "total_len {}",
           slot, bytes.size(), total_len);
  // Same-bank units are contiguous pages, so one write per populated bank is
  // sufficient.
  const uint32_t max_pages_per_bank = (units + num_banks - 1) / num_banks;
  std::vector<uint8_t> scratch(static_cast<size_t>(max_pages_per_bank) *
                               aligned_page);
  const uint32_t r = base_unit % num_banks;
  uint64_t gather_ns = 0; // diagnostic: strided memcpy vs DMA split
                          // (promote_gather_ns_/promote_dma_ns_)
  uint64_t dma_ns = 0;
  for (uint32_t bank = 0; bank < num_banks; ++bank) {
    // Smallest value-local unit index whose heap unit (base_unit + u) falls in
    // this bank; skip the bank if the value has no unit here (value shorter
    // than num_banks units).
    const uint32_t first_u = (bank + num_banks - r) % num_banks;
    if (first_u >= units) {
      continue;
    }
    const uint32_t first_page_addr =
        base + ((base_unit + first_u) / num_banks) * aligned_page;
    const auto tg0 = std::chrono::steady_clock::now();
    uint32_t n_pages = 0;
    for (uint32_t u = first_u; u < units; u += num_banks) {
      const uint32_t dst = n_pages * aligned_page;
      const uint32_t off = u * U;
      const uint32_t n = std::min<uint32_t>(U, total_len - off);
      std::memcpy(scratch.data() + dst, bytes.data() + off, n);
      if (n < aligned_page) {
        // Zero the tail unit's remainder + any aligned_page > U page padding
        // (dead in-page space, never read for this value) so the coalesced
        // write carries no stale bytes.
        std::fill(scratch.begin() + dst + n,
                  scratch.begin() + dst + aligned_page, 0);
      }
      ++n_pages;
    }
    const auto tg1 = std::chrono::steady_clock::now();
    tt::tt_metal::detail::WriteToDeviceDRAMChannel(
        device, static_cast<int>(bank), first_page_addr,
        std::span<const uint8_t>(scratch.data(),
                                 static_cast<size_t>(n_pages) * aligned_page));
    const auto td1 = std::chrono::steady_clock::now();
    gather_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(tg1 - tg0).count();
    dma_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(td1 - tg1).count();
  }
  if (kvcache_profile_) {
    promote_gather_ns_.fetch_add(gather_ns, std::memory_order_relaxed);
    promote_dma_ns_.fetch_add(dma_ns, std::memory_order_relaxed);
  }
}

// Release an in-place pool value immediately, or publish copied staging blocks
// as they become visible. The caller resets progress before posting the
// request. See ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §6.
void ServerRuntimeCore::stage_host_value_for_stream(
    IDevice *device, uint32_t slot, uint32_t total_len,
    uint32_t stage_progress_addr, uint32_t stage_read_off) {
  constexpr uint32_t kStageBlockChunks = 64;
  const uint32_t U = protocol::kMaxChunkValueBytes;
  const uint32_t num_chunks = protocol::chunk_count_for(total_len);
  const auto tg0 = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(host_values_mutex_);
  auto it = host_values_.find(slot);
  TT_FATAL(it != host_values_.end(),
           "stage_host_value_for_stream: slot {} not in host_values_", slot);
  TT_FATAL(it->second.len == total_len,
           "stage_host_value_for_stream: slot {} host_values_ size {} != "
           "total_len {}",
           slot, it->second.len, total_len);
  uint64_t copy_ns = 0; // profile only: the block memcpy vs the WriteToDeviceL1
                        // that publishes it
  uint64_t pub_ns = 0;

  // The caller already told the BRISC which of the two it will be, via
  // commit_req.
  TT_FATAL((stage_read_off != 0) == (it->second.pool_units > 0),
           "stage_host_value_for_stream: slot {} read offset {} disagrees with "
           "its {} units of pool residency",
           slot, stage_read_off, it->second.pool_units);

  if (stage_read_off != 0) {
    // Nothing to copy, so release the gate in one write and let the BRISC read
    // at its own pace.
    const auto tc1 = std::chrono::steady_clock::now();
    publish_stage_progress(device, stage_progress_addr, num_chunks);
    if (kvcache_profile_) {
      pub_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - tc1)
                   .count();
    }
  } else {
    TT_FATAL(
        total_len <= stage_bytes_,
        "stage_host_value_for_stream: total_len {} exceeds staging capacity {}",
        total_len, stage_bytes_);
    const uint8_t *src = host_value_bytes(it->second);
    for (uint32_t base = 0; base < num_chunks; base += kStageBlockChunks) {
      const uint32_t this_block =
          std::min<uint32_t>(kStageBlockChunks, num_chunks - base);
      const uint32_t off = base * U;
      const uint32_t n = std::min<uint32_t>(this_block * U, total_len - off);
      const auto tc0 =
          kvcache_profile_ ? std::chrono::steady_clock::now() : tg0;
      std::memcpy(stage_host_ptr_ + off, src + off, n);
      __builtin_ia32_sfence(); // this block's bytes visible before the progress
                               // that publishes them
      const auto tc1 =
          kvcache_profile_ ? std::chrono::steady_clock::now() : tg0;
      publish_stage_progress(device, stage_progress_addr, base + this_block);
      if (kvcache_profile_) {
        const auto tc2 = std::chrono::steady_clock::now();
        copy_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(tc1 - tc0)
                .count();
        pub_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(tc2 - tc1)
                .count();
      }
    }
  }
  if (kvcache_profile_) {
    const auto tg1 = std::chrono::steady_clock::now();
    promote_dma_ns_.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(tg1 - tg0).count(),
        std::memory_order_relaxed);
    promote_stage_copy_ns_.fetch_add(copy_ns, std::memory_order_relaxed);
    promote_stage_pub_ns_.fetch_add(pub_ns, std::memory_order_relaxed);
  }
}

// Host tiering worker: polls the L1 move_req channel, manages host-value
// storage, and coordinates mapped-host device push/pull or targeted-copy
// fallbacks. The ServerRuntimeCore dispatch loop admits one move at a time
// across Clients, so a single-slot channel suffices. See
// ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §6.
void ServerRuntimeCore::tiering_worker_loop() {
  while (!server_workload_running_.load(std::memory_order_acquire) &&
         !worker_stop_.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  if (worker_stop_.load(std::memory_order_relaxed)) {
    return;
  }

  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  if (device == nullptr) {
    return; // 1x1 mesh / no device: nothing to service.
  }

  // [move_req_sem | move_req slot] are contiguous in L1, so one scoped read
  // observes both. The BRISC reserves promotion heap space and carries the
  // chosen offset in move_req; this worker never writes shared allocator state.
  const uint32_t move_req_sem_addr = server_l1_base_ + move_req_sem_offset_;
  const uint32_t commit_req_sem_addr = server_l1_base_ + commit_req_sem_offset_;
  const uint32_t commit_req_addr = server_l1_base_ + commit_req_offset_;
  const uint32_t move_processed_addr = server_l1_base_ + move_processed_offset_;
  const uint32_t promote_done_addr = server_l1_base_ + promote_done_offset_;
  const uint32_t evict_ack_sem_addr = server_l1_base_ + evict_ack_sem_offset_;
  const uint32_t evict_ack_addr = server_l1_base_ + evict_ack_offset_;
  const uint32_t evict_done_addr = server_l1_base_ + evict_done_offset_;
  constexpr uint32_t kPollSemOff = 0;
  constexpr uint32_t kPollReqOff = protocol::kMoveReqSemSize;
  uint32_t seen = test_initial_counter_;          // move_req_sem observed
  uint32_t commit_issued = test_initial_counter_; // commit_req_sem published
  uint32_t evict_issued =
      test_initial_counter_; // device-push destinations published
  std::vector<uint8_t> poll_buf(
      protocol::kMoveReqSemSize + protocol::kMoveReqSlotBytes, 0);
  // Keep the worker hot across the next high-load PUT transfer, then return to
  // the configured idle backoff. This removes scheduler wake-up jitter from
  // bursts without polling PCIe forever at idle.
  constexpr auto kHotPollWindow = std::chrono::milliseconds(2);
  auto hot_poll_until = std::chrono::steady_clock::now() + kHotPollWindow;

  while (!worker_stop_.load(std::memory_order_relaxed)) {
    // Diagnostic timing for every host-source operation.
    uint64_t sem_ns = 0, commit_ns = 0;

    // The BRISC writes the slot, fences, then inc's the sem, so a snapshot
    // observing the new sem has a valid request and its already-reserved value
    // offset.
    auto tphase = std::chrono::steady_clock::now();
    // The channel lives only on kServerCore, so a core-scoped barrier is
    // sufficient.
    tt::tt_metal::detail::ReadFromDeviceL1(
        device, kServerCore, move_req_sem_addr,
        std::span<uint8_t>(poll_buf.data(), poll_buf.size()),
        tt::CoreType::WORKER,
        /*barrier_scoped_to_core=*/true);
    sem_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::steady_clock::now() - tphase)
                 .count();
    const uint32_t posted =
        *reinterpret_cast<const uint32_t *>(poll_buf.data() + kPollSemOff);
    if (posted == seen) {
      if (std::chrono::steady_clock::now() < hot_poll_until) {
        __builtin_ia32_pause();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(poll_us_));
      }
      continue;
    }
    const auto move_started = tphase;

    const uint8_t *req_bytes = poll_buf.data() + kPollReqOff;
    const uint32_t op = *reinterpret_cast<const uint32_t *>(req_bytes + 0);
    const uint32_t slot = *reinterpret_cast<const uint32_t *>(req_bytes + 4);
    const uint32_t value_byte_off =
        *reinterpret_cast<const uint32_t *>(req_bytes + 8);
    const uint32_t total_len =
        *reinterpret_cast<const uint32_t *>(req_bytes + 12);
    const uint32_t key_len =
        *reinterpret_cast<const uint32_t *>(req_bytes + 16);

    bool moved_to_host = false;
    bool moved_to_device = false;
    auto worker_finished = move_started;
    if (op == protocol::kMoveOpWriteThrough) {
      // With a usable mapping, BRISC pushes into the host pool or staging
      // window. Otherwise, the worker reads the device value through targeted
      // host copies.
      if (stage_noc_ok_) {
        const auto reserve_started =
            kvcache_profile_ ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
        if (kvcache_profile_) {
          move_to_host_device_pushes_.fetch_add(1, std::memory_order_relaxed);
          write_through_reqread_ns_.fetch_add(sem_ns,
                                              std::memory_order_relaxed);
        }
        const DevicePushDestination destination =
            reserve_device_push_destination(slot, total_len);
        if (kvcache_profile_) {
          write_through_reserve_ns_.fetch_add(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - reserve_started)
                  .count(),
              std::memory_order_relaxed);
        }
        std::array<uint8_t, protocol::kEvictAckSize> ack{};
        std::memcpy(ack.data() +
                        protocol::kEvictAckReadOffWord * sizeof(uint32_t),
                    &destination.read_off, sizeof(uint32_t));
        std::memcpy(ack.data() +
                        protocol::kEvictAckGenerationWord * sizeof(uint32_t),
                    &destination.generation, sizeof(uint32_t));
        ++evict_issued;
        std::array<uint8_t, protocol::kEvictAckSemSize> ack_sem{};
        std::memcpy(ack_sem.data(), &evict_issued, sizeof(uint32_t));
        const auto ack_started = kvcache_profile_
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
        tt::tt_metal::detail::WriteToDeviceL1(
            device, kServerCore, evict_ack_addr,
            std::span<const uint8_t>(ack.data(), ack.size()),
            tt::CoreType::WORKER);
        tt::tt_metal::detail::WriteToDeviceL1(
            device, kServerCore, evict_ack_sem_addr,
            std::span<const uint8_t>(ack_sem.data(), ack_sem.size()),
            tt::CoreType::WORKER);
        if (kvcache_profile_) {
          write_through_ack_ns_.fetch_add(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - ack_started)
                  .count(),
              std::memory_order_relaxed);
        }

        std::array<uint8_t, protocol::kEvictDoneSemSize> done_buf{};
        const auto push_wait_started =
            kvcache_profile_ ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
        while (!worker_stop_.load(std::memory_order_relaxed)) {
          tt::tt_metal::detail::ReadFromDeviceL1(
              device, kServerCore, evict_done_addr,
              std::span<uint8_t>(done_buf.data(), done_buf.size()),
              tt::CoreType::WORKER,
              /*barrier_scoped_to_core=*/true);
          if (*reinterpret_cast<const uint32_t *>(done_buf.data()) ==
              evict_issued) {
            break;
          }
          // The scoped read itself spaces polls on this known-active mapped
          // transfer.
          __builtin_ia32_pause();
        }
        if (kvcache_profile_) {
          write_through_push_wait_ns_.fetch_add(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - push_wait_started)
                  .count(),
              std::memory_order_relaxed);
        }
        if (worker_stop_.load(std::memory_order_relaxed)) {
          if (destination.pool_unit.has_value()) {
            std::lock_guard<std::mutex> lk(host_values_mutex_);
            invalidate_pool_meta(slot);
            pool_release(*destination.pool_unit,
                         protocol::chunk_count_for(total_len));
          }
          return;
        }
        const auto publish_started =
            kvcache_profile_ ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
        __builtin_ia32_lfence();
        publish_device_pushed_value(slot, total_len, destination);
        if (kvcache_profile_) {
          write_through_publish_ns_.fetch_add(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - publish_started)
                  .count(),
              std::memory_order_relaxed);
        }
      } else {
        stage_value_device_to_host(slot, value_byte_off, total_len);
      }
      moved_to_host = true;
      worker_finished = std::chrono::steady_clock::now();
    } else if (op == protocol::kMoveOpRemoveBacking) {
      std::lock_guard<std::mutex> lk(host_values_mutex_);
      drop_host_value(slot);
    } else if (op == protocol::kMoveOpDevicePromote ||
               op == protocol::kMoveOpHostServe) {
      // Explicit admission stages before publication; mapped host-serve
      // publishes first and streams blocks. The BRISC owns admission and the
      // host copy remains authoritative. See
      // ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §6.
      const bool host_serve = (op == protocol::kMoveOpHostServe);
      const bool fused = host_serve && stage_noc_ok_;
      const uint32_t new_off = value_byte_off;

      // Build commit_req {slot, new_off, total_len, key_len, key} + its
      // absolute sem count (this worker is the sole writer, so an absolute
      // write beats an atomic inc; the BRISC reads it against a local
      // expected).
      std::vector<uint8_t> creq(protocol::kCommitReqSlotBytes, 0);
      std::memcpy(creq.data() + 0, &slot, sizeof(uint32_t));
      std::memcpy(creq.data() + 4, &new_off, sizeof(uint32_t));
      std::memcpy(creq.data() + 8, &total_len, sizeof(uint32_t));
      std::memcpy(creq.data() + 12, &key_len, sizeof(uint32_t));
      const uint32_t copy_key =
          std::min<uint32_t>(key_len, protocol::kMaxKeyBytesDefault);
      std::memcpy(creq.data() + protocol::kCommitReqHeaderBytes,
                  req_bytes + protocol::kMoveReqHeaderBytes, copy_key);
      // Where the BRISC reads the value from, decided once here and handed to
      // the staging call. Both cache admission and host serve read
      // pool-resident values in place.
      const uint32_t stage_read_off = promote_read_off(slot);
      std::memcpy(creq.data() + protocol::kCommitReqStageReadOffByteOffset,
                  &stage_read_off, sizeof(uint32_t));
      ++commit_issued;
      std::array<uint8_t, protocol::kCommitReqSemSize> csem{};
      std::memcpy(csem.data(), &commit_issued, sizeof(uint32_t));
      auto post_commit_req = [&]() {
        tt::tt_metal::detail::WriteToDeviceL1(
            device, kServerCore, commit_req_addr,
            std::span<const uint8_t>(creq.data(), creq.size()),
            tt::CoreType::WORKER);
        tt::tt_metal::detail::WriteToDeviceL1(
            device, kServerCore, commit_req_sem_addr,
            std::span<const uint8_t>(csem.data(), csem.size()),
            tt::CoreType::WORKER);
      };

      if (fused) {
        const uint32_t stage_progress_addr =
            server_l1_base_ + stage_progress_offset_;
        // Reset stage_progress to 0 BEFORE the commit_req that releases the
        // BRISC, so it observes THIS operation's count (never a stale larger
        // value from a previous promote). Ordered PCIe writes (reset ->
        // commit_req slot -> sem) give the BRISC a valid new_off + zeroed
        // progress.
        std::array<uint8_t, protocol::kStageProgressSize> zprog{};
        tt::tt_metal::detail::WriteToDeviceL1(
            device, kServerCore, stage_progress_addr,
            std::span<const uint8_t>(zprog.data(), zprog.size()),
            tt::CoreType::WORKER);
        tphase = std::chrono::steady_clock::now();
        post_commit_req(); // early: the BRISC reads new_off and starts the fuse
                           // loop
        commit_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - tphase)
                        .count();
        stage_host_value_for_stream(device, slot, total_len,
                                    stage_progress_addr, stage_read_off);
      } else {
        stage_value_host_to_device(slot, new_off, total_len, stage_read_off);
        tphase = std::chrono::steady_clock::now();
        post_commit_req(); // after staging: value durable, the (C) channel may
                           // pull it
        commit_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - tphase)
                        .count();
      }

      if (kvcache_profile_) {
        // reqread = the combined [sem|slot] read; commit = commit_req writes;
        // gather/dma is booked by the staging call.
        promote_count_.fetch_add(1, std::memory_order_relaxed);
        promote_reqread_ns_.fetch_add(sem_ns, std::memory_order_relaxed);
        promote_commit_ns_.fetch_add(commit_ns, std::memory_order_relaxed);
      }

      moved_to_device = true;
      worker_finished = std::chrono::steady_clock::now();
    }

    // The BRISC must finish consuming the staged source before the single move
    // slot is reusable.
    if (moved_to_device) {
      const auto completion_wait_started = std::chrono::steady_clock::now();
      bool used_l1_fallback = false;
      if (stage_noc_ok_) {
        // Delay the L1 fallback probe until the payload's optimistic transfer
        // floor so its PCIe reads do not contend with the bulk pull.
        constexpr uint64_t kCompletionProbeFloorNs = 200'000;
        constexpr uint64_t kBytesPerNsAt80Gbps = 10;
        const uint64_t transfer_floor_ns =
            (static_cast<uint64_t>(total_len) + kBytesPerNsAt80Gbps - 1) /
            kBytesPerNsAt80Gbps;
        const auto l1_probe_start =
            completion_wait_started +
            std::chrono::nanoseconds(
                std::max<uint64_t>(kCompletionProbeFloorNs, transfer_floor_ns));
        std::array<uint8_t, protocol::kPromoteDoneSemSize> done_buf{};
        uint32_t spins = 0;
        while (!worker_stop_.load(std::memory_order_relaxed)) {
          __builtin_ia32_lfence();
          if (*stage_completion_host_ptr_ == commit_issued) {
            break;
          }
          __builtin_ia32_pause();
          if ((++spins & 0xFFu) == 0 &&
              std::chrono::steady_clock::now() >= l1_probe_start) {
            used_l1_fallback = true;
            tt::tt_metal::detail::ReadFromDeviceL1(
                device, kServerCore, promote_done_addr,
                std::span<uint8_t>(done_buf.data(), done_buf.size()),
                tt::CoreType::WORKER,
                /*barrier_scoped_to_core=*/true);
            if (*reinterpret_cast<const uint32_t *>(done_buf.data()) ==
                commit_issued) {
              break;
            }
          }
        }
      } else {
        std::array<uint8_t, protocol::kPromoteDoneSemSize> done_buf{};
        while (!worker_stop_.load(std::memory_order_relaxed)) {
          tt::tt_metal::detail::ReadFromDeviceL1(
              device, kServerCore, promote_done_addr,
              std::span<uint8_t>(done_buf.data(), done_buf.size()),
              tt::CoreType::WORKER,
              /*barrier_scoped_to_core=*/true);
          const uint32_t done =
              *reinterpret_cast<const uint32_t *>(done_buf.data());
          if (done == commit_issued) {
            break;
          }
          std::this_thread::sleep_for(std::chrono::microseconds(poll_us_));
        }
      }
      if (kvcache_profile_) {
        promote_completion_wait_ns_.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - completion_wait_started)
                .count(),
            std::memory_order_relaxed);
        if (used_l1_fallback) {
          promote_completion_l1_fallback_count_.fetch_add(
              1, std::memory_order_relaxed);
        }
      }
      if (worker_stop_.load(std::memory_order_relaxed)) {
        return;
      }
    }
    ++seen;

    // Publish the cumulative processed count so the BRISC knows the single move
    // channel has drained (the channel is reusable only when move_req_sem ==
    // move_processed). Written after the op completes, so "caught up" implies
    // the copy (+ any commit_req) landed.
    std::array<uint8_t, protocol::kMoveProcessedSemSize> proc_buf{};
    std::memcpy(proc_buf.data(), &seen, sizeof(uint32_t));
    const auto processed_started =
        kvcache_profile_ && moved_to_host
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, move_processed_addr,
        std::span<const uint8_t>(proc_buf.data(), proc_buf.size()),
        tt::CoreType::WORKER);
    const auto move_finished = std::chrono::steady_clock::now();
    if (kvcache_profile_ && moved_to_host) {
      write_through_processed_ns_.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              move_finished - processed_started)
              .count(),
          std::memory_order_relaxed);
    }
    hot_poll_until = move_finished + kHotPollWindow;

    if (kvcache_profile_) {
      const uint64_t worker_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(worker_finished -
                                                               move_started)
              .count();
      const uint64_t complete_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(move_finished -
                                                               move_started)
              .count();
      if (moved_to_host) {
        move_to_host_worker_ns_.fetch_add(worker_ns, std::memory_order_relaxed);
        move_to_host_complete_ns_.fetch_add(complete_ns,
                                            std::memory_order_relaxed);
        move_to_host_count_.fetch_add(1, std::memory_order_relaxed);
      } else if (moved_to_device) {
        move_to_device_worker_ns_.fetch_add(worker_ns,
                                            std::memory_order_relaxed);
        move_to_device_complete_ns_.fetch_add(complete_ns,
                                              std::memory_order_relaxed);
        move_to_device_count_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    moves_completed_.fetch_add(1, std::memory_order_release);
  }
}
} // namespace kvcache_manager::detail::server_runtime
