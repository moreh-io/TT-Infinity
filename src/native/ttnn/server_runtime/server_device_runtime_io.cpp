// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#include "server_runtime/server_device_runtime_impl.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "server_runtime/server_runtime_core.hpp"
#include "ttnn/operations/kvcache_manager/common/protocol/layout.hpp"
#include "ttnn/operations/kvcache_manager/common/protocol/packet.hpp"

namespace kvcache_manager::detail::server_runtime {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace {

ServerHostRequestKind host_request_kind(uint32_t op) {
  switch (op) {
  case protocol::kMoveOpWriteThrough:
    return ServerHostRequestKind::WriteThrough;
  case protocol::kMoveOpRemoveBacking:
    return ServerHostRequestKind::RemoveBacking;
  case protocol::kMoveOpDevicePromote:
    return ServerHostRequestKind::PromoteToDevice;
  case protocol::kMoveOpHostServe:
    return ServerHostRequestKind::ServeGet;
  default:
    TT_THROW("ServerRuntimeCore host channel returned unsupported operation {}",
             op);
  }
}

uint32_t read_u32(const uint8_t *bytes, uint32_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes + offset, sizeof(value));
  return value;
}

} // namespace

void ServerRuntimeCore::validate_external_request(
    const ServerHostRequest &request) const {
  TT_FATAL(host_policy_mode_ == HostPolicyMode::External,
           "host request SPI requires external host policy");
  TT_FATAL(external_host_request_.active,
           "ServerRuntimeCore host request is no longer active");
  TT_FATAL(request.sequence == external_host_request_.sequence,
           "ServerRuntimeCore host request sequence {} != active {}",
           request.sequence, external_host_request_.sequence);
  TT_FATAL(request.slot == external_host_request_.slot &&
               request.value_len == external_host_request_.total_len &&
               request.device_value_offset ==
                   external_host_request_.value_byte_off,
           "ServerRuntimeCore host request fields do not match the active "
           "device request");
  TT_FATAL(request.kind == host_request_kind(external_host_request_.op),
           "ServerRuntimeCore host request kind does not match the active "
           "device request");
}

std::optional<ServerHostRequest>
ServerRuntimeCore::poll_external_host_request() {
  TT_FATAL(host_policy_mode_ == HostPolicyMode::External,
           "host request SPI requires external host policy");
  const LifecycleState lifecycle = lifecycle_state_.load();
  TT_FATAL(lifecycle == LifecycleState::Ready ||
               lifecycle == LifecycleState::Closing,
           "host request poll requires a running or closing ServerRuntimeCore");
  TT_FATAL(!external_host_request_.active,
           "complete the active ServerRuntimeCore host request before polling "
           "again");
  if (external_host_io_cancelled_.load(std::memory_order_acquire)) {
    return std::nullopt;
  }
  if (!server_workload_running_.load(std::memory_order_acquire)) {
    return std::nullopt;
  }

  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr,
           "ServerRuntimeCore host request poll has no device at (0,0)");
  const uint32_t move_req_sem_addr = server_l1_base_ + move_req_sem_offset_;
  const auto started = std::chrono::steady_clock::now();
  tt::tt_metal::detail::ReadFromDeviceL1(
      device, kServerCore, move_req_sem_addr,
      std::span<uint8_t>(external_poll_buffer_.data(),
                         external_poll_buffer_.size()),
      tt::CoreType::WORKER,
      /*barrier_scoped_to_core=*/true);
  const uint64_t poll_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - started)
                               .count();

  const uint32_t posted = read_u32(external_poll_buffer_.data(), 0);
  if (posted == external_seen_) {
    return std::nullopt;
  }
  TT_FATAL(posted == external_seen_ + 1,
           "ServerRuntimeCore host request sequence advanced from {} to {} "
           "before the prior slot was completed",
           external_seen_, posted);

  const uint8_t *request_bytes =
      external_poll_buffer_.data() + protocol::kMoveReqSemSize;
  ExternalHostRequestState decoded{
      .active = true,
      .sequence = posted,
      .op = read_u32(request_bytes, 0),
      .slot = read_u32(request_bytes, 4),
      .value_byte_off = read_u32(request_bytes, 8),
      .total_len = read_u32(request_bytes, 12),
      .key_len = read_u32(request_bytes, 16),
  };
  const ServerHostRequestKind kind = host_request_kind(decoded.op);
  TT_FATAL(decoded.slot < max_keys_,
           "ServerRuntimeCore host request slot {} >= max_keys {}",
           decoded.slot, max_keys_);
  TT_FATAL(decoded.key_len <= max_key_len_,
           "ServerRuntimeCore host request key length {} exceeds configured "
           "maximum {}",
           decoded.key_len, max_key_len_);
  if (kind == ServerHostRequestKind::RemoveBacking) {
    TT_FATAL(
        decoded.value_byte_off == 0 && decoded.total_len == 0 &&
            decoded.key_len == 0,
        "remove-backing request must not carry a value offset, length, or key");
  } else {
    TT_FATAL(
        decoded.total_len > 0 && decoded.total_len <= max_value_bytes_,
        "ServerRuntimeCore host request value length {} is outside [1, {}]",
        decoded.total_len, max_value_bytes_);
    const bool key_shape_valid = kind == ServerHostRequestKind::PromoteToDevice
                                     ? decoded.key_len > 0
                                     : decoded.key_len == 0;
    TT_FATAL(key_shape_valid, "ServerRuntimeCore host request carries an "
                              "invalid key length for its operation");

    const uint64_t value_end =
        static_cast<uint64_t>(decoded.value_byte_off) +
        protocol::chunk_aligned_value_bytes(decoded.total_len);
    const bool offset_may_be_unallocated =
        kind == ServerHostRequestKind::ServeGet && decoded.value_byte_off == 0;
    TT_FATAL(offset_may_be_unallocated ||
                 (decoded.value_byte_off % protocol::kMaxChunkValueBytes == 0 &&
                  value_end <= value_heap_bytes_),
             "ServerRuntimeCore host request device value range [{}, {}) is "
             "outside the {}-byte heap",
             decoded.value_byte_off, value_end, value_heap_bytes_);
  }
  decoded.key.assign(request_bytes + protocol::kMoveReqHeaderBytes,
                     request_bytes + protocol::kMoveReqHeaderBytes +
                         decoded.key_len);
  external_host_request_ = std::move(decoded);

  return ServerHostRequest{
      .sequence = external_host_request_.sequence,
      .kind = kind,
      .slot = external_host_request_.slot,
      .value_len = external_host_request_.total_len,
      .device_value_offset = external_host_request_.value_byte_off,
      .poll_ns = poll_ns,
  };
}

ServerReceivedValue
ServerRuntimeCore::receive_external_value(const ServerHostRequest &request,
                                          bool prefer_mapped_lease) {
  validate_external_request(request);
  TT_FATAL(request.kind == ServerHostRequestKind::WriteThrough,
           "receive_external_value requires a write-through request");
  TT_FATAL(external_host_request_.phase ==
               ExternalHostRequestState::Phase::Pending,
           "write-through value transfer has already started");
  external_host_request_.phase =
      ExternalHostRequestState::Phase::TransferInProgress;

  ServerReceivedValue result;
  const uint32_t units = protocol::chunk_count_for(request.value_len);
  std::optional<uint32_t> pool_unit;
  uint32_t read_offset = 0;
  uint32_t generation = 0;

  auto release_reserved_pool = [&] {
    if (!pool_unit.has_value()) {
      return;
    }
    std::lock_guard<std::mutex> lock(host_values_mutex_);
    invalidate_pool_meta(request.slot);
    pool_release(*pool_unit, units);
    pool_unit.reset();
  };
  auto publish_lease = [&]() {
    TT_FATAL(pool_unit.has_value(),
             "mapped host lease requires a reserved pool extent");
    try {
      result.lease = ServerHostLeaseHandle(
          new ServerHostLease(this, request.slot, *pool_unit, units,
                              read_offset, generation, request.value_len));
    } catch (...) {
      release_reserved_pool();
      throw;
    }
    {
      std::lock_guard<std::mutex> lock(host_values_mutex_);
      external_host_leases_.insert(result.lease.get());
      external_pool_units_in_use_ += units;
    }
    pool_unit.reset();
  };

  if (stage_noc_ok_) {
    const auto reserve_started = std::chrono::steady_clock::now();
    if (prefer_mapped_lease) {
      std::lock_guard<std::mutex> lock(host_values_mutex_);
      pool_unit = pool_alloc(units);
      if (pool_unit.has_value()) {
        read_offset =
            pool_read_off_ + *pool_unit * protocol::kMaxChunkValueBytes;
        generation =
            publish_pool_meta(request.slot, read_offset, request.value_len);
      }
    }
    result.reserve_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - reserve_started)
                            .count();

    IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
    TT_FATAL(device != nullptr, "write-through receive has no device at (0,0)");
    const uint32_t evict_ack_sem_addr = server_l1_base_ + evict_ack_sem_offset_;
    const uint32_t evict_ack_addr = server_l1_base_ + evict_ack_offset_;
    const uint32_t evict_done_addr = server_l1_base_ + evict_done_offset_;

    std::array<uint8_t, protocol::kEvictAckSize> ack{};
    std::memcpy(ack.data() + protocol::kEvictAckReadOffWord * sizeof(uint32_t),
                &read_offset, sizeof(uint32_t));
    std::memcpy(ack.data() +
                    protocol::kEvictAckGenerationWord * sizeof(uint32_t),
                &generation, sizeof(uint32_t));
    ++external_evict_issued_;
    std::array<uint8_t, protocol::kEvictAckSemSize> ack_sem{};
    std::memcpy(ack_sem.data(), &external_evict_issued_, sizeof(uint32_t));
    const auto ack_started = std::chrono::steady_clock::now();
    tt::tt_metal::detail::WriteToDeviceL1(device, kServerCore, evict_ack_addr,
                                          std::span<const uint8_t>(ack),
                                          tt::CoreType::WORKER);
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, evict_ack_sem_addr,
        std::span<const uint8_t>(ack_sem), tt::CoreType::WORKER);
    result.ack_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - ack_started)
                        .count();

    std::array<uint8_t, protocol::kEvictDoneSemSize> done{};
    const auto wait_started = std::chrono::steady_clock::now();
    while (!external_host_io_cancelled_.load(std::memory_order_acquire)) {
      tt::tt_metal::detail::ReadFromDeviceL1(
          device, kServerCore, evict_done_addr, std::span<uint8_t>(done),
          tt::CoreType::WORKER,
          /*barrier_scoped_to_core=*/true);
      if (read_u32(done.data(), 0) == external_evict_issued_) {
        break;
      }
      __builtin_ia32_pause();
    }
    result.push_wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - wait_started)
                              .count();
    if (external_host_io_cancelled_.load(std::memory_order_acquire)) {
      release_reserved_pool();
      TT_THROW("ServerRuntimeCore host write-through receive was cancelled");
    }

    __builtin_ia32_lfence();
    result.used_device_push = true;
    if (pool_unit.has_value()) {
      publish_lease();
    } else {
      result.bytes.assign(stage_host_ptr_, stage_host_ptr_ + request.value_len);
    }
    external_host_request_.phase =
        ExternalHostRequestState::Phase::TransferComplete;
    return result;
  }

  ServerRuntimeCore::ValueReadProfile read_profile;
  if (prefer_mapped_lease) {
    std::lock_guard<std::mutex> lock(host_values_mutex_);
    pool_unit = pool_alloc(units);
  }
  if (pool_unit.has_value()) {
    uint8_t *destination = pool_host_ptr_ + static_cast<size_t>(*pool_unit) *
                                                protocol::kMaxChunkValueBytes;
    read_value_units_into(
        std::span<uint8_t>(destination, static_cast<size_t>(units) *
                                            protocol::kMaxChunkValueBytes),
        request.device_value_offset, request.value_len, &read_profile);
    __builtin_ia32_sfence();
    if (pool_meta_host_ptr_ != nullptr) {
      std::lock_guard<std::mutex> lock(host_values_mutex_);
      read_offset = pool_read_off_ + *pool_unit * protocol::kMaxChunkValueBytes;
      generation =
          publish_pool_meta(request.slot, read_offset, request.value_len);
    }
    publish_lease();
  } else {
    result.bytes.resize(static_cast<size_t>(units) *
                        protocol::kMaxChunkValueBytes);
    read_value_units_into(std::span<uint8_t>(result.bytes),
                          request.device_value_offset, request.value_len,
                          &read_profile);
    result.bytes.resize(request.value_len);
  }
  result.read_metrics = {
      .calls = read_profile.calls,
      .bytes = read_profile.bytes,
      .passes = read_profile.passes,
      .retries = read_profile.retries,
      .io_ns = read_profile.io_ns,
      .scatter_ns = read_profile.scatter_ns,
  };
  external_host_request_.phase =
      ExternalHostRequestState::Phase::TransferComplete;
  return result;
}

ServerSupplyMetrics
ServerRuntimeCore::supply_external_value(const ServerHostRequest &request,
                                         const ServerHostLease *lease,
                                         std::span<const uint8_t> heap_bytes) {
  validate_external_request(request);
  TT_FATAL(request.kind == ServerHostRequestKind::PromoteToDevice ||
               request.kind == ServerHostRequestKind::ServeGet,
           "supply_external_value requires a promote or host-serve request");
  TT_FATAL(external_host_request_.phase ==
               ExternalHostRequestState::Phase::Pending,
           "host value supply has already started");
  TT_FATAL((lease != nullptr) != (!heap_bytes.empty()),
           "supply_external_value requires exactly one mapped lease or heap "
           "byte source");
  if (lease != nullptr) {
    TT_FATAL(
        lease->owner() == this,
        "mapped host lease belongs to a different ServerRuntimeCore runtime");
    TT_FATAL(lease->value_len() == request.value_len,
             "mapped host lease length {} != request length {}",
             lease->value_len(), request.value_len);
  } else {
    TT_FATAL(heap_bytes.size() == request.value_len,
             "heap host backing length {} != request length {}",
             heap_bytes.size(), request.value_len);
  }
  external_host_request_.phase =
      ExternalHostRequestState::Phase::TransferInProgress;

  ServerSupplyMetrics metrics;
  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr, "host value supply has no device at (0,0)");
  const bool host_serve = request.kind == ServerHostRequestKind::ServeGet;
  const bool fused = host_serve && stage_noc_ok_;
  const uint32_t stage_read_offset =
      lease == nullptr ? 0 : lease->read_offset();

  std::vector<uint8_t> commit_request(protocol::kCommitReqSlotBytes, 0);
  std::memcpy(commit_request.data() + 0, &request.slot, sizeof(uint32_t));
  std::memcpy(commit_request.data() + 4, &request.device_value_offset,
              sizeof(uint32_t));
  std::memcpy(commit_request.data() + 8, &request.value_len, sizeof(uint32_t));
  std::memcpy(commit_request.data() + 12, &external_host_request_.key_len,
              sizeof(uint32_t));
  std::memcpy(commit_request.data() + protocol::kCommitReqHeaderBytes,
              external_host_request_.key.data(),
              external_host_request_.key.size());
  std::memcpy(commit_request.data() +
                  protocol::kCommitReqStageReadOffByteOffset,
              &stage_read_offset, sizeof(uint32_t));
  ++external_commit_issued_;
  std::array<uint8_t, protocol::kCommitReqSemSize> commit_sem{};
  std::memcpy(commit_sem.data(), &external_commit_issued_, sizeof(uint32_t));
  const uint32_t commit_req_addr = server_l1_base_ + commit_req_offset_;
  const uint32_t commit_req_sem_addr = server_l1_base_ + commit_req_sem_offset_;
  auto post_commit_request = [&] {
    const auto started = std::chrono::steady_clock::now();
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, commit_req_addr,
        std::span<const uint8_t>(commit_request), tt::CoreType::WORKER);
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, commit_req_sem_addr,
        std::span<const uint8_t>(commit_sem), tt::CoreType::WORKER);
    metrics.commit_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - started)
                            .count();
  };

  if (fused) {
    const uint32_t stage_progress_addr =
        server_l1_base_ + stage_progress_offset_;
    std::array<uint8_t, protocol::kStageProgressSize> zero_progress{};
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, stage_progress_addr,
        std::span<const uint8_t>(zero_progress), tt::CoreType::WORKER);
    post_commit_request();

    const uint32_t num_chunks = protocol::chunk_count_for(request.value_len);
    if (lease != nullptr) {
      const auto publish_started = std::chrono::steady_clock::now();
      publish_stage_progress(device, stage_progress_addr, num_chunks);
      metrics.stage_publish_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - publish_started)
              .count();
    } else {
      constexpr uint32_t kStageBlockChunks = 64;
      TT_FATAL(request.value_len <= stage_bytes_,
               "host-serve value exceeds the staging window");
      for (uint32_t base = 0; base < num_chunks; base += kStageBlockChunks) {
        const uint32_t block_chunks =
            std::min<uint32_t>(kStageBlockChunks, num_chunks - base);
        const uint32_t offset = base * protocol::kMaxChunkValueBytes;
        const uint32_t bytes =
            std::min<uint32_t>(block_chunks * protocol::kMaxChunkValueBytes,
                               request.value_len - offset);
        const auto copy_started = std::chrono::steady_clock::now();
        std::memcpy(stage_host_ptr_ + offset, heap_bytes.data() + offset,
                    bytes);
        __builtin_ia32_sfence();
        const auto publish_started = std::chrono::steady_clock::now();
        metrics.stage_copy_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                publish_started - copy_started)
                .count();
        publish_stage_progress(device, stage_progress_addr,
                               base + block_chunks);
        metrics.stage_publish_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - publish_started)
                .count();
      }
    }
    metrics.dma_ns = metrics.stage_copy_ns + metrics.stage_publish_ns;
  } else {
    if (stage_noc_ok_) {
      if (lease == nullptr) {
        TT_FATAL(request.value_len <= stage_bytes_,
                 "promote value exceeds the staging window");
        const auto copy_started = std::chrono::steady_clock::now();
        std::memcpy(stage_host_ptr_, heap_bytes.data(), request.value_len);
        __builtin_ia32_sfence();
        metrics.dma_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - copy_started)
                             .count();
      }
    } else {
      TT_FATAL(lease == nullptr,
               "mapped host lease requires a usable NoC host mapping");
      auto *value_device_buffer =
          dram_value_buf_->get_device_buffer(MeshCoordinate(0, 0));
      TT_FATAL(value_device_buffer != nullptr,
               "value MeshBuffer has no device buffer at (0,0)");
      const uint32_t unit = protocol::kMaxChunkValueBytes;
      const uint32_t num_banks =
          device->allocator()->get_num_banks(BufferType::DRAM);
      const uint32_t aligned_page =
          static_cast<uint32_t>(value_device_buffer->aligned_page_size());
      const uint32_t base_address =
          static_cast<uint32_t>(value_device_buffer->address());
      const uint32_t base_unit = request.device_value_offset / unit;
      const uint32_t units = protocol::chunk_count_for(request.value_len);
      const uint32_t pages_per_bank = (units + num_banks - 1) / num_banks;
      std::vector<uint8_t> scratch(static_cast<size_t>(pages_per_bank) *
                                   aligned_page);
      const uint32_t rotation = base_unit % num_banks;
      for (uint32_t bank = 0; bank < num_banks; ++bank) {
        const uint32_t first_unit = (bank + num_banks - rotation) % num_banks;
        if (first_unit >= units) {
          continue;
        }
        const uint32_t page_address =
            base_address +
            ((base_unit + first_unit) / num_banks) * aligned_page;
        const auto gather_started = std::chrono::steady_clock::now();
        uint32_t pages = 0;
        for (uint32_t local_unit = first_unit; local_unit < units;
             local_unit += num_banks) {
          const uint32_t destination_offset = pages * aligned_page;
          const uint32_t source_offset = local_unit * unit;
          const uint32_t bytes =
              std::min<uint32_t>(unit, request.value_len - source_offset);
          std::memcpy(scratch.data() + destination_offset,
                      heap_bytes.data() + source_offset, bytes);
          std::fill(scratch.begin() + destination_offset + bytes,
                    scratch.begin() + destination_offset + aligned_page, 0);
          ++pages;
        }
        const auto dma_started = std::chrono::steady_clock::now();
        metrics.gather_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(dma_started -
                                                                 gather_started)
                .count();
        tt::tt_metal::detail::WriteToDeviceDRAMChannel(
            device, static_cast<int>(bank), page_address,
            std::span<const uint8_t>(
                scratch.data(), static_cast<size_t>(pages) * aligned_page));
        metrics.dma_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - dma_started)
                              .count();
      }
    }
    post_commit_request();
  }

  const auto wait_started = std::chrono::steady_clock::now();
  const uint32_t promote_done_addr = server_l1_base_ + promote_done_offset_;
  if (stage_noc_ok_) {
    constexpr uint64_t kCompletionProbeFloorNs = 200'000;
    constexpr uint64_t kBytesPerNsAt80Gbps = 10;
    const uint64_t transfer_floor_ns =
        (static_cast<uint64_t>(request.value_len) + kBytesPerNsAt80Gbps - 1) /
        kBytesPerNsAt80Gbps;
    const auto l1_probe_start =
        wait_started + std::chrono::nanoseconds(std::max<uint64_t>(
                           kCompletionProbeFloorNs, transfer_floor_ns));
    std::array<uint8_t, protocol::kPromoteDoneSemSize> done{};
    uint32_t spins = 0;
    while (!external_host_io_cancelled_.load(std::memory_order_acquire)) {
      __builtin_ia32_lfence();
      if (*stage_completion_host_ptr_ == external_commit_issued_) {
        break;
      }
      __builtin_ia32_pause();
      if ((++spins & 0xFFu) == 0 &&
          std::chrono::steady_clock::now() >= l1_probe_start) {
        metrics.used_completion_l1_fallback = true;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device, kServerCore, promote_done_addr, std::span<uint8_t>(done),
            tt::CoreType::WORKER,
            /*barrier_scoped_to_core=*/true);
        if (read_u32(done.data(), 0) == external_commit_issued_) {
          break;
        }
      }
    }
  } else {
    std::array<uint8_t, protocol::kPromoteDoneSemSize> done{};
    while (!external_host_io_cancelled_.load(std::memory_order_acquire)) {
      tt::tt_metal::detail::ReadFromDeviceL1(
          device, kServerCore, promote_done_addr, std::span<uint8_t>(done),
          tt::CoreType::WORKER,
          /*barrier_scoped_to_core=*/true);
      if (read_u32(done.data(), 0) == external_commit_issued_) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(poll_us_));
    }
  }
  metrics.completion_wait_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - wait_started)
          .count();
  if (external_host_io_cancelled_.load(std::memory_order_acquire)) {
    TT_THROW("ServerRuntimeCore host value supply was cancelled");
  }
  external_host_request_.phase =
      ExternalHostRequestState::Phase::TransferComplete;
  return metrics;
}

void ServerRuntimeCore::complete_external_host_request(
    const ServerHostRequest &request) {
  validate_external_request(request);
  const bool remove_without_transfer =
      request.kind == ServerHostRequestKind::RemoveBacking &&
      external_host_request_.phase == ExternalHostRequestState::Phase::Pending;
  TT_FATAL(remove_without_transfer ||
               external_host_request_.phase ==
                   ExternalHostRequestState::Phase::TransferComplete,
           "ServerRuntimeCore host request cannot complete before its value "
           "transfer finishes");
  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr, "host request completion has no device at (0,0)");
  std::array<uint8_t, protocol::kMoveProcessedSemSize> processed{};
  std::memcpy(processed.data(), &request.sequence, sizeof(uint32_t));
  tt::tt_metal::detail::WriteToDeviceL1(
      device, kServerCore, server_l1_base_ + move_processed_offset_,
      std::span<const uint8_t>(processed), tt::CoreType::WORKER);
  external_seen_ = request.sequence;
  external_host_request_ = {};
}

void ServerRuntimeCore::cancel_external_host_io() noexcept {
  external_host_io_cancelled_.store(true, std::memory_order_release);
}

void ServerRuntimeCore::release_external_host_lease(
    ServerHostLease &lease) noexcept {
  std::lock_guard<std::mutex> lock(host_values_mutex_);
  if (lease.owner_ != this) {
    return;
  }

  const auto tracked = external_host_leases_.find(&lease);
  lease.owner_ = nullptr;
  if (tracked != external_host_leases_.end()) {
    external_host_leases_.erase(tracked);
    if (external_pool_units_in_use_ >= lease.pool_units_) {
      external_pool_units_in_use_ -= lease.pool_units_;
    } else {
      external_pool_units_in_use_ = 0;
    }
  }

  // Reclamation may allocate while merging the free-list. Untrack the lease
  // first so a failed reclamation can only leak this extent until runtime
  // teardown, never leave a dangling lease.
  try {
    invalidate_pool_meta_if_generation(lease.slot_, lease.generation_);
    pool_release(lease.pool_unit_, lease.pool_units_);
  } catch (...) {
    // Runtime teardown releases the underlying host allocation.
  }
}

void ServerRuntimeCore::detach_external_host_leases() noexcept {
  std::lock_guard<std::mutex> lock(host_values_mutex_);
  for (auto *lease : external_host_leases_) {
    lease->owner_ = nullptr;
  }
  external_host_leases_.clear();
  external_pool_units_in_use_ = 0;
}

} // namespace kvcache_manager::detail::server_runtime

namespace kvcache_manager::detail::server_runtime {

std::optional<ServerHostRequest> ServerDeviceRuntime::poll_host_request() {
  const auto current = state();
  if (current == ServerDeviceRuntimeState::Terminated) {
    return std::nullopt;
  }
  TT_FATAL(
      current == ServerDeviceRuntimeState::Running ||
          current == ServerDeviceRuntimeState::Closing,
      "ServerRuntimeCore host request poll requires Running or Closing state");
  return server_->poll_external_host_request();
}

ServerReceivedValue
ServerDeviceRuntime::receive_value(const ServerHostRequest &request,
                                   bool prefer_mapped_lease) {
  const auto current = state();
  TT_FATAL(current == ServerDeviceRuntimeState::Running ||
               current == ServerDeviceRuntimeState::Closing,
           "ServerRuntimeCore value receive requires Running or Closing state");
  return server_->receive_external_value(request, prefer_mapped_lease);
}

ServerSupplyMetrics
ServerDeviceRuntime::supply_value(const ServerHostRequest &request,
                                  const ServerHostLease *lease,
                                  std::span<const uint8_t> heap_bytes) {
  const auto current = state();
  TT_FATAL(current == ServerDeviceRuntimeState::Running ||
               current == ServerDeviceRuntimeState::Closing,
           "ServerRuntimeCore value supply requires Running or Closing state");
  return server_->supply_external_value(request, lease, heap_bytes);
}

void ServerDeviceRuntime::complete_host_request(
    const ServerHostRequest &request) {
  const auto current = state();
  TT_FATAL(current == ServerDeviceRuntimeState::Running ||
               current == ServerDeviceRuntimeState::Closing,
           "ServerRuntimeCore host request completion requires Running or "
           "Closing state");
  server_->complete_external_host_request(request);
}

std::optional<ServerHostRequest>
server_device_runtime_poll_host_request(ServerDeviceRuntime &runtime) {
  return runtime.poll_host_request();
}

ServerReceivedValue
server_device_runtime_receive_value(ServerDeviceRuntime &runtime,
                                    const ServerHostRequest &request,
                                    bool prefer_mapped_lease) {
  return runtime.receive_value(request, prefer_mapped_lease);
}

ServerSupplyMetrics server_device_runtime_supply_value(
    ServerDeviceRuntime &runtime, const ServerHostRequest &request,
    const ServerHostLease *lease, std::span<const uint8_t> heap_bytes) {
  return runtime.supply_value(request, lease, heap_bytes);
}

void server_device_runtime_complete_host_request(
    ServerDeviceRuntime &runtime, const ServerHostRequest &request) {
  runtime.complete_host_request(request);
}

} // namespace kvcache_manager::detail::server_runtime
