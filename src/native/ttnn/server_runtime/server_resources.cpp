// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#include "server_runtime/server_runtime_core.hpp"

#include <sys/mman.h> // mmap MAP_HUGETLB for the device-pull staging buffer

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <thread>
#include <tuple>
#include <unordered_map>
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

#include "shared_memory_mapping.hpp"
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
using ::kvcache_manager::detail::SharedMemoryMapping;

// Members are destroyed in reverse order, so the pin is released before its
// backing mapping.
struct SharedHostArena {
  std::shared_ptr<void> hugemap;
  std::shared_ptr<SharedMemoryMapping> shm;
  std::shared_ptr<experimental::PinnedMemory> pinned;
  size_t backing_bytes = 0;
  size_t mapping_bytes = 0;
  size_t slice_bytes = 0;
  size_t stage_and_control_bytes = 0;
  uint32_t worker_groups = 0;
};

namespace {
constexpr size_t kSystemPage = 4096;
constexpr size_t kHugePage = 1ull << 30;
constexpr double kDeviceDramBudgetRatio = 0.9;

uint64_t checked_add_dram_bytes(uint64_t lhs, uint64_t rhs) {
  TT_FATAL(rhs <= std::numeric_limits<uint64_t>::max() - lhs,
           "device DRAM reservation addition overflows uint64_t ({} + {})", lhs,
           rhs);
  return lhs + rhs;
}

uint64_t checked_multiply_dram_bytes(uint64_t lhs, uint64_t rhs) {
  TT_FATAL(lhs == 0 || rhs <= std::numeric_limits<uint64_t>::max() / lhs,
           "device DRAM reservation multiplication overflows uint64_t ({} * "
           "{})",
           lhs, rhs);
  return lhs * rhs;
}

// Cap on the host-value pool (see reserve_stage_buffer). Unset uses the
// remaining NoC-mapped region; zero forces host-source reads through the
// staging copy. A small non-zero cap makes the heap fallback testable because
// the default pool rarely fills in a normal run.
std::optional<size_t> parse_host_pool_bytes() {
  const auto parsed = config::read_u64_env("KVM_SERVER_HOST_POOL_BYTES");
  if (!parsed.has_value()) {
    return std::nullopt;
  }
  TT_FATAL(*parsed <= std::numeric_limits<size_t>::max(),
           "KVM_SERVER_HOST_POOL_BYTES={} exceeds size_t range", *parsed);
  return static_cast<size_t>(*parsed);
}

struct SharedServerL1Reservation {
  std::weak_ptr<MeshBuffer> buffer;
  uint32_t base = 0;
  uint32_t bytes = 0;
  uint32_t worker_groups = 0;
};

struct SharedAuxiliaryWorkerL1Reservation {
  std::weak_ptr<MeshBuffer> buffer;
  uint32_t base = 0;
  uint32_t bytes = 0;
  uint32_t worker_groups = 0;
};

std::mutex shared_server_l1_mutex;
std::unordered_map<MeshDevice *, SharedServerL1Reservation> shared_server_l1;

std::mutex shared_auxiliary_worker_l1_mutex;
std::unordered_map<
    MeshDevice *,
    std::unordered_map<uint32_t, SharedAuxiliaryWorkerL1Reservation>>
    shared_auxiliary_worker_l1;

std::mutex shared_host_arena_mutex;
std::unordered_map<MeshDevice *, std::weak_ptr<SharedHostArena>>
    shared_host_arenas;

uint32_t configured_server_worker_groups() {
  const auto parsed = config::read_u64_env("T3KNIC_SERVER_SHARDS").value_or(1);
  TT_FATAL(parsed >= 1 && parsed <= 8,
           "T3KNIC_SERVER_SHARDS must be in [1, 8], got {}", parsed);
  return static_cast<uint32_t>(parsed);
}

size_t page_align_down(size_t bytes) {
  return (bytes / kSystemPage) * kSystemPage;
}

size_t page_align_up(size_t bytes) {
  TT_FATAL(bytes <= std::numeric_limits<size_t>::max() - (kSystemPage - 1),
           "host arena size {} cannot be page-aligned without overflow", bytes);
  return ((bytes + kSystemPage - 1) / kSystemPage) * kSystemPage;
}

size_t checked_arena_bytes(size_t slice_bytes, uint32_t worker_groups) {
  TT_FATAL(slice_bytes <= std::numeric_limits<size_t>::max() / worker_groups,
           "host arena slice {} B * {} worker groups overflows size_t",
           slice_bytes, worker_groups);
  return slice_bytes * worker_groups;
}

std::shared_ptr<SharedHostArena>
create_host_arena(MeshDevice *server_submesh, size_t slice_bytes,
                  size_t stage_and_control_bytes, uint32_t worker_groups) {
  auto arena = std::make_shared<SharedHostArena>();
  arena->slice_bytes = slice_bytes;
  arena->stage_and_control_bytes = stage_and_control_bytes;
  arena->worker_groups = worker_groups;

  // MAP_HUGE_1GB requires a whole number of 1 GiB pages.
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << 26) // 30 = log2(1 GiB); 26 = MAP_HUGE_SHIFT
#endif
  void *hp = MAP_FAILED;
  if (slice_bytes >= stage_and_control_bytes) {
    hp = mmap(nullptr, kHugePage, PROT_READ | PROT_WRITE,
              MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0);
  }
  uint8_t *host_ptr = nullptr;
  if (hp != MAP_FAILED) {
    arena->hugemap =
        std::shared_ptr<void>(hp, [](void *p) { munmap(p, kHugePage); });
    arena->backing_bytes = kHugePage;
    arena->mapping_bytes = kHugePage;
    host_ptr = static_cast<uint8_t *>(hp);
    log_info(tt::LogOp,
             "kvcache_manager: created one endpoint host arena: {} B NoC "
             "mapping backed by one 1 GiB hugepage, {} worker-group slices "
             "of {} B.",
             arena->mapping_bytes, worker_groups, arena->slice_bytes);
  } else {
    // This is a correctness-only staging arena. It intentionally leaves no
    // per-shard host-value pool.
    arena->slice_bytes = page_align_up(stage_and_control_bytes);
    arena->mapping_bytes =
        checked_arena_bytes(arena->slice_bytes, worker_groups);
    arena->backing_bytes = arena->mapping_bytes;
    arena->shm = std::make_shared<SharedMemoryMapping>(
        SharedMemoryMapping::create(arena->backing_bytes));
    host_ptr = static_cast<uint8_t *>(arena->shm->data());
    if (slice_bytes < stage_and_control_bytes) {
      log_warning(
          tt::LogOp,
          "kvcache_manager: required staging/control region {} B exceeds the "
          "endpoint's default {} B host-arena slice; using one {} B 4 KiB "
          "shared-memory staging arena with {} worker-group slices of {} B "
          "and no host-value pool.",
          stage_and_control_bytes, slice_bytes, arena->mapping_bytes,
          worker_groups, arena->slice_bytes);
    } else {
      log_warning(tt::LogOp,
                  "kvcache_manager: endpoint 1 GiB hugepage arena "
                  "unavailable; using one {} B 4 KiB shared-memory staging "
                  "arena with {} worker-group slices of {} B and no "
                  "host-value pool.",
                  arena->mapping_bytes, worker_groups, arena->slice_bytes);
    }
  }

  auto pin_alias = std::shared_ptr<uint8_t[]>(host_ptr, [](uint8_t *) {});
  HostBuffer view(
      tt::stl::Span<uint32_t>(reinterpret_cast<uint32_t *>(host_ptr),
                              arena->mapping_bytes / sizeof(uint32_t)),
      MemoryPin(pin_alias));
  MeshCoordinateRangeSet range(MeshCoordinateRange(MeshCoordinate(0, 0)));
  try {
    arena->pinned = experimental::PinnedMemory::Create(
        *server_submesh, range, view, /*map_to_noc=*/true);
  } catch (const std::exception &e) {
    log_warning(tt::LogOp,
                "kvcache_manager: NoC mapping of the endpoint host arena "
                "failed ({}); falling back to one DMA-only arena "
                "(host-source transfers use targeted PCIe copies).",
                e.what());
    arena->pinned = experimental::PinnedMemory::Create(
        *server_submesh, range, view, /*map_to_noc=*/false);
  }
  return arena;
}

} // namespace

// Reserve the server L1 carve-out, sized at runtime from the resolved ring
// count (never a fixed hard-cap). Every configured worker group shares one
// multi-core bottom-up allocation, so every bootstrap core has the same base.
// Guarded against the reserve ceiling and the physical worker-L1 budget. See
// ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §2/§4.
void ServerRuntimeCore::reserve_l1() {
  // server_pipeline_end mirrors launch_server_kernel's offset derivation;
  // page-round the result.
  const uint64_t pipeline_end = protocol::server_pipeline_end(ring_slots_);
  constexpr uint64_t kReservePage = 4096;
  const uint64_t reserve_aligned =
      ((pipeline_end + kReservePage - 1) / kReservePage) * kReservePage;

  TT_FATAL(reserve_aligned <= protocol::kServerL1ReserveMaxBytes,
           "kvcache_manager: server L1 reserve {} B exceeds the reserve "
           "ceiling {} B "
           "(ring_slots={}). Lower KVM_SERVER_L1_RING_SLOTS.",
           reserve_aligned, protocol::kServerL1ReserveMaxBytes, ring_slots_);

  // The carve-out must fit the server core's per-bank L1. Back-solve the max
  // ring_slots that would still fit so the TT_FATAL names exactly which knob to
  // lower (per slot = 4368 B).
  const uint64_t l1_budget =
      server_submesh_->allocator()->get_bank_size(BufferType::L1);
  const uint64_t residency_lru_bytes =
      protocol::residency_lru_l1_bytes(max_keys_);
  const uint64_t get_value_cb_bytes =
      static_cast<uint64_t>(protocol::kGetValueCbDepth) *
      protocol::kMaxChunkValueBytes;
  const uint64_t total_server_l1_bytes =
      reserve_aligned + residency_lru_bytes + get_value_cb_bytes;
  const uint64_t per_slot = protocol::kRingSlotStride + protocol::kRingSemSize;
  const uint64_t non_ring_part =
      reserve_aligned - static_cast<uint64_t>(ring_slots_) * per_slot;
  const uint64_t max_ring_slots = (l1_budget > non_ring_part)
                                      ? ((l1_budget - non_ring_part) / per_slot)
                                      : 0;
  TT_FATAL(reserve_aligned <= l1_budget,
           "kvcache_manager: server L1 reserve {} B exceeds available worker "
           "L1 {} B "
           "(ring_slots={}). Lower KVM_SERVER_L1_RING_SLOTS to <= {}.",
           reserve_aligned, l1_budget, ring_slots_, max_ring_slots);
  TT_FATAL(total_server_l1_bytes <= l1_budget,
           "kvcache_manager: server L1 working set {} B exceeds available "
           "worker L1 {} B "
           "(carve-out={}, residency_lru={} for max_keys={}, get_cb={}). Lower "
           "KVM_SERVER_L1_RING_SLOTS or KVM_SERVER_DRAM_MAX_KEYS.",
           total_server_l1_bytes, l1_budget, reserve_aligned,
           residency_lru_bytes, max_keys_, get_value_cb_bytes);
  server_l1_reserve_bytes_ = static_cast<uint32_t>(reserve_aligned);
  const uint32_t reserve_bytes = static_cast<uint32_t>(reserve_aligned);

  const uint32_t worker_groups =
      static_intermesh_t3k_server_ ? configured_server_worker_groups() : 1;
  auto shard = ShardSpecBuffer(
      CoreRangeSet(CoreRange(CoreCoord{0, kServerCore.y},
                             CoreCoord{worker_groups - 1, kServerCore.y})),
      /*shard_shape*/ {1, 1}, ShardOrientation::ROW_MAJOR,
      /*page_shape*/ {1, 1},
      /*tensor2d_shape_in_pages*/ {1, 1});

  DeviceLocalBufferConfig local{
      .page_size = reserve_bytes,
      .buffer_type = BufferType::L1,
      .sharding_args =
          BufferShardingArgs(shard, TensorMemoryLayout::HEIGHT_SHARDED),
      .bottom_up = true,
      .sub_device_id = std::nullopt,
  };

  ReplicatedBufferConfig global{
      .size = reserve_bytes,
  };

  {
    std::lock_guard<std::mutex> lock(shared_server_l1_mutex);
    auto &reservation = shared_server_l1[server_submesh_];
    server_l1_buf_ = reservation.buffer.lock();
    if (server_l1_buf_ != nullptr) {
      TT_FATAL(reservation.bytes == reserve_bytes &&
                   reservation.worker_groups == worker_groups,
               "all KVM Server shards on one submesh must use the same L1 "
               "layout");
      server_l1_base_ = reservation.base;
    } else {
      server_l1_buf_ = MeshBuffer::create(global, local, server_submesh_);
      server_l1_base_ = static_cast<uint32_t>(server_l1_buf_->address());
      reservation = SharedServerL1Reservation{
          .buffer = server_l1_buf_,
          .base = server_l1_base_,
          .bytes = reserve_bytes,
          .worker_groups = worker_groups,
      };
    }
  }

  const uint32_t allocator_base =
      server_submesh_->allocator()->get_base_allocator_addr(HalMemType::L1);
  TT_FATAL(server_l1_base_ == allocator_base,
           "kvcache_manager: shared Server L1 reserve base {} differs from "
           "the endpoint-free bootstrap base {}",
           server_l1_base_, allocator_base);

  const uint32_t lru_bytes = static_cast<uint32_t>(residency_lru_bytes);
  auto lru_shard =
      ShardSpecBuffer(CoreRangeSet(CoreRange(kServerCore)),
                      /*shard_shape*/ {1, 1}, ShardOrientation::ROW_MAJOR,
                      /*page_shape*/ {1, 1},
                      /*tensor2d_shape_in_pages*/ {1, 1});
  DeviceLocalBufferConfig lru_local{
      .page_size = lru_bytes,
      .buffer_type = BufferType::L1,
      .sharding_args =
          BufferShardingArgs(lru_shard, TensorMemoryLayout::HEIGHT_SHARDED),
      .bottom_up = false,
      .sub_device_id = std::nullopt,
  };
  ReplicatedBufferConfig lru_global{.size = lru_bytes};
  residency_lru_buf_ =
      MeshBuffer::create(lru_global, lru_local, server_submesh_);
  residency_lru_base_ = static_cast<uint32_t>(residency_lru_buf_->address());
  TT_FATAL(residency_lru_base_ % 32 == 0,
           "kvcache_manager: residency LRU L1 addr {} must be 32-aligned",
           residency_lru_base_);
}

void ServerRuntimeCore::reserve_auxiliary_worker_l1(FabricFreeWorker &worker,
                                                    CoreCoord core,
                                                    uint32_t bytes,
                                                    const char *role_name) {
  TT_ASSERT(worker.enabled);
  TT_ASSERT(worker.l1 == nullptr);
  const uint64_t l1_budget =
      server_submesh_->allocator()->get_bank_size(BufferType::L1);
  TT_FATAL(bytes <= l1_budget,
           "kvcache_manager: {} worker needs {} B but worker L1 has {} B",
           role_name, bytes, l1_budget);

  const uint32_t worker_groups =
      static_intermesh_t3k_server_ ? configured_server_worker_groups() : 1;
  if (static_intermesh_t3k_server_) {
    TT_FATAL(core.x == worker_group_,
             "kvcache_manager: {} worker core x={} does not match worker "
             "group {}",
             role_name, core.x, worker_group_);
  }
  const CoreRange worker_range =
      static_intermesh_t3k_server_
          ? CoreRange(CoreCoord{0, core.y},
                      CoreCoord{worker_groups - 1, core.y})
          : CoreRange(core);
  auto shard =
      ShardSpecBuffer(CoreRangeSet(worker_range),
                      /*shard_shape*/ {1, 1}, ShardOrientation::ROW_MAJOR,
                      /*page_shape*/ {1, 1},
                      /*tensor2d_shape_in_pages*/ {1, 1});
  DeviceLocalBufferConfig local{
      .page_size = bytes,
      .buffer_type = BufferType::L1,
      .sharding_args =
          BufferShardingArgs(shard, TensorMemoryLayout::HEIGHT_SHARDED),
      .bottom_up = false,
      .sub_device_id = std::nullopt,
  };
  ReplicatedBufferConfig global{.size = bytes};

  bool created_reservation = true;
  if (static_intermesh_t3k_server_) {
    std::lock_guard<std::mutex> lock(shared_auxiliary_worker_l1_mutex);
    auto &reservation = shared_auxiliary_worker_l1[server_submesh_][core.y];
    worker.l1 = reservation.buffer.lock();
    if (worker.l1 != nullptr) {
      TT_FATAL(reservation.bytes == bytes &&
                   reservation.worker_groups == worker_groups,
               "all KVM Server shards on one submesh must use the same {} "
               "L1 layout",
               role_name);
      worker.l1_base = reservation.base;
      created_reservation = false;
    } else {
      worker.l1 = MeshBuffer::create(global, local, server_submesh_);
      worker.l1_base = static_cast<uint32_t>(worker.l1->address());
      reservation = SharedAuxiliaryWorkerL1Reservation{
          .buffer = worker.l1,
          .base = worker.l1_base,
          .bytes = bytes,
          .worker_groups = worker_groups,
      };
    }
  } else {
    worker.l1 = MeshBuffer::create(global, local, server_submesh_);
    worker.l1_base = static_cast<uint32_t>(worker.l1->address());
  }

  TT_FATAL(worker.l1_base % 32 == 0,
           "kvcache_manager: {} L1 base {} must be 32-byte aligned", role_name,
           worker.l1_base);
  if (static_intermesh_t3k_server_) {
    log_info(tt::LogOp,
             "kvcache_manager: worker group {} uses {} endpoint-shared {} L1 "
             "reservation at {} ({} B per core across {} cores).",
             worker_group_, created_reservation ? "new" : "shared", role_name,
             worker.l1_base, bytes, worker_groups);
  }
}

void ServerRuntimeCore::reserve_get_workers() {
  if (scalar_get_pipeline_.ingress.enabled) {
    reserve_auxiliary_worker_l1(scalar_get_pipeline_.ingress, kGetIngressCore,
                                protocol::kGetIngressL1Bytes,
                                "scalar GET ingress");
    scalar_get_pipeline_.ingress.termination_addr =
        scalar_get_pipeline_.ingress.l1_base +
        protocol::kGetIngressTerminationOffset;
  }
  if (scalar_get_pipeline_.admission.enabled) {
    reserve_auxiliary_worker_l1(
        scalar_get_pipeline_.admission, kGetAdmissionCore,
        protocol::kGetAdmissionL1Bytes, "GET admission");
    scalar_get_pipeline_.admission.termination_addr =
        scalar_get_pipeline_.admission.l1_base +
        protocol::kGetAdmissionTerminationOffset;
  }
}

// Reserve the server DRAM buffers (sub-page packing): the value heap
// (num_units_ pages of U = kMaxChunkValueBytes; a value occupies ceil(len/U)
// contiguous U-aligned units), the pending-host FIFO (one slot/token descriptor
// per directory entry), and the key directory. See
// ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §4/§5.
void ServerRuntimeCore::reserve_dram() {
  const uint64_t value_total_bytes =
      static_cast<uint64_t>(num_units_) * protocol::kMaxChunkValueBytes;
  const uint64_t put_host_queue_total_bytes =
      static_cast<uint64_t>(max_keys_) * protocol::kPutHostQueueEntryBytes;

  IDevice *dev = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(dev != nullptr, "server_submesh has no chip at (0,0)");

  if (!dev->is_mmio_capable()) {
    log_warning(tt::LogOp,
                "kvcache_manager: server chip (device id {}) is not "
                "MMIO-capable; host/device transfers "
                "will use targeted PCIe copies. Map the server submesh to an "
                "MMIO-capable chip for the "
                "NoC-mapped host path.",
                dev->id());
  }

  const uint64_t directory_total_bytes =
      static_cast<uint64_t>(max_keys_) * dir_entry_bytes_;
  const uint64_t hashmap_total_bytes =
      static_cast<uint64_t>(num_buckets_) * protocol::kHashBucketBytes;
  const uint64_t heap_metadata_total_bytes =
      static_cast<uint64_t>(num_units_) * protocol::kHeapMetadataEntryBytes;

  const auto &allocator = dev->allocator();
  const uint64_t dram_bank_size =
      static_cast<uint64_t>(allocator->get_bank_size(BufferType::DRAM));
  const uint32_t dram_num_banks = allocator->get_num_banks(BufferType::DRAM);
  const uint32_t dram_alignment = allocator->get_alignment(BufferType::DRAM);
  TT_FATAL(dram_num_banks > 0 && dram_alignment > 0,
           "reserve_dram: invalid DRAM allocator geometry (banks={}, "
           "alignment={})",
           dram_num_banks, dram_alignment);

  const std::array<std::pair<uint64_t, uint32_t>, 5> buffers{{
      {value_total_bytes, protocol::kMaxChunkValueBytes},
      {put_host_queue_total_bytes, protocol::kPutHostQueueEntryBytes},
      {directory_total_bytes, dir_entry_bytes_},
      {hashmap_total_bytes, protocol::kHashBucketBytes},
      {heap_metadata_total_bytes, protocol::kHeapMetadataEntryBytes},
  }};
  uint64_t logical_reserve_per_server = 0;
  uint64_t physical_reserve_per_bank = 0;
  for (const auto &[logical_bytes, page_bytes] : buffers) {
    logical_reserve_per_server =
        checked_add_dram_bytes(logical_reserve_per_server, logical_bytes);
    physical_reserve_per_bank = checked_add_dram_bytes(
        physical_reserve_per_bank,
        tt::tt_metal::detail::calculate_bank_size_spread(
            logical_bytes, page_bytes, dram_num_banks, dram_alignment));
  }
  const uint64_t physical_reserve_per_server =
      checked_multiply_dram_bytes(physical_reserve_per_bank, dram_num_banks);
  const uint64_t dram_total =
      checked_multiply_dram_bytes(dram_bank_size, dram_num_banks);

  const uint32_t reservation_count =
      static_intermesh_t3k_server_ ? configured_server_worker_groups() : 1;
  const uint64_t aggregate_reserve = checked_multiply_dram_bytes(
      physical_reserve_per_server, reservation_count);
  // The Server owns this endpoint's storage role; keep 10% for dispatch,
  // Fabric, and future allocations.
  const uint64_t aggregate_budget = static_cast<uint64_t>(
      static_cast<double>(dram_total) * kDeviceDramBudgetRatio);
  TT_FATAL(aggregate_reserve <= aggregate_budget,
           "reserve_dram: endpoint physical reservation {} bytes ({} per "
           "Server * {} reservations) exceeds the 90% device DRAM budget {} "
           "of {} bytes (logical={} per Server, value_heap={} per Server, "
           "max_keys={} per Server). Lower "
           "KVM_SERVER_DRAM_VALUE_HEAP_BYTES / KVM_SERVER_DRAM_MAX_KEYS or use "
           "fewer Server shards.",
           aggregate_reserve, physical_reserve_per_server, reservation_count,
           aggregate_budget, dram_total, logical_reserve_per_server,
           value_heap_bytes_, max_keys_);
  if (!static_intermesh_t3k_server_ || worker_group_ == 0) {
    log_info(tt::LogOp,
             "kvcache_manager: endpoint physical DRAM reservation is {} B "
             "across {} Server instance(s) ({} physical / {} logical B per "
             "Server, value_heap={} B per Server), within the 90% budget {} B "
             "of {} B device DRAM.",
             aggregate_reserve, reservation_count, physical_reserve_per_server,
             logical_reserve_per_server, value_heap_bytes_, aggregate_budget,
             dram_total);
  }

  {
    DeviceLocalBufferConfig local{
        .page_size = protocol::kMaxChunkValueBytes,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(std::optional<ShardSpecBuffer>{},
                                            TensorMemoryLayout::INTERLEAVED),
        .bottom_up = false,
        .sub_device_id = std::nullopt,
    };
    ReplicatedBufferConfig global{.size = value_total_bytes};
    dram_value_buf_ = MeshBuffer::create(global, local, server_submesh_);
    dram_value_base_ = static_cast<uint32_t>(dram_value_buf_->address());
  }
  TT_FATAL(dram_value_base_ % 32 == 0, "dram_value base {} must be 32-aligned",
           dram_value_base_);

  {
    DeviceLocalBufferConfig local{
        .page_size = protocol::kPutHostQueueEntryBytes,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(std::optional<ShardSpecBuffer>{},
                                            TensorMemoryLayout::INTERLEAVED),
        .bottom_up = false,
        .sub_device_id = std::nullopt,
    };
    ReplicatedBufferConfig global{.size = put_host_queue_total_bytes};
    dram_put_host_queue_buf_ =
        MeshBuffer::create(global, local, server_submesh_);
    dram_put_host_queue_base_ =
        static_cast<uint32_t>(dram_put_host_queue_buf_->address());
  }
  TT_FATAL(dram_put_host_queue_base_ % 32 == 0,
           "dram_put_host_queue base {} must be 32-aligned",
           dram_put_host_queue_base_);

  // Key directory: INTERLEAVED, one dir_entry_bytes_ entry per page. The BRISC
  // addresses pages with InterleavedAddrGen (GET/EXISTS/commit); the
  // host-direct path reads the buffer whole.
  {
    DeviceLocalBufferConfig local{
        .page_size = dir_entry_bytes_,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(std::optional<ShardSpecBuffer>{},
                                            TensorMemoryLayout::INTERLEAVED),
        .bottom_up = false,
        .sub_device_id = std::nullopt,
    };
    ReplicatedBufferConfig global{.size = directory_total_bytes};
    dram_directory_buf_ = MeshBuffer::create(global, local, server_submesh_);
    dram_directory_base_ =
        static_cast<uint32_t>(dram_directory_buf_->address());
  }
  TT_FATAL(dram_directory_base_ % 32 == 0,
           "dram_directory base {} must be 32-aligned", dram_directory_base_);

  // The entry stride must already be DRAM-page-aligned so the BRISC's
  // InterleavedAddrGen and the host's (bank, slot * aligned_page) math agree;
  // guards a platform whose DRAM alignment exceeds 32.
  {
    auto *dir_dev =
        dram_directory_buf_->get_device_buffer(MeshCoordinate(0, 0));
    TT_FATAL(
        dir_dev != nullptr,
        "reserve_dram: directory MeshBuffer has no device buffer at (0,0)");
    TT_FATAL(static_cast<uint32_t>(dir_dev->aligned_page_size()) ==
                 dir_entry_bytes_,
             "directory aligned page size {} != dir_entry_bytes {} (DRAM "
             "alignment exceeds the 32-byte entry "
             "rounding; raise the entry rounding to the DRAM alignment)",
             dir_dev->aligned_page_size(), dir_entry_bytes_);
  }

  // BRISC-only auxiliary hash index; the host-direct path scans the
  // authoritative directory.
  {
    DeviceLocalBufferConfig local{
        .page_size = protocol::kHashBucketBytes,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(std::optional<ShardSpecBuffer>{},
                                            TensorMemoryLayout::INTERLEAVED),
        .bottom_up = false,
        .sub_device_id = std::nullopt,
    };
    ReplicatedBufferConfig global{.size = hashmap_total_bytes};
    dram_hashmap_buf_ = MeshBuffer::create(global, local, server_submesh_);
    dram_hashmap_base_ = static_cast<uint32_t>(dram_hashmap_buf_->address());
  }
  TT_FATAL(dram_hashmap_base_ % 32 == 0,
           "dram_hashmap base {} must be 32-aligned", dram_hashmap_base_);

  // Buddy allocator metadata: INTERLEAVED, one kHeapMetadataEntryBytes entry
  // per heap unit.
  {
    DeviceLocalBufferConfig local{
        .page_size = protocol::kHeapMetadataEntryBytes,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(std::optional<ShardSpecBuffer>{},
                                            TensorMemoryLayout::INTERLEAVED),
        .bottom_up = false,
        .sub_device_id = std::nullopt,
    };
    ReplicatedBufferConfig global{.size = heap_metadata_total_bytes};
    dram_heap_metadata_buf_ =
        MeshBuffer::create(global, local, server_submesh_);
    dram_heap_metadata_base_ =
        static_cast<uint32_t>(dram_heap_metadata_buf_->address());
  }
  TT_FATAL(dram_heap_metadata_base_ % 32 == 0,
           "dram_heap_metadata base {} must be 32-aligned",
           dram_heap_metadata_base_);
}

// Reserve this worker group's slice of the endpoint-shared NoC-mapped staging
// and host-pool arena. Hugepage allocation enables the pool; shared-memory and
// targeted-copy fallbacks preserve correctness through copies. See
// ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §5 and §6.
void ServerRuntimeCore::reserve_stage_buffer() {
  const uint32_t U = protocol::kMaxChunkValueBytes;
  const auto host_pool_cap = parse_host_pool_bytes();
  stage_bytes_ = protocol::chunk_aligned_value_bytes(max_value_bytes_);
  stage_completion_offset_ = stage_bytes_;
  get_completion_offset_ =
      stage_completion_offset_ + protocol::kPromoteHostCompletionBytes;
  get_ingress_offset_ =
      get_completion_offset_ + protocol::kGetHostCompletionBytes;
  const size_t pool_meta_offset =
      static_cast<size_t>(get_ingress_offset_) + protocol::kGetIngressHostBytes;
  const size_t pool_meta_bytes =
      static_cast<size_t>(max_keys_) * protocol::kHostPoolMetaBytes;
  const size_t stage_and_control_bytes = pool_meta_offset + pool_meta_bytes;
  TT_FATAL(stage_and_control_bytes <= std::numeric_limits<uint32_t>::max(),
           "staging/control region {} B exceeds the 32-bit device offset range",
           stage_and_control_bytes);
  pool_meta_read_off_ = static_cast<uint32_t>(pool_meta_offset);
  pool_meta_generations_.assign(max_keys_, 0);

  IDevice *dev = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(dev != nullptr,
           "reserve_stage_buffer: server_submesh has no chip at (0,0)");
  const auto did = dev->id();
  const uint32_t worker_groups =
      static_intermesh_t3k_server_ ? configured_server_worker_groups() : 1;
  const size_t configured_slice_bytes =
      page_align_down(kHugePage / worker_groups);

  bool created_arena = false;
  {
    std::lock_guard<std::mutex> lock(shared_host_arena_mutex);
    auto &weak_arena = shared_host_arenas[server_submesh_];
    stage_arena_ = weak_arena.lock();
    if (stage_arena_ != nullptr) {
      TT_FATAL(stage_arena_->worker_groups == worker_groups &&
                   stage_arena_->stage_and_control_bytes ==
                       stage_and_control_bytes,
               "all KVM Server shards on one endpoint must use the same host "
               "arena layout");
    } else {
      stage_arena_ = create_host_arena(server_submesh_, configured_slice_bytes,
                                       stage_and_control_bytes, worker_groups);
      weak_arena = stage_arena_;
      created_arena = true;
    }
  }

  const size_t slice_offset = stage_arena_->slice_bytes * worker_group_;
  TT_FATAL(
      slice_offset + stage_arena_->slice_bytes <= stage_arena_->mapping_bytes,
      "worker group {} host-arena slice [{}..{}) exceeds the {} B arena",
      worker_group_, slice_offset, slice_offset + stage_arena_->slice_bytes,
      stage_arena_->mapping_bytes);
  log_info(tt::LogOp,
           "kvcache_manager: worker group {} uses {} endpoint host-arena "
           "slice at offset {} ({} B).",
           worker_group_, created_arena ? "new" : "shared", slice_offset,
           stage_arena_->slice_bytes);

  // The worker copies into the same slice that device NoC operations address.
  stage_host_ptr_ =
      static_cast<uint8_t *>(stage_arena_->pinned->get_host_ptr()) +
      slice_offset;
  stage_completion_host_ptr_ = reinterpret_cast<volatile uint32_t *>(
      stage_host_ptr_ + stage_completion_offset_);
  std::memset(stage_host_ptr_ + stage_completion_offset_, 0,
              protocol::kPromoteHostCompletionBytes);
  *stage_completion_host_ptr_ = test_initial_counter_;
  get_completion_host_ptr_ = reinterpret_cast<volatile uint32_t *>(
      stage_host_ptr_ + get_completion_offset_);
  std::memset(stage_host_ptr_ + get_completion_offset_, 0,
              protocol::kGetHostCompletionBytes);
  get_ingress_host_ptr_ = stage_host_ptr_ + get_ingress_offset_;
  std::memset(get_ingress_host_ptr_, 0, protocol::kGetIngressHostBytes);
  get_ingress_generation_ = test_initial_counter_;
  pool_meta_host_ptr_ = stage_host_ptr_ + pool_meta_read_off_;
  std::memset(pool_meta_host_ptr_, 0, pool_meta_bytes);

  const auto noc = stage_arena_->pinned->get_noc_addr(did);
  stage_noc_ok_ = noc.has_value() && stage_arena_->pinned->usable_from_noc(did);
  scalar_get_pipeline_.mapped_completion_enabled =
      scalar_get_pipeline_.mapped_completion_enabled && stage_noc_ok_;
  const CoreCoord grid = server_submesh_->compute_with_storage_grid_size();
  const bool ingress_core_available =
      grid.x > kGetIngressCore.x && grid.y > kGetIngressCore.y;
  const bool admission_core_available =
      grid.x > kGetAdmissionCore.x && grid.y > kGetAdmissionCore.y;
  scalar_get_pipeline_.ingress.enabled =
      scalar_get_pipeline_.ingress.enabled &&
      scalar_get_pipeline_.mapped_completion_enabled && ingress_core_available;
  scalar_get_pipeline_.admission.enabled =
      scalar_get_pipeline_.admission.enabled && stage_noc_ok_ &&
      admission_core_available;
  if (!ingress_core_available) {
    log_warning(tt::LogOp,
                "kvcache_manager: scalar GET mapped ingress disabled because "
                "logical worker core ({},{}) is outside grid "
                "({},{})",
                kGetIngressCore.x, kGetIngressCore.y, grid.x, grid.y);
  }
  if (!admission_core_available) {
    log_warning(tt::LogOp,
                "kvcache_manager: GET admission worker disabled because "
                "logical worker core ({},{}) is outside grid "
                "({},{})",
                kGetAdmissionCore.x, kGetAdmissionCore.y, grid.x, grid.y);
  }
  if (stage_noc_ok_) {
    TT_FATAL(noc->addr <= std::numeric_limits<uint64_t>::max() - slice_offset,
             "NoC arena base 0x{:x} + slice offset {} overflows uint64_t",
             noc->addr, slice_offset);
    const uint64_t stage_noc_addr = noc->addr + slice_offset;
    stage_pcie_xy_enc_ = noc->pcie_xy_enc;
    stage_addr_lo_ = static_cast<uint32_t>(stage_noc_addr & 0xFFFFFFFFull);
    stage_addr_hi_ = static_cast<uint32_t>(stage_noc_addr >> 32);
    // Kernels add offsets to the low 32 bits only; reject mappings that would
    // wrap that add. See
    // ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §5.
    TT_FATAL((stage_noc_addr & 0xFFFFFFFFull) + stage_arena_->slice_bytes <=
                 0x100000000ull,
             "NoC-mapped region 0x{:x} + {} B crosses a 4 GiB boundary; the "
             "kernel's 32-bit offset add would wrap",
             stage_noc_addr, stage_arena_->slice_bytes);

    // Everything past the completion cacheline and per-slot metadata becomes
    // the host-value pool, so write-through can land where later host-source
    // operations read without another copy.
    const size_t pool_offset = stage_and_control_bytes;
    size_t pool_bytes = stage_arena_->slice_bytes - pool_offset;
    if (host_pool_cap.has_value()) {
      pool_bytes = std::min(pool_bytes, *host_pool_cap);
    }
    const uint32_t pool_units = static_cast<uint32_t>(pool_bytes / U);
    if (pool_units > 0) {
      pool_host_ptr_ = stage_host_ptr_ + pool_offset;
      pool_read_off_ = static_cast<uint32_t>(pool_offset);
      pool_units_total_ = pool_units;
      pool_free_.clear();
      pool_free_.emplace(0u, pool_units);
    }
    log_info(tt::LogOp,
             "kvcache_manager: NoC host-source reads enabled (server chip {}, "
             "staging {} B + host-value "
             "pool {} B, NoC 0x{:x} in a {} B mapping backed by {} B).",
             did, stage_bytes_, static_cast<size_t>(pool_units) * U,
             stage_noc_addr, stage_arena_->mapping_bytes,
             stage_arena_->backing_bytes);
  } else {
    log_warning(tt::LogOp,
                "kvcache_manager: server chip {} host staging is not "
                "NoC-usable; host/device transfers use "
                "targeted PCIe copies. Use an MMIO-capable chip and a "
                "NoC-usable host mapping for the mapped path.",
                did);
  }
}

} // namespace kvcache_manager::detail::server_runtime
