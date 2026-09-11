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
using ttnn_kvm::is_kvcache_2d_fabric;
using ttnn_kvm::is_kvcache_wire_reachable;
using ttnn_kvm::kvcache_first_hop_direction;
using ttnn_kvm::kvcache_kernel_defines;
using ttnn_kvm::kvcache_unicast_route_word;
using ttnn_kvm::kvcache_wire_link_indices;
using ttnn_kvm::KvcacheKernelTarget;
using ttnn_kvm::KvcacheRouteMode;

namespace {
constexpr uint32_t kImmediateTerminationSignal = 2;

constexpr uint32_t kTerminationOffset = 0;

const std::string kServerKernelPath =
    "ttnn/cpp/ttnn/operations/kvcache_manager/server/device/kernels/dataflow/"
    "server_kernel.cpp";

// BRISC (RISCV_0) directory, allocator, and DRAM authority. Shares the server
// core with the NCRISC Fabric kernel and opens no Fabric connection.
const std::string kDramWorkerKernelPath =
    "ttnn/cpp/ttnn/operations/kvcache_manager/server/device/kernels/dataflow/"
    "dram_worker_kernel.cpp";

const std::string kGetIngressKernelPath =
    "ttnn/cpp/ttnn/operations/kvcache_manager/server/device/kernels/dataflow/"
    "get_ingress_kernel.cpp";

const std::string kGetAdmissionKernelPath =
    "ttnn/cpp/ttnn/operations/kvcache_manager/server/device/kernels/dataflow/"
    "get_admission_kernel.cpp";

uint32_t configured_static_server_shards() {
  const char *value = std::getenv("T3KNIC_SERVER_SHARDS");
  if (value == nullptr || *value == '\0') {
    return 1;
  }

  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  TT_FATAL(end != value && *end == '\0' && parsed >= 1 && parsed <= 8,
           "T3KNIC_SERVER_SHARDS must be in [1,8]");
  return static_cast<uint32_t>(parsed);
}

class FailureFlagGuard {
public:
  explicit FailureFlagGuard(bool *failure_flag) : failure_flag_(failure_flag) {}

  FailureFlagGuard(const FailureFlagGuard &) = delete;
  FailureFlagGuard &operator=(const FailureFlagGuard &) = delete;

  ~FailureFlagGuard() {
    if (armed_ && failure_flag_ != nullptr) {
      *failure_flag_ = true;
    }
  }

  void release() noexcept { armed_ = false; }

private:
  bool *failure_flag_ = nullptr;
  bool armed_ = true;
};

} // namespace

struct ServerRuntimeCore::StaticIntermeshMuxService {
  static constexpr uint32_t kLinks = 2;
  // Static inter-mesh firmware leaves room for one max-packet slot per
  // channel on the lane-local MUX cores. This remains credit-safe while all
  // shard producers share one persistent workload.
  static constexpr uint32_t kBuffersPerChannel = 1;

  MeshDevice *submesh = nullptr;
  uint32_t shard_count = 0;
  std::array<CoreCoord, kLinks> logical_cores{};
  std::array<CoreCoord, kLinks> virtual_cores{};
  std::array<std::unique_ptr<tt::tt_fabric::FabricMuxV2Config>, kLinks>
      configs{};
  Program program = CreateProgram();
  std::unique_ptr<MeshWorkload> workload;
  std::vector<ServerRuntimeCore *> servers;
  std::vector<std::shared_ptr<void>> retired_kernel_resources;
  uint32_t terminated_servers = 0;
  bool prepared = false;
  bool preparation_failed = false;
  bool launched = false;
  std::mutex mutex;

  StaticIntermeshMuxService(MeshDevice *server_submesh, uint32_t num_shards)
      : submesh(server_submesh), shard_count(num_shards) {
    TT_FATAL(submesh != nullptr,
             "static intermesh MUX requires a server submesh");
    TT_FATAL(shard_count >= kLinks,
             "dual-link KVM MUX requires at least two Server shards");

    auto *device = submesh->get_device(MeshCoordinate(0, 0));
    TT_FATAL(device != nullptr, "server_submesh has no chip at (0,0)");
    const auto grid = submesh->compute_with_storage_grid_size();
    TT_FATAL(grid.x >= kLinks && grid.y >= 4,
             "dual-link KVM MUX requires free logical cores (0..1,3)");

    const auto mux_l1_base = static_cast<uint32_t>(
        device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const auto channel_bytes =
        tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    for (uint32_t lane = 0; lane < kLinks; ++lane) {
      const uint32_t lane_channels =
          (shard_count + (kLinks - 1u - lane)) / kLinks;
      TT_FATAL(lane_channels > 0 && lane_channels <= 4,
               "KVM MUX lane {} has invalid channel count {}", lane,
               lane_channels);

      logical_cores[lane] = CoreCoord{lane, 3};
      virtual_cores[lane] =
          device->worker_core_from_logical_core(logical_cores[lane]);
      configs[lane] = std::make_unique<tt::tt_fabric::FabricMuxV2Config>(
          static_cast<uint8_t>(lane_channels),
          static_cast<uint8_t>(kBuffersPerChannel), channel_bytes, mux_l1_base);

      std::vector<uint32_t> downstream_args;
      tt::tt_fabric::append_static_intermesh_lane_connection_rt_args(
          lane, program, logical_cores[lane], downstream_args);
      tt::tt_fabric::add_fabric_mux_v2_to_program(
          program, *configs[lane], logical_cores[lane], downstream_args,
          NOC::RISCV_0_default);
    }
  }
};

// Derive the runtime L1 offsets (ring_slots_-dependent), guard the DRAM-NoC
// 32-byte alignment invariants, create the NCRISC server + BRISC DRAM-worker
// kernels, and start the host tiering worker. L1 / DRAM layout and the 32-byte
// alignment rule are documented in
// ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §4.
// Host initialization protects state observable before asynchronous device
// cold-start and clears state that may survive a prior session. See
// ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §8 and §9.
void ServerRuntimeCore::prepare_server_kernel(
    const config::ServerDirectoryPolicy &directory_policy) {
  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr, "server_submesh has no chip at (0,0)");
  CoreCoord admission_virtual{0, 0};
  if (scalar_get_pipeline_.admission.enabled) {
    admission_virtual =
        device->worker_core_from_logical_core(kGetAdmissionCore);
  }

  const uint32_t l1_base = server_l1_base_;
  termination_addr_ = l1_base + kTerminationOffset;

  // Fixed-prefix offsets shift with ring_slots_ (the layout.hpp constexprs are
  // the kernel default only), so re-derive them from the runtime slot count.
  const auto checked_offset = [this](uint64_t offset, const char *name) {
    TT_FATAL(offset <= server_l1_reserve_bytes_,
             "{} offset {} exceeds the validated server L1 reserve {}", name,
             offset, server_l1_reserve_bytes_);
    return static_cast<uint32_t>(offset);
  };
  ring_sem_base_offset_ = checked_offset(
      protocol::server_ring_sem_base_offset(ring_slots_), "ring_sem_base");
  peer_lookup_table_offset_ =
      checked_offset(protocol::server_peer_lookup_table_offset(ring_slots_),
                     "peer_lookup_table");
  host_any_get_done_monitor_offset_ = checked_offset(
      protocol::server_host_any_get_done_monitor_offset(ring_slots_),
      "host_any_get_done_monitor");
  get_credit_sem_offset_ = checked_offset(
      protocol::server_get_credit_sem_offset(ring_slots_), "get_credit_sem");

  uint32_t get_value_cb_base = 0;
  {
    std::vector<uint32_t> zero(1, 0);
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, termination_addr_, zero, tt::CoreType::WORKER);
  }

  {
    std::vector<uint8_t> bootstrap_sems(protocol::kBootstrapRequestSemTableSize,
                                        0);
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, l1_base + protocol::kBootstrapRequestSemBaseOffset,
        std::span<const uint8_t>(bootstrap_sems.data(), bootstrap_sems.size()),
        tt::CoreType::WORKER);

    protocol::EstablishResponse response_template{
        .magic = protocol::kBootstrapMagic,
        .bootstrap_version = protocol::kBootstrapProtocolVersion,
        .status =
            static_cast<uint32_t>(protocol::EstablishResponseStatus::Accepted),
        .capabilities = protocol::kRequiredCapabilities,
        .nonce_low = 0,
        .nonce_high = 0,
        .client_token = 0,
        .reserved = 0,
        .descriptor = make_descriptor_unchecked(),
        .reserved_tail = 0,
    };
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore,
        l1_base + protocol::kBootstrapResponseTemplateOffset,
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t *>(&response_template),
            sizeof(response_template)),
        tt::CoreType::WORKER);
  }

  // One 16-byte direct-data completion cacheline PER CLIENT precedes the
  // read-back scratch. Word 0 is that client's landed fence and word 1 its
  // independent large-PUT stream progress.
  direct_put_done_offset_ = checked_offset(
      protocol::server_fixed_prefix_end(ring_slots_), "direct_put_done");
  put_readback_scratch_offset_ =
      direct_put_done_offset_ + protocol::kPutDirectDoneRegionBytes;
  TT_FATAL(
      (l1_base + put_readback_scratch_offset_) % 32 == 0,
      "put_readback_scratch absolute L1 addr {} not 32-aligned (offset {})",
      l1_base + put_readback_scratch_offset_, put_readback_scratch_offset_);

  // close() leaves L1 intact. Initialize both logical counters before either
  // persistent RISC can observe a stale prior-session progress event; the
  // NCRISC repeats the word-0 initialization.
  {
    // kMaxClients landed cachelines then the shared progress cacheline; every
    // one of them starts at the configured baseline, so neither RISC can
    // observe a stale prior-session count.
    std::vector<uint8_t> counters(protocol::kPutDirectDoneRegionBytes, 0);
    for (uint32_t id = 0; id <= protocol::kMaxClients; ++id) {
      uint8_t *entry = counters.data() + id * protocol::kPutDirectDoneSemSize;
      std::memcpy(entry, &test_initial_counter_, sizeof(test_initial_counter_));
    }
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, l1_base + direct_put_done_offset_,
        std::span<const uint8_t>(counters.data(), counters.size()),
        tt::CoreType::WORKER);
  }

  commit_issued_offset_ =
      put_readback_scratch_offset_ + protocol::kPutReadbackScratchSize;
  commit_done_offset_ =
      commit_issued_offset_ + protocol::kPutCommitIssuedSemSize;
  const uint32_t put_pipeline_end =
      commit_done_offset_ + protocol::kPutCommitDoneSemSize;

  {
    std::array<uint8_t, protocol::kPutCommitCounterRegionSize> counters{};
    std::memcpy(counters.data(), &test_initial_counter_,
                sizeof(test_initial_counter_));
    std::memcpy(counters.data() + protocol::kPutCommitIssuedSemSize,
                &test_initial_counter_, sizeof(test_initial_counter_));
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, l1_base + commit_issued_offset_,
        std::span<const uint8_t>(counters.data(), counters.size()),
        tt::CoreType::WORKER);
  }

  // Directory L1 region (end of carve-out): BRISC DRAM-NoC staging buffers
  // (32-aligned and phase-matched to directory and queue DRAM pages), then the
  // 16-byte sems / mailboxes / epoch counter. See
  // ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §4. The PUT region
  // end is only 4-aligned, so round the scan-staging start up to a 32-byte
  // boundary (kDirRegionAlignPad absorbs the <=28B). Sizes are compile-time,
  // mirroring layout.hpp kDirRegionL1Size.
  const uint32_t dir_abs = ((l1_base + put_pipeline_end) + 31u) & ~31u;
  dir_scan_staging_offset_ = dir_abs - l1_base;
  dir_commit_entry_staging_offset_ =
      dir_scan_staging_offset_ + protocol::kDirScanStagingSize;
  dir_req_mailbox_offset_ =
      dir_commit_entry_staging_offset_ + protocol::kDirCommitEntryStagingSize;
  dir_req_sem_offset_ = dir_req_mailbox_offset_ + protocol::kDirReqMailboxSize;
  dir_done_sem_offset_ = dir_req_sem_offset_ + protocol::kDirReqSemSize;
  dir_resp_mailbox_offset_ = dir_done_sem_offset_ + protocol::kDirDoneSemSize;
  dir_epoch_counter_offset_ =
      dir_resp_mailbox_offset_ + protocol::kDirRespMailboxSize;
  const uint32_t dir_region_end =
      dir_epoch_counter_offset_ + protocol::kDirEpochCounterSize;

  // BRISC-owned buddy allocator state, then the host-tiering worker
  // move_req/commit_req channels. The host worker never mutates allocator
  // state. reserve_l1 already sized the carve-out.
  heap_allocator_offset_ = dir_region_end;
  move_req_sem_offset_ =
      heap_allocator_offset_ + protocol::kHeapAllocatorStateSize;
  move_req_offset_ = move_req_sem_offset_ + protocol::kMoveReqSemSize;
  commit_req_sem_offset_ = move_req_offset_ + protocol::kMoveReqSlotBytes;
  commit_req_offset_ = commit_req_sem_offset_ + protocol::kCommitReqSemSize;
  move_processed_offset_ = commit_req_offset_ + protocol::kCommitReqSlotBytes;
  stage_progress_offset_ =
      move_processed_offset_ + protocol::kMoveProcessedSemSize;
  promote_done_offset_ = stage_progress_offset_ + protocol::kStageProgressSize;
  evict_ack_sem_offset_ = promote_done_offset_ + protocol::kPromoteDoneSemSize;
  evict_ack_offset_ = evict_ack_sem_offset_ + protocol::kEvictAckSemSize;
  evict_done_offset_ = evict_ack_offset_ + protocol::kEvictAckSize;
  hash_probe_staging_offset_ = evict_done_offset_ +
                               protocol::kEvictDoneSemSize +
                               protocol::kMoveChannelsAlignPad;
  heap_metadata_staging_offset_ =
      hash_probe_staging_offset_ + protocol::kHashProbeStagingSize;
  cold_start_done_offset_ =
      heap_metadata_staging_offset_ + protocol::kHeapMetadataStagingSize;
  allocator_profile_offset_ =
      cold_start_done_offset_ + protocol::kColdStartDoneSize;
  const uint32_t server_region_end =
      allocator_profile_offset_ + protocol::kAllocatorProfileSize;

  // reserve_l1 used the same runtime layout helper and already validated the
  // physical budget.
  TT_FATAL(server_region_end <= server_l1_reserve_bytes_,
           "PUT+directory+buddy+move L1 region end {} exceeds carve-out {} "
           "(ring_slots={}). "
           "Lower KVM_SERVER_L1_RING_SLOTS.",
           server_region_end, server_l1_reserve_bytes_, ring_slots_);
  // These BRISC DRAM-NoC endpoints must be 32-aligned; the round above
  // guarantees scan staging, and the rest follow by 32-multiple strides. See
  // ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §4.
  TT_FATAL((l1_base + dir_scan_staging_offset_) % 32 == 0,
           "directory scan staging absolute L1 addr {} not 32-aligned",
           l1_base + dir_scan_staging_offset_);
  TT_FATAL((l1_base + dir_commit_entry_staging_offset_) % 32 == 0,
           "directory commit-entry staging absolute L1 addr {} not 32-aligned",
           l1_base + dir_commit_entry_staging_offset_);
  TT_FATAL((l1_base + hash_probe_staging_offset_) % 32 == 0,
           "hash probe staging absolute L1 addr {} not 32-aligned",
           l1_base + hash_probe_staging_offset_);
  TT_FATAL((l1_base + heap_metadata_staging_offset_) % 32 == 0,
           "heap metadata staging absolute L1 addr {} not 32-aligned",
           l1_base + heap_metadata_staging_offset_);
  TT_FATAL((l1_base + dir_req_mailbox_offset_) % 32 == 0,
           "directory request mailbox absolute L1 addr {} not 32-aligned",
           l1_base + dir_req_mailbox_offset_);

  const uint32_t static_server_shards = configured_static_server_shards();
  if (static_intermesh_t3k_server_) {
    TT_FATAL(worker_group_ < static_server_shards,
             "static intermesh KVM Server worker group {} is outside the {} "
             "configured shards",
             worker_group_, static_server_shards);
    if (static_server_shards > 1) {
      TT_FATAL(tt::tt_fabric::get_static_intermesh_endpoint_link_count() ==
                   StaticIntermeshMuxService::kLinks,
               "multi-shard static intermesh KVM requires exactly {} endpoint "
               "links",
               StaticIntermeshMuxService::kLinks);
    }
  }

  if (static_intermesh_t3k_server_ && static_server_shards > 1) {
    static std::mutex mux_registry_mutex;
    static std::unordered_map<MeshDevice *,
                              std::weak_ptr<StaticIntermeshMuxService>>
        mux_registry;
    {
      std::lock_guard<std::mutex> lock(mux_registry_mutex);
      auto &weak = mux_registry[server_submesh_];
      static_intermesh_mux_service_ = weak.lock();
      if (static_intermesh_mux_service_ == nullptr) {
        static_intermesh_mux_service_ =
            std::make_shared<StaticIntermeshMuxService>(server_submesh_,
                                                        static_server_shards);
        weak = static_intermesh_mux_service_;
      }
    }
    TT_FATAL(static_intermesh_mux_service_->shard_count == static_server_shards,
             "all KVM Server shards must agree on T3KNIC_SERVER_SHARDS");
  }

  std::unique_lock<std::mutex> shared_program_lock;
  if (static_intermesh_mux_service_ != nullptr) {
    shared_program_lock =
        std::unique_lock<std::mutex>(static_intermesh_mux_service_->mutex);
    TT_FATAL(!static_intermesh_mux_service_->preparation_failed,
             "the shared KVM MUX program cannot be reused; close the existing "
             "Server shards before retrying");
    TT_FATAL(!static_intermesh_mux_service_->prepared &&
                 !static_intermesh_mux_service_->launched,
             "cannot append a KVM Server shard after the shared MUX program "
             "was prepared");
  }
  FailureFlagGuard shared_program_failure(
      static_intermesh_mux_service_ != nullptr
          ? &static_intermesh_mux_service_->preparation_failed
          : nullptr);

  // The MUX firmware and every persistent shard producer must share one
  // Program. Separate persistent MeshWorkloads on the same command queue
  // serialize and would leave later producers unable to start.
  Program local_program = CreateProgram();
  Program &program = static_intermesh_mux_service_ != nullptr
                         ? static_intermesh_mux_service_->program
                         : local_program;
  CoreRange core_range(kServerCore, kServerCore);

  // GET value handoff CB (BRISC produces value chunks, NCRISC sends them;
  // on-core cb_push_back / cb_wait_front flow control). One page = one value
  // chunk (U), depth kGetValueCbDepth. The data buffer is allocated top-down
  // and aliased via set_globally_allocated_address so it does not clash with
  // the manual L1 carve-out. See
  // ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §4 / §6 GET.
  {
    const uint32_t cb_bytes =
        protocol::kGetValueCbDepth * protocol::kMaxChunkValueBytes;
    auto cb_shard =
        ShardSpecBuffer(CoreRangeSet(CoreRange(kServerCore)),
                        /*shard_shape*/ {1, 1}, ShardOrientation::ROW_MAJOR,
                        /*page_shape*/ {1, 1},
                        /*tensor2d_shape_in_pages*/ {1, 1});
    DeviceLocalBufferConfig cb_local{
        .page_size = cb_bytes,
        .buffer_type = BufferType::L1,
        .sharding_args =
            BufferShardingArgs(cb_shard, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
        .sub_device_id = std::nullopt,
    };
    ReplicatedBufferConfig cb_global{.size = cb_bytes};
    get_value_cb_buf_ =
        MeshBuffer::create(cb_global, cb_local, server_submesh_);
    auto *cb_dev = get_value_cb_buf_->get_device_buffer(MeshCoordinate(0, 0));
    get_value_cb_base = static_cast<uint32_t>(cb_dev->address());
    tt::tt_metal::CircularBufferConfig cb_get_value_config =
        tt::tt_metal::CircularBufferConfig(
            cb_bytes, {{tt::CBIndex::c_0, tt::DataFormat::UInt8}})
            .set_page_size(tt::CBIndex::c_0, protocol::kMaxChunkValueBytes)
            .set_globally_allocated_address(*cb_dev);
    CreateCircularBuffer(program, core_range, cb_get_value_config);
  }

  std::vector<uint32_t> rt_args = {
      termination_addr_,
      /*num_connections placeholder*/ 0u,
      l1_base,
      dram_value_base_,
      max_keys_,
      l1_base + direct_put_done_offset_,
      l1_base + put_readback_scratch_offset_,
      l1_base + host_any_get_done_monitor_offset_,
      l1_base + get_credit_sem_offset_,
      static_cast<uint32_t>(scalar_get_pipeline_.admission.enabled ? 1u : 0u),
      l1_base + dir_req_mailbox_offset_,
      l1_base + dir_req_sem_offset_,
      l1_base + dir_done_sem_offset_,
      l1_base + dir_resp_mailbox_offset_,
      l1_base + cold_start_done_offset_,
      l1_base + commit_issued_offset_,
      l1_base + commit_done_offset_,
      test_initial_counter_,
      session_generation_seed_,
      static_intermesh_remote_mesh_id_,
  };
  TT_ASSERT(rt_args.size() == protocol::kServerTransportFixedRtArgs);

  // Standard mode enumerates every locally reachable peer and opens one
  // connection per first-hop direction. Static T3K mode instead opens the
  // configured inter-mesh endpoint directly and maps every Galaxy fabric chip
  // ID to that one connection; its response headers carry preseeded Galaxy
  // routes. ttnn uses fabric.hpp free functions, not tt_metal/impl.
  const auto server_node =
      server_submesh_->get_fabric_node_id(MeshCoordinate(0, 0));
  std::map<tt::tt_fabric::eth_chan_directions, uint32_t> direction_to_conn_idx;
  std::vector<tt::tt_fabric::FabricNodeId> connection_peers;
  std::vector<std::vector<uint32_t>> connection_links;
  std::vector<std::tuple<uint32_t /*chip_id*/, uint32_t /*conn_idx*/,
                         uint32_t /*route_word*/>>
      peer_entries;
  std::vector<uint32_t> unregisterable_peers;
  uint32_t next_conn_idx = 0;
  uint32_t num_connections = 0;

  if (static_intermesh_t3k_server_) {
    const uint32_t endpoint_links =
        tt::tt_fabric::get_static_intermesh_endpoint_link_count();
    TT_FATAL(endpoint_links > 0 &&
                 endpoint_links <= protocol::kMaxServerConnections,
             "static inter-mesh KVM server requested {} endpoint lanes, but "
             "its transport kernel supports at most {}",
             endpoint_links, protocol::kMaxServerConnections);

    if (static_intermesh_mux_service_ != nullptr) {
      const uint32_t lane = worker_group_ % StaticIntermeshMuxService::kLinks;
      const uint32_t logical_channel =
          worker_group_ / StaticIntermeshMuxService::kLinks;
      const auto flow_control_sem = CreateSemaphore(program, kServerCore, 0);
      const auto teardown_sem = CreateSemaphore(program, kServerCore, 0);
      static_intermesh_mux_service_->configs[lane]
          ->append_client_connection_rt_args(
              static_intermesh_mux_service_->virtual_cores[lane],
              static_cast<uint8_t>(logical_channel),
              tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{
                  .flow_control_sem_id = flow_control_sem,
                  .teardown_sem_id = teardown_sem,
              },
              rt_args);
      num_connections = 1;
    } else {
      num_connections = endpoint_links;
      for (uint32_t lane = 0; lane < num_connections; ++lane) {
        tt::tt_fabric::append_static_intermesh_lane_connection_rt_args(
            lane, program, kServerCore, rt_args);
      }
    }

    for (uint32_t galaxy_chip_id = 0;
         galaxy_chip_id < protocol::kPeerLookupTableMaxEntries;
         ++galaxy_chip_id) {
      // Before a valid ESTABLISH supplies the live graph route, retain the
      // legacy 4-column estimate only as a best-effort rejection path. It must
      // still use the versioned packed format consumed by the device helper.
      const uint32_t provisional_route = protocol::pack_return_route(
          galaxy_chip_id, static_intermesh_remote_mesh_id_, galaxy_chip_id / 4,
          galaxy_chip_id % 4,
          /*ns_direction=*/1, /*ew_direction=*/1);
      peer_entries.emplace_back(galaxy_chip_id,
                                static_intermesh_mux_service_ != nullptr
                                    ? 0u
                                    : galaxy_chip_id % num_connections,
                                provisional_route);
    }
  } else {
    auto parent_mesh = server_submesh_->get_parent_mesh();

    MeshCoordinateRange all_coords(parent_mesh->shape());
    for (const MeshCoordinate &peer_coord : all_coords) {
      const auto peer_node = parent_mesh->get_fabric_node_id(peer_coord);
      // Skip self — compare both mesh_id and chip_id so multi-mesh / Galaxy
      // expansion cannot silently treat a different mesh's same-chip_id as
      // self.
      if (peer_node.mesh_id == server_node.mesh_id &&
          peer_node.chip_id == server_node.chip_id) {
        continue;
      }
      if (!is_kvcache_wire_reachable(server_node, peer_node)) {
        continue;
      }
      // The peer table is a dense array indexed by chip_id, so chips numbered
      // beyond its capacity cannot be registered. Skip them rather than
      // refusing to launch: on a mesh wider than the table the server still
      // serves every chip that fits, and a Client on a skipped chip is rejected
      // by name in its constructor instead of stalling in ESTABLISH.
      if (peer_node.chip_id >= protocol::kPeerLookupTableMaxEntries) {
        unregisterable_peers.push_back(peer_node.chip_id);
        continue;
      }
      const auto first_hop =
          kvcache_first_hop_direction(server_node, peer_node);
      TT_FATAL(first_hop.has_value(),
               "kvcache wire: peer chip {} is reachable from server chip {} "
               "but has no forwarding "
               "direction",
               peer_node.chip_id, server_node.chip_id);
      const tt::tt_fabric::eth_chan_directions dir = first_hop.value();
      const uint32_t route_word = kvcache_unicast_route_word(peer_node);
      auto peer_links = kvcache_wire_link_indices(server_node, peer_node);
      // The connection is opened toward the first peer seen on this direction
      // and then reused for every later peer sharing it. Its link plane must be
      // valid for every such destination; packets carry the remaining absolute
      // route in their headers.
      uint32_t conn_idx;
      auto it = direction_to_conn_idx.find(dir);
      if (it == direction_to_conn_idx.end()) {
        conn_idx = next_conn_idx++;
        direction_to_conn_idx[dir] = conn_idx;
        connection_peers.push_back(peer_node);
        connection_links.push_back(std::move(peer_links));
      } else {
        conn_idx = it->second;
        auto &common_links = connection_links[conn_idx];
        common_links.erase(
            std::remove_if(common_links.begin(), common_links.end(),
                           [&peer_links](uint32_t link) {
                             return std::find(peer_links.begin(),
                                              peer_links.end(),
                                              link) == peer_links.end();
                           }),
            common_links.end());
        TT_FATAL(!common_links.empty(),
                 "kvcache wire: peers sharing one first-hop direction from "
                 "server chip {} have no common "
                 "forwarding link plane (latest peer chip {})",
                 server_node.chip_id, peer_node.chip_id);
      }
      peer_entries.emplace_back(peer_node.chip_id, conn_idx, route_word);
    }
    if (!unregisterable_peers.empty()) {
      log_warning(
          tt::LogOp,
          "kvcache_manager: {} reachable chips are numbered at or beyond the "
          "{}-entry peer table and will not "
          "be served by the ServerRuntimeCore on chip {} (first: chip {}).",
          unregisterable_peers.size(), protocol::kPeerLookupTableMaxEntries,
          server_node.chip_id, unregisterable_peers.front());
    }
    num_connections = static_cast<uint32_t>(direction_to_conn_idx.size());
    // The kernel sizes its connection array by protocol::kMaxServerConnections;
    // one connection per eth direction keeps this well inside the bound no
    // matter how many peers were enumerated.
    TT_FATAL(num_connections <= protocol::kMaxServerConnections,
             "kvcache wire: {} fabric connections needed from server chip {} "
             "but the server kernel holds "
             "at most {}",
             num_connections, server_node.chip_id,
             protocol::kMaxServerConnections);
    TT_ASSERT(connection_peers.size() == num_connections &&
              connection_links.size() == num_connections);
    for (uint32_t conn_idx = 0; conn_idx < num_connections; ++conn_idx) {
      TT_ASSERT(!connection_links[conn_idx].empty());
      tt::tt_fabric::append_fabric_connection_rt_args(
          server_node, connection_peers[conn_idx],
          connection_links[conn_idx].front(), program, kServerCore, rt_args);
    }
  }
  rt_args[1] = num_connections;

  // The kernel is created here, after enumeration, because its defines depend
  // on the outcome: with no reachable peer it serves nothing and only waits for
  // termination, and compiling the request handlers into that build overruns
  // the NCRISC IRAM ceiling (the num_connections==0 branch cannot help — it is
  // a runtime value, so the handlers are emitted regardless). Nothing between
  // the rt_args above and here needs the kernel handle;
  // append_fabric_connection_rt_args takes the program.
  auto directory_kernel_defines = kvcache_kernel_defines(
      ring_slots_, value_ring_depth_, KvcacheKernelTarget::ServerDirectory,
      profile_detail_, directory_policy);
  const bool instrumented_directory =
      directory_kernel_defines.contains("KVM_SERVER_ALLOCATOR_PROFILE") ||
      directory_kernel_defines.contains("KVM_PROFILE_DIRECTORY_PUT") ||
      directory_kernel_defines.contains("KVM_PROFILE_DIRECTORY_GET") ||
      directory_kernel_defines.contains("KVM_PROFILE_DIRECTORY_MOVE");
  const auto transport_route_mode =
      static_intermesh_t3k_server_
          ? KvcacheRouteMode::StaticIntermeshT3kToGalaxy
          : KvcacheRouteMode::Standard;
  auto transport_kernel_defines = kvcache_kernel_defines(
      ring_slots_, value_ring_depth_, KvcacheKernelTarget::ServerTransport,
      profile_detail_, std::nullopt, transport_route_mode);
  if (static_intermesh_mux_service_ != nullptr) {
    transport_kernel_defines.emplace("KVM_SERVER_USE_MUX_V2", "1");
  }
  if (test_delay_establish_ack_) {
    transport_kernel_defines.emplace("KVM_INJECT_DELAY_ESTABLISH_RESPONSE",
                                     "1");
  }
  if (test_drop_first_establish_response_) {
    transport_kernel_defines.emplace("KVM_INJECT_DROP_FIRST_ESTABLISH_RESPONSE",
                                     "1");
  }
  KernelBuildOptLevel transport_opt_level = KernelBuildOptLevel::O2;
  if (instrumented_directory) {
    // Data-movement opt levels are absent from the kernel cache hash, so tag
    // the Oz variant.
    transport_kernel_defines.emplace("KVM_SERVER_DIAGNOSTIC_SIZE_BUILD", "1");
    directory_kernel_defines.emplace("KVM_SERVER_DIAGNOSTIC_SIZE_BUILD", "1");
    transport_opt_level = KernelBuildOptLevel::Oz;
  } else if (static_intermesh_mux_service_ != nullptr) {
    // FabricMuxV2Sender adds per-channel handshake and credit state to the
    // near-limit production transport image. Oz keeps this variant inside the
    // TENSIX kernel-config budget; tag it because the opt level is not cached.
    transport_kernel_defines.emplace("KVM_STATIC_INTERMESH_MUX_OZ_BUILD", "1");
    transport_opt_level = KernelBuildOptLevel::Oz;
  } else if (static_intermesh_t3k_server_) {
    // The preseeded response route pushes the production NCRISC image past the
    // TENSIX kernel-config budget at O2. Tag this Os variant because opt levels
    // are not cache-keyed.
    transport_kernel_defines.emplace("KVM_STATIC_INTERMESH_SERVER_OS_BUILD",
                                     "1");
    transport_opt_level = KernelBuildOptLevel::Os;
  } else if (is_kvcache_2d_fabric()) {
    // Absolute 2D routing plus the full operation set exceeds the kernel-config
    // budget at O2. Keep direct-neighbour 1D at O2 and tag this variant because
    // opt levels are not cache-keyed.
    transport_kernel_defines.emplace("KVM_FABRIC_2D_SERVER_OS_BUILD", "1");
    transport_opt_level = KernelBuildOptLevel::Os;
  }
  if (num_connections == 0) {
    transport_kernel_defines.emplace("KVM_SERVER_IDLE_ONLY", "1");
    log_debug(tt::LogOp,
              "kvcache_manager: server chip {} has no reachable peer; building "
              "the idle-only server kernel",
              server_node.chip_id);
  }

  KernelHandle kernel_id = CreateKernel(
      program, kServerKernelPath, core_range,
      DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                         .noc = NOC::RISCV_1_default,
                         .defines = transport_kernel_defines,
                         .opt_level = transport_opt_level});

  // Build the host-side peer_lookup_table and ship it to server submesh L1.
  {
    std::vector<uint32_t> table(protocol::kPeerLookupTableMaxEntries * 2, 0);
    for (uint32_t i = 0; i < protocol::kPeerLookupTableMaxEntries; ++i) {
      table[2 * i] = protocol::kPeerLookupEmptySentinel;
      table[2 * i + 1] = 0;
    }
    for (const auto &[chip_id, conn_idx, route_word] : peer_entries) {
      TT_FATAL(chip_id < protocol::kPeerLookupTableMaxEntries,
               "peer chip_id {} exceeds kPeerLookupTableMaxEntries {}", chip_id,
               protocol::kPeerLookupTableMaxEntries);
      table[2 * chip_id] = conn_idx;
      table[2 * chip_id + 1] = route_word;
    }
    tt::tt_metal::detail::WriteToDeviceL1(device, kServerCore,
                                          l1_base + peer_lookup_table_offset_,
                                          table, tt::CoreType::WORKER);
  }

  SetRuntimeArgs(program, kernel_id, kServerCore, rt_args);

  // BRISC (RISCV_0) DRAM worker kernel on the same core: GET/EXISTS/REMOVE
  // lookup + PUT-commit / promote directory authority. Shares the termination
  // word with the NCRISC kernel; no fabric. Instrumented (allocator-profile /
  // directory-detail) builds carry extra counters in BOTH kernels; at O2 the
  // combined program overflows the TENSIX kernel config buffer (~72.5 KB vs
  // 71680). Oz keeps the diagnostic build fitting; production builds stay O2.
  // The size-build tag must go on this kernel too: opt levels are absent from
  // the kernel cache hash, so an untagged Oz variant would silently reuse a
  // cached O2 image.
  KernelHandle dram_worker_kernel_id = CreateKernel(
      program, kDramWorkerKernelPath, core_range,
      DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                         .noc = NOC::RISCV_0_default,
                         .defines = std::move(directory_kernel_defines),
                         .opt_level = instrumented_directory
                                          ? KernelBuildOptLevel::Oz
                                          : KernelBuildOptLevel::O2});
  std::vector<uint32_t> dir_rt_args = {
      termination_addr_,
      dram_directory_base_,
      max_keys_,
      dir_entry_bytes_,
      max_key_len_,
      l1_base + dir_scan_staging_offset_,
      l1_base + dir_req_mailbox_offset_,
      l1_base + dir_req_sem_offset_,
      l1_base + dir_done_sem_offset_,
      l1_base + dir_resp_mailbox_offset_,
      l1_base + dir_epoch_counter_offset_,
      dram_put_host_queue_base_,
      l1_base + commit_done_offset_,
      l1_base + dir_commit_entry_staging_offset_,
      l1_base + move_req_sem_offset_,
      l1_base + move_req_offset_,
      l1_base + commit_req_sem_offset_,
      l1_base + commit_req_offset_,
      value_heap_bytes_,
      l1_base + heap_allocator_offset_,
      l1_base + move_processed_offset_,
      dram_value_base_,
      stage_pcie_xy_enc_,
      stage_addr_lo_,
      stage_addr_hi_,
      static_cast<uint32_t>(stage_noc_ok_ ? 1u : 0u),
      stage_completion_offset_,
      static_cast<uint32_t>(
          scalar_get_pipeline_.mapped_completion_enabled ? 1u : 0u),
      get_completion_offset_,
      l1_base + host_any_get_done_monitor_offset_,
      pool_meta_read_off_,
      l1_base + stage_progress_offset_,
      l1_base + promote_done_offset_,
      l1_base + evict_ack_sem_offset_,
      l1_base + evict_ack_offset_,
      l1_base + evict_done_offset_,
      dram_hashmap_base_,
      num_buckets_,
      l1_base + hash_probe_staging_offset_,
      dram_heap_metadata_base_,
      l1_base + heap_metadata_staging_offset_,
      l1_base + cold_start_done_offset_,
      test_first_put_commit_delay_cycles_,
      test_initial_counter_,
      // Shared large-PUT stream progress: one cacheline after the per-client
      // landed fences. The BRISC watches this single address, which is well
      // defined because PUTs are serialized through the NCRISC dispatch loop
      // (see PutSessionConfig::progress_l1_addr).
      l1_base + protocol::put_stream_progress_offset(direct_put_done_offset_),
      l1_base + allocator_profile_offset_,
      residency_lru_base_,
      static_cast<uint32_t>(scalar_get_pipeline_.admission.enabled ? 1u : 0u),
      static_cast<uint32_t>(admission_virtual.x),
      static_cast<uint32_t>(admission_virtual.y),
      scalar_get_pipeline_.admission.l1_base +
          protocol::kGetAdmissionReadyOffset,
      scalar_get_pipeline_.admission.l1_base +
          protocol::kGetAdmissionDescriptorOffset,
      get_value_cb_base,
      l1_base + get_credit_sem_offset_,
  };
  SetRuntimeArgs(program, dram_worker_kernel_id, kServerCore, dir_rt_args);

  if (scalar_get_pipeline_.ingress.enabled) {
    TT_ASSERT(scalar_get_pipeline_.ingress.l1 != nullptr);
    const CoreCoord server_virtual =
        device->worker_core_from_logical_core(kServerCore);
    KernelHandle ingress_kernel_id = CreateKernel(
        program, kGetIngressKernelPath,
        CoreRange(kGetIngressCore, kGetIngressCore),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .defines = {},
                           .opt_level = KernelBuildOptLevel::O2});
    std::vector<uint32_t> ingress_rt_args = {
        scalar_get_pipeline_.ingress.termination_addr,
        scalar_get_pipeline_.ingress.l1_base +
            protocol::kGetIngressDoorbellOffset,
        scalar_get_pipeline_.ingress.l1_base +
            protocol::kGetIngressScratchOffset,
        stage_pcie_xy_enc_,
        stage_addr_lo_,
        stage_addr_hi_,
        get_ingress_offset_,
        static_cast<uint32_t>(server_virtual.x),
        static_cast<uint32_t>(server_virtual.y),
        l1_base + protocol::kRingBaseOffset,
        l1_base + ring_sem_base_offset_,
        ring_slots_,
        test_initial_counter_,
    };
    SetRuntimeArgs(program, ingress_kernel_id, kGetIngressCore,
                   ingress_rt_args);

    std::array<uint8_t, protocol::kGetIngressDoorbellOffset +
                            protocol::kGetIngressDoorbellSize>
        ingress_init{};
    std::memcpy(ingress_init.data() + protocol::kGetIngressDoorbellOffset,
                &test_initial_counter_, sizeof(test_initial_counter_));
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kGetIngressCore, scalar_get_pipeline_.ingress.l1_base,
        std::span<const uint8_t>(ingress_init.data(), ingress_init.size()),
        tt::CoreType::WORKER);
  }

  if (scalar_get_pipeline_.admission.enabled) {
    TT_ASSERT(scalar_get_pipeline_.admission.l1 != nullptr);
    const CoreCoord server_virtual =
        device->worker_core_from_logical_core(kServerCore);
    KernelHandle admission_kernel_id = CreateKernel(
        program, kGetAdmissionKernelPath,
        CoreRange(kGetAdmissionCore, kGetAdmissionCore),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .defines = {},
                           .opt_level = KernelBuildOptLevel::O2});
    std::vector<uint32_t> admission_rt_args = {
        scalar_get_pipeline_.admission.termination_addr,
        scalar_get_pipeline_.admission.l1_base +
            protocol::kGetAdmissionReadyOffset,
        scalar_get_pipeline_.admission.l1_base +
            protocol::kGetAdmissionDescriptorOffset,
        scalar_get_pipeline_.admission.l1_base +
            protocol::kGetAdmissionScratchOffset,
        static_cast<uint32_t>(server_virtual.x),
        static_cast<uint32_t>(server_virtual.y),
        get_value_cb_base,
        l1_base + get_credit_sem_offset_,
        dram_value_base_,
        test_initial_counter_,
    };
    SetRuntimeArgs(program, admission_kernel_id, kGetAdmissionCore,
                   admission_rt_args);

    std::array<uint8_t, protocol::kGetAdmissionDescriptorOffset +
                            sizeof(protocol::GetAdmissionDescriptor)>
        admission_init{};
    std::memcpy(admission_init.data() + protocol::kGetAdmissionReadyOffset,
                &test_initial_counter_, sizeof(test_initial_counter_));
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kGetAdmissionCore, scalar_get_pipeline_.admission.l1_base,
        std::span<const uint8_t>(admission_init.data(), admission_init.size()),
        tt::CoreType::WORKER);
  }

  {
    std::array<uint8_t, protocol::kGetCreditSemSize> admission_progress{};
    std::memcpy(admission_progress.data() +
                    protocol::kGetAdmissionSourceDoneWord * sizeof(uint32_t),
                &test_initial_counter_, sizeof(test_initial_counter_));
    std::memcpy(admission_progress.data() +
                    protocol::kGetAdmissionWriteDoneWord * sizeof(uint32_t),
                &test_initial_counter_, sizeof(test_initial_counter_));
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, l1_base + get_credit_sem_offset_,
        std::span<const uint8_t>(admission_progress.data(),
                                 admission_progress.size()),
        tt::CoreType::WORKER);
  }

  // Publish the worker-visible baseline before asynchronous cold-start can race
  // its first poll. See ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md
  // §8 and §9.
  {
    std::array<uint8_t, protocol::kMoveReqSemSize> zsem{};
    std::memcpy(zsem.data(), &test_initial_counter_,
                sizeof(test_initial_counter_));
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, l1_base + move_req_sem_offset_,
        std::span<const uint8_t>(zsem.data(), zsem.size()),
        tt::CoreType::WORKER);
  }

  // A reused L1 completion count must not satisfy the worker's first wait.
  {
    std::array<uint8_t, protocol::kPromoteDoneSemSize> zsem{};
    std::memcpy(zsem.data(), &test_initial_counter_,
                sizeof(test_initial_counter_));
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, l1_base + promote_done_offset_,
        std::span<const uint8_t>(zsem.data(), zsem.size()),
        tt::CoreType::WORKER);
  }

  // Clear the reused-L1 eviction handshake before the worker can observe it.
  {
    std::array<uint8_t, protocol::kEvictAckSemSize + protocol::kEvictAckSize +
                            protocol::kEvictDoneSemSize>
        zhandshake{};
    std::memcpy(zhandshake.data(), &test_initial_counter_,
                sizeof(test_initial_counter_));
    std::memcpy(zhandshake.data() + protocol::kEvictAckSemSize +
                    protocol::kEvictAckSize,
                &test_initial_counter_, sizeof(test_initial_counter_));
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, l1_base + evict_ack_sem_offset_,
        std::span<const uint8_t>(zhandshake.data(), zhandshake.size()),
        tt::CoreType::WORKER);
  }

  // Reset the cold-start gate and profiling state before either persistent RISC
  // can serve.
  {
    std::array<uint8_t,
               protocol::kColdStartDoneSize + protocol::kAllocatorProfileSize>
        zcs{};
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kServerCore, l1_base + cold_start_done_offset_,
        std::span<const uint8_t>(zcs.data(), zcs.size()), tt::CoreType::WORKER);
  }

  if (static_intermesh_mux_service_ != nullptr) {
    auto &service = *static_intermesh_mux_service_;
    TT_ASSERT(shared_program_lock.owns_lock());
    service.servers.push_back(this);
    if (service.servers.size() < service.shard_count) {
      shared_program_failure.release();
      return;
    }
    TT_FATAL(service.servers.size() == service.shard_count,
             "shared KVM MUX program expected {} shards, got {}",
             service.shard_count, service.servers.size());

    // JIT failure is definitely unsent. MUXes and all producers are compiled
    // together so each persistent kernel can make progress concurrently.
    try {
      tt::tt_metal::detail::CompileProgram(server_submesh_, service.program);
      service.workload = std::make_unique<MeshWorkload>();
      service.workload->add_program(MeshCoordinateRange(MeshCoordinate(0, 0)),
                                    std::move(service.program));
      service.prepared = true;
    } catch (...) {
      service.preparation_failed = true;
      service.servers.erase(
          std::remove(service.servers.begin(), service.servers.end(), this),
          service.servers.end());
      throw;
    }
    shared_program_failure.release();
    return;
  }

  // JIT failure is definitely-unsent. Once EnqueueMeshWorkload is called, the
  // CQ API cannot report whether a prefix was submitted, so every exception
  // beyond that boundary quarantines.
  tt::tt_metal::detail::CompileProgram(server_submesh_, local_program);

  workload_ = std::make_unique<MeshWorkload>();
  MeshCoordinateRange chip_range(MeshCoordinate(0, 0));
  workload_->add_program(chip_range, std::move(local_program));
}

void ServerRuntimeCore::launch_server_kernel() {
  if (static_intermesh_mux_service_ != nullptr) {
    auto &service = *static_intermesh_mux_service_;
    std::lock_guard<std::mutex> lock(service.mutex);
    if (!service.prepared) {
      // The final configured shard compiles and submits the shared workload.
      // Earlier Server objects may enter their host polling loop against the
      // initialized idle L1 state in the meantime.
      return;
    }
    TT_FATAL(!service.preparation_failed,
             "the shared KVM MUX program became unusable before launch");
    TT_FATAL(service.servers.size() == service.shard_count,
             "shared KVM MUX launch expected {} live shards, got {}",
             service.shard_count, service.servers.size());
    if (service.launched) {
      return;
    }
    TT_FATAL(service.workload != nullptr,
             "shared KVM MUX workload was not prepared");

    for (ServerRuntimeCore *server : service.servers) {
      server->server_launch_may_have_started_ = true;
    }
    try {
      EnqueueMeshWorkload(server_submesh_->mesh_command_queue(),
                          *service.workload, /*blocking=*/false);
      service.launched = true;
      for (ServerRuntimeCore *server : service.servers) {
        if (server->test_fail_launch_after_enqueue_) {
          TT_THROW("injected ServerRuntimeCore launch failure after enqueue");
        }
      }
    } catch (...) {
      // Submission outcome is unknown. Every shard shares this workload, so
      // retaining only the constructor that observed the exception would let
      // another wrapper free memory still visible to a persistent kernel.
      for (ServerRuntimeCore *server : service.servers) {
        server->server_workload_running_.store(false,
                                               std::memory_order_release);
        server->external_host_io_cancelled_.store(true,
                                                  std::memory_order_release);
        server->lifecycle_state_.store(LifecycleState::Quarantined,
                                       std::memory_order_release);
        server->quarantine_failed_teardown();
      }
      throw;
    }
    for (ServerRuntimeCore *server : service.servers) {
      server->server_workload_running_.store(true, std::memory_order_release);
    }
    return;
  }

  TT_FATAL(workload_ != nullptr,
           "ServerRuntimeCore workload must be prepared before launch");

  // From this point onward an exception cannot prove that no command reached
  // the device. Keep workload_ and every referenced allocation alive until
  // reset if enqueue or any later step fails.
  server_launch_may_have_started_ = true;
  EnqueueMeshWorkload(server_submesh_->mesh_command_queue(), *workload_,
                      /*blocking=*/false);

  if (test_fail_launch_after_enqueue_) {
    TT_THROW("injected ServerRuntimeCore launch failure after enqueue");
  }
  server_workload_running_.store(true, std::memory_order_release);
}

void ServerRuntimeCore::launch_internal_worker_and_server() {
  // Construct the worker before launching the persistent kernel, so std::thread
  // allocation failure cannot orphan a kernel. A small gate keeps it off the
  // device until enqueue has succeeded.
  constexpr uint32_t kWorkerWaiting = 0;
  constexpr uint32_t kWorkerRun = 1;
  constexpr uint32_t kWorkerCancel = 2;
  auto start_gate = std::make_shared<std::atomic<uint32_t>>(kWorkerWaiting);
  worker_stop_.store(false, std::memory_order_relaxed);
  auto worker = std::make_unique<std::thread>([this, start_gate] {
    uint32_t action = kWorkerWaiting;
    while ((action = start_gate->load(std::memory_order_acquire)) ==
           kWorkerWaiting) {
      start_gate->wait(kWorkerWaiting, std::memory_order_acquire);
    }
    if (action == kWorkerRun) {
      try {
        if (test_fail_tiering_worker_) {
          TT_THROW("injected tiering worker failure");
        }
        tiering_worker_loop();
      } catch (...) {
        record_tiering_worker_failure(std::current_exception());
        worker_stop_.store(true, std::memory_order_release);
      }
    }
  });

  try {
    launch_server_kernel();
  } catch (...) {
    worker_stop_.store(true, std::memory_order_relaxed);
    start_gate->store(kWorkerCancel, std::memory_order_release);
    start_gate->notify_one();
    worker->join();
    throw;
  }

  tiering_worker_ = std::move(worker);
  start_gate->store(kWorkerRun, std::memory_order_release);
  start_gate->notify_one();
}

void ServerRuntimeCore::terminate_server_kernel() {
  if (workload_ == nullptr && static_intermesh_mux_service_ == nullptr) {
    return;
  }

  // close() enters Closing, then drains issued PUTs while keeping the tiering
  // worker alive.
  if (kvcache_profile_) {
    log_info(tt::LogOp,
             "KVCACHE_HOST[waits] drain_count={} drain_total_ns={} "
             "sync_commit_count={} sync_commit_total_ns={}",
             drain_wait_count_.load(std::memory_order_relaxed),
             drain_wait_ns_.load(std::memory_order_relaxed),
             sync_commit_wait_count_.load(std::memory_order_relaxed),
             sync_commit_wait_ns_.load(std::memory_order_relaxed));
    // Host-source operation per-phase breakdown: reqread / gather / dma /
    // commit. stage_copy and stage_pub SUBDIVIDE dma (fused staging only), so
    // do not add them to the phase total.
    log_info(
        tt::LogOp,
        "KVCACHE_HOST[promote] count={} reqread_ns={} gather_ns={} dma_ns={} "
        "commit_ns={} "
        "stage_copy_ns={} stage_pub_ns={} completion_wait_ns={} "
        "completion_l1_fallbacks={}",
        promote_count_.load(std::memory_order_relaxed),
        promote_reqread_ns_.load(std::memory_order_relaxed),
        promote_gather_ns_.load(std::memory_order_relaxed),
        promote_dma_ns_.load(std::memory_order_relaxed),
        promote_commit_ns_.load(std::memory_order_relaxed),
        promote_stage_copy_ns_.load(std::memory_order_relaxed),
        promote_stage_pub_ns_.load(std::memory_order_relaxed),
        promote_completion_wait_ns_.load(std::memory_order_relaxed),
        promote_completion_l1_fallback_count_.load(std::memory_order_relaxed));
    log_info(tt::LogOp,
             "KVCACHE_HOST[write_through] count={} reqread_ns={} reserve_ns={} "
             "ack_ns={} push_wait_ns={} "
             "publish_ns={} processed_ns={} worker_ns={} complete_ns={}",
             move_to_host_count_.load(std::memory_order_relaxed),
             write_through_reqread_ns_.load(std::memory_order_relaxed),
             write_through_reserve_ns_.load(std::memory_order_relaxed),
             write_through_ack_ns_.load(std::memory_order_relaxed),
             write_through_push_wait_ns_.load(std::memory_order_relaxed),
             write_through_publish_ns_.load(std::memory_order_relaxed),
             write_through_processed_ns_.load(std::memory_order_relaxed),
             move_to_host_worker_ns_.load(std::memory_order_relaxed),
             move_to_host_complete_ns_.load(std::memory_order_relaxed));
  }

  if (static_intermesh_mux_service_ != nullptr) {
    auto &service = *static_intermesh_mux_service_;
    IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
    TT_FATAL(device != nullptr, "server_submesh has no chip at (0,0)");

    std::lock_guard<std::mutex> lock(service.mutex);
    if (!service.launched) {
      server_workload_running_.store(false, std::memory_order_release);
      // The shared Program already contains this shard's kernels and resource
      // addresses. It cannot safely accept a replacement shard; let the weak
      // registry expire after every member closes, then construct a fresh set.
      service.preparation_failed = true;
      const auto it =
          std::find(service.servers.begin(), service.servers.end(), this);
      if (it != service.servers.end()) {
        service.servers.erase(it);
      }
      return;
    }

    if (!server_termination_signalled_) {
      std::vector<uint32_t> signal(1, kImmediateTerminationSignal);
      tt::tt_metal::detail::WriteToDeviceL1(
          device, kServerCore, termination_addr_, signal, tt::CoreType::WORKER);
      if (scalar_get_pipeline_.ingress.enabled) {
        tt::tt_metal::detail::WriteToDeviceL1(
            device, kGetIngressCore,
            scalar_get_pipeline_.ingress.termination_addr, signal,
            tt::CoreType::WORKER);
      }
      if (scalar_get_pipeline_.admission.enabled) {
        tt::tt_metal::detail::WriteToDeviceL1(
            device, kGetAdmissionCore,
            scalar_get_pipeline_.admission.termination_addr, signal,
            tt::CoreType::WORKER);
      }

      server_termination_signalled_ = true;
      ++service.terminated_servers;
    }
    server_workload_running_.store(false, std::memory_order_release);
    TT_FATAL(service.terminated_servers <= service.shard_count,
             "shared KVM Server shard termination count exceeded its shard "
             "count");
    if (service.terminated_servers == service.shard_count &&
        service.workload != nullptr) {
      Finish(server_submesh_->mesh_command_queue());
      service.workload.reset();
      service.retired_kernel_resources.clear();
      service.launched = false;
      if (test_terminate_failures_remaining_ > 0) {
        --test_terminate_failures_remaining_;
        test_quarantine_safe_to_release_ = true;
        TT_THROW("kvcache_manager: injected terminate failure after "
                 "successful Finish");
      }
    }
    return;
  }

  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr, "server_submesh has no chip at (0,0)");

  // Client teardown quiesces its CQ, GET tail, and PUT commits before
  // ServerRuntimeCore close. Fabric's graceful signal is non-functional, so the
  // persistent workers terminate immediately after that host boundary. The
  // persistent kernels share Fabric's host/device termination-word ABI.
  std::vector<uint32_t> signal(1, kImmediateTerminationSignal);
  tt::tt_metal::detail::WriteToDeviceL1(device, kServerCore, termination_addr_,
                                        signal, tt::CoreType::WORKER);
  if (scalar_get_pipeline_.ingress.enabled) {
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kGetIngressCore, scalar_get_pipeline_.ingress.termination_addr,
        signal, tt::CoreType::WORKER);
  }
  if (scalar_get_pipeline_.admission.enabled) {
    tt::tt_metal::detail::WriteToDeviceL1(
        device, kGetAdmissionCore,
        scalar_get_pipeline_.admission.termination_addr, signal,
        tt::CoreType::WORKER);
  }

  // Drain the server_submesh's own CQ (separate from the parent mesh's CQ).
  Finish(server_submesh_->mesh_command_queue());
  server_workload_running_.store(false, std::memory_order_release);

  if (test_terminate_failures_remaining_ > 0) {
    --test_terminate_failures_remaining_;
    test_quarantine_safe_to_release_ = true;
    TT_THROW(
        "kvcache_manager: injected terminate failure after successful Finish");
  }

  workload_.reset();
}

void ServerRuntimeCore::retain_static_intermesh_mux_resources(
    std::shared_ptr<void> resources) {
  if (static_intermesh_mux_service_ == nullptr || resources == nullptr) {
    return;
  }

  auto &service = *static_intermesh_mux_service_;
  std::lock_guard<std::mutex> lock(service.mutex);
  if (service.workload != nullptr) {
    service.retired_kernel_resources.push_back(std::move(resources));
  }
}

} // namespace kvcache_manager::detail::server_runtime
