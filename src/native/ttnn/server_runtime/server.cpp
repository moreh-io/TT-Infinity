// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#include "server_runtime/server_runtime_core.hpp"

#include "bootstrap_contract.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
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
#include <tt-metalium/tt_metal_profiler.hpp>

#include "ttnn/operations/kvcache_manager/common/config.hpp"
#include "ttnn/operations/kvcache_manager/common/fabric_helpers.hpp"
#include "ttnn/operations/kvcache_manager/common/protocol/layout.hpp"
#include "ttnn/operations/kvcache_manager/common/protocol/packet.hpp"

namespace kvcache_manager::detail::server_runtime {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using ttnn_kvm::is_kvcache_2d_fabric;
using ttnn_kvm::validate_kvcache_fabric_config;

namespace {
using RegistryKey = std::pair<MeshDevice *, uint32_t>;
std::mutex registry_mutex;
std::set<RegistryKey> active_slots;

// Keep the pending window long enough for a host-side counter read to observe
// it reliably.
constexpr uint32_t kTestFirstPutCommitDelayCycles = 250'000'000;
std::atomic<uint32_t> server_generation_counter{1};

uint32_t make_server_generation_seed() {
  const uint64_t now = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  const uint32_t counter =
      server_generation_counter.fetch_add(1, std::memory_order_relaxed);
  uint32_t seed = static_cast<uint32_t>(
      (now ^ (static_cast<uint64_t>(counter) * 0x9E3779B9u)) &
      protocol::kClientGenerationMask);
  return seed == 0 ? 1 : seed;
}

uint32_t configured_server_worker_groups() {
  const auto parsed = config::read_u64_env("T3KNIC_SERVER_SHARDS").value_or(1);
  TT_FATAL(parsed >= 1 && parsed <= 8,
           "T3KNIC_SERVER_SHARDS must be in [1, 8], got {}", parsed);
  return static_cast<uint32_t>(parsed);
}

// Directory entry count is the DRAM-resident key retention limit. It is
// independent of both PUT pipeline depth and value-heap capacity.
uint32_t parse_max_keys() {
  constexpr uint32_t kDefault = protocol::kMaxKeysDefault;
  constexpr uint32_t kCap = protocol::kMaxKeysHardCap;
  const auto parsed = config::read_u64_env("KVM_SERVER_DRAM_MAX_KEYS");
  if (!parsed.has_value()) {
    return kDefault;
  }
  TT_FATAL(*parsed > 0, "KVM_SERVER_DRAM_MAX_KEYS must be positive");
  if (*parsed > kCap) {
    log_warning(tt::LogOp,
                "KVM_SERVER_DRAM_MAX_KEYS={} exceeds hard cap {}; clamping.",
                *parsed, kCap);
    return kCap;
  }
  return static_cast<uint32_t>(*parsed);
}

// Value heap total byte cap, rounded down to a whole number of U =
// kMaxChunkValueBytes units and clamped to >= the aligned storage span of one
// maximum-size value.
uint32_t parse_value_heap_bytes(uint32_t max_value_bytes) {
  constexpr uint32_t kU = protocol::kMaxChunkValueBytes;
  const uint32_t kMin = protocol::buddy_block_units_for(max_value_bytes) * kU;
  constexpr uint32_t kDefault = protocol::kValueHeapBytesDefault;
  uint64_t bytes = config::read_u64_env("KVM_SERVER_DRAM_VALUE_HEAP_BYTES")
                       .value_or(kDefault);
  TT_FATAL(bytes > 0, "KVM_SERVER_DRAM_VALUE_HEAP_BYTES must be positive");
  if (bytes > protocol::kMaxChunkAlignedValueBytes) {
    log_warning(tt::LogOp,
                "KVM_SERVER_DRAM_VALUE_HEAP_BYTES={} exceeds the largest "
                "chunk-aligned uint32 value {}; clamping.",
                bytes, protocol::kMaxChunkAlignedValueBytes);
    bytes = protocol::kMaxChunkAlignedValueBytes;
  }
  const uint32_t narrowed_bytes = static_cast<uint32_t>(bytes);
  const uint32_t rounded_bytes = (narrowed_bytes / kU) * kU;
  if (rounded_bytes < kMin) {
    log_warning(tt::LogOp,
                "KVM_SERVER_DRAM_VALUE_HEAP_BYTES={} is below the "
                "buddy-rounded single-value span {}; raising to {}.",
                narrowed_bytes, kMin, kMin);
  }
  return protocol::normalize_value_heap_bytes(narrowed_bytes, max_value_bytes);
}

// Idle host-worker and targeted-copy fallback poll interval, clamped to
// [1 us, 1 s]. Active mapped transfers and explicit durability waits do not
// scheduler-sleep.
uint32_t parse_poll_us() {
  constexpr uint32_t kDefault = 10;
  constexpr uint32_t kMaxUs = 1'000'000; // 1 s clamp
  const auto parsed = config::read_u64_env("KVM_SERVER_HOST_POLL_US");
  if (!parsed.has_value()) {
    return kDefault;
  }
  TT_FATAL(*parsed > 0, "KVM_SERVER_HOST_POLL_US must be positive");
  if (*parsed > kMaxUs) {
    log_warning(tt::LogOp,
                "KVM_SERVER_HOST_POLL_US={} exceeds {} μs; clamping.", *parsed,
                kMaxUs);
    return kMaxUs;
  }
  return static_cast<uint32_t>(*parsed);
}

} // namespace

// A persistent kernel whose teardown failed twice may still dereference these
// allocations. The holder is allocated before launch, and resource declaration
// order keeps the submesh alive until last.
struct ServerRuntimeCore::FailedTeardownResources {
  std::shared_ptr<MeshDevice> server_submesh;
  MeshDevice *registry_key = nullptr;
  uint32_t worker_group = 0;
  std::shared_ptr<MeshBuffer> server_l1;
  std::shared_ptr<MeshBuffer> get_value_cb;
  std::shared_ptr<MeshBuffer> get_ingress_l1;
  std::shared_ptr<MeshBuffer> get_admission_l1;
  std::shared_ptr<MeshBuffer> residency_lru;
  std::shared_ptr<MeshBuffer> dram_value;
  std::shared_ptr<MeshBuffer> dram_put_host_queue;
  std::shared_ptr<MeshBuffer> dram_directory;
  std::shared_ptr<MeshBuffer> dram_hashmap;
  std::shared_ptr<MeshBuffer> dram_heap_metadata;
  std::shared_ptr<SharedHostArena> stage_arena;
  std::unique_ptr<MeshWorkload> workload;
  std::shared_ptr<StaticIntermeshMuxService> static_intermesh_mux_service;
  bool safe_to_release_for_test = false;
  FailedTeardownResources *next = nullptr;
};

std::atomic<ServerRuntimeCore::FailedTeardownResources *>
    ServerRuntimeCore::failed_teardown_head_{nullptr};

std::unique_lock<std::mutex> ServerRuntimeCore::acquire_host_operation() {
  return std::unique_lock<std::mutex>(host_operation_mutex_);
}

ServerRuntimeCore::ServerRuntimeCore(MeshDevice *server_submesh,
                                     bool static_intermesh_t3k_server,
                                     uint32_t static_intermesh_remote_mesh_id,
                                     uint32_t worker_group)
    : ServerRuntimeCore(server_submesh, TestOptions{},
                        static_intermesh_t3k_server,
                        static_intermesh_remote_mesh_id, worker_group) {}

ServerRuntimeCore::ServerRuntimeCore(MeshDevice *server_submesh,
                                     TestOptions test_options)
    : ServerRuntimeCore(server_submesh, test_options, false, 0, 0) {}

ServerRuntimeCore::ServerRuntimeCore(MeshDevice *server_submesh,
                                     TestOptions test_options,
                                     bool static_intermesh_t3k_server,
                                     uint32_t static_intermesh_remote_mesh_id,
                                     uint32_t worker_group,
                                     HostPolicyMode host_policy_mode)
    : server_submesh_(server_submesh),
      static_intermesh_t3k_server_(static_intermesh_t3k_server),
      static_intermesh_remote_mesh_id_(static_intermesh_remote_mesh_id),
      worker_group_(worker_group), host_policy_mode_(host_policy_mode),
      kServerCore{worker_group, 0}, kGetIngressCore{worker_group, 1},
      kGetAdmissionCore{worker_group, 2},
      test_delay_establish_ack_(test_options.delay_establish_ack),
      test_drop_first_establish_response_(
          test_options.drop_first_establish_response),
      test_first_put_commit_delay_cycles_(test_options.delay_first_put_commit
                                              ? kTestFirstPutCommitDelayCycles
                                              : 0u),
      test_terminate_failures_remaining_(test_options.terminate_failure_count),
      test_fail_launch_after_enqueue_(test_options.fault ==
                                      TestFault::FailLaunchAfterEnqueue),
      test_fail_tiering_worker_(test_options.fault ==
                                TestFault::FailTieringWorker),
      test_initial_counter_(test_options.initial_counter) {
  TT_FATAL(server_submesh_ != nullptr, "server_submesh must be non-null");
  TT_FATAL(IsDefaultMetalContext(*server_submesh_),
           "kvcache_manager ServerRuntimeCore currently supports only the "
           "default MetalContext");
  TT_FATAL(IsFastDispatchEnabled(*server_submesh_),
           "kvcache_manager ServerRuntimeCore requires fast dispatch for its "
           "persistent workload");
  TT_FATAL(server_submesh_->shape() == MeshShape(1, 1),
           "server_submesh must have shape (1,1), got {}",
           server_submesh_->shape());
  TT_FATAL(static_intermesh_t3k_server_ || worker_group_ == 0,
           "non-static kvcache_manager Server supports only worker group 0");
  const uint32_t configured_worker_groups = configured_server_worker_groups();
  TT_FATAL(worker_group_ < configured_worker_groups,
           "KVM worker group {} is outside the {} configured shards",
           worker_group_, configured_worker_groups);
  const auto worker_grid = server_submesh_->compute_with_storage_grid_size();
  TT_FATAL(worker_group_ < worker_grid.x && worker_grid.y >= 3,
           "KVM worker group {} requires logical cores ({},0..2), outside "
           "worker grid ({},{})",
           worker_group_, worker_group_, worker_grid.x, worker_grid.y);
  TT_FATAL(server_submesh_->get_parent_mesh() != nullptr,
           "server_submesh must be a real submesh (parent mesh required)");
  ::kvcache_manager::detail::require_bootstrap_worker_noc(
      server_submesh_->get_device(MeshCoordinate(0, 0)), kServerCore,
      "kvcache_manager Server");
  validate_kvcache_fabric_config("kvcache_manager ServerRuntimeCore");
  if (static_intermesh_t3k_server_) {
    TT_FATAL(is_kvcache_2d_fabric(),
             "a static inter-mesh T3K ServerRuntimeCore requires FABRIC_2D");
    TT_FATAL(
        static_intermesh_remote_mesh_id_ <= 0xFFFFu,
        "static_intermesh_remote_mesh_id {} exceeds the 16-bit route encoding",
        static_intermesh_remote_mesh_id_);
    const auto server_node =
        server_submesh_->get_fabric_node_id(MeshCoordinate(0, 0));
    const auto endpoint_node =
        tt::tt_fabric::get_static_intermesh_endpoint_fabric_node_id();
    TT_FATAL(server_node == endpoint_node,
             "static inter-mesh ServerRuntimeCore must run on the configured "
             "T3K endpoint (server mesh {} chip {}, endpoint "
             "mesh {} chip {})",
             *server_node.mesh_id, server_node.chip_id, *endpoint_node.mesh_id,
             endpoint_node.chip_id);
  }
  session_generation_seed_ = make_server_generation_seed();

  // Preallocate the emergency holder before registry insertion or kernel
  // launch. The destructor's double-failure path must not allocate while
  // preserving addresses a kernel may still reference.
  failed_teardown_holder_ = std::make_unique<FailedTeardownResources>();
  failed_teardown_holder_->server_submesh = server_submesh_->shared_from_this();
  failed_teardown_holder_->registry_key = server_submesh_;
  failed_teardown_holder_->worker_group = worker_group_;

  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    RegistryKey key{server_submesh_, worker_group_};
    TT_FATAL(active_slots.find(key) == active_slots.end(),
             "A ServerRuntimeCore is already active on server_submesh {}",
             static_cast<const void *>(server_submesh_));
    active_slots.insert(key);
  }

  try {
    kvcache_profile_ =
        config::read_bool_env("KVM_SERVER_PROFILE").value_or(false);
    const auto wire_config = config::resolve_wire_config();
    // Catalog, device-cache, and per-value limits are independent
    // configuration axes.
    max_value_bytes_ = wire_config.max_value_bytes;
    max_get_chunks_ = protocol::max_get_chunks_for(max_value_bytes_);
    max_keys_ = parse_max_keys();
    value_heap_bytes_ = parse_value_heap_bytes(max_value_bytes_);
    num_units_ = value_heap_bytes_ / protocol::kMaxChunkValueBytes;
    ring_slots_ = wire_config.server_ring_slots;
    value_ring_depth_ = wire_config.client_value_ring_depth;
    max_key_len_ = wire_config.max_key_bytes;
    dir_entry_bytes_ = protocol::dir_entry_bytes(max_key_len_);
    num_buckets_ = protocol::hash_num_buckets(max_keys_);

    poll_us_ = parse_poll_us();
    scalar_get_pipeline_.mapped_completion_enabled =
        config::read_bool_env("KVM_SERVER_MAPPED_GET_COMPLETION")
            .value_or(true);
    // Device profiling uses the request kernel so KVM-GET remains a complete
    // payload operation zone. Explicitly setting the ingress toggle still
    // permits focused overlap probes.
    const bool device_profiler_enabled =
        tt::tt_metal::IsDeviceProfilerEnabled(*server_submesh_);
    const auto directory_policy = config::resolve_server_directory_policy();
    allocator_profile_enabled_ = directory_policy.allocator_profile_enabled;
    profile_detail_ = config::resolve_profile_detail();
    config::validate_server_profile_config(
        device_profiler_enabled, allocator_profile_enabled_, profile_detail_);
    scalar_get_pipeline_.ingress.enabled =
        config::read_bool_env("KVM_SERVER_MAPPED_GET_INGRESS")
            .value_or(!device_profiler_enabled);
    scalar_get_pipeline_.admission.enabled =
        config::read_bool_env("KVM_SERVER_GET_ADMISSION_WORKER").value_or(true);

    reserve_l1();
    reserve_dram();
    reserve_stage_buffer();
    reserve_get_workers();
    prepare_server_kernel(directory_policy);
    if (host_policy_mode_ == HostPolicyMode::Internal) {
      launch_internal_worker_and_server();
    }
  } catch (...) {
    const auto failure = std::current_exception();
    if (server_launch_may_have_started_) {
      log_error(tt::LogOp,
                "kvcache_manager: ServerRuntimeCore launch outcome is unknown; "
                "quarantining device resources until reset");
      lifecycle_state_.store(LifecycleState::Quarantined);
      quarantine_failed_teardown();
    } else {
      std::lock_guard<std::mutex> lock(registry_mutex);
      active_slots.erase(RegistryKey{server_submesh_, worker_group_});
    }
    std::rethrow_exception(failure);
  }

  if (host_policy_mode_ == HostPolicyMode::Internal) {
    lifecycle_state_.store(LifecycleState::Ready);
  }
}

ServerRuntimeCore::~ServerRuntimeCore() {
  if (host_policy_mode_ == HostPolicyMode::External) {
    const LifecycleState state = lifecycle_state_.load();
    if (state == LifecycleState::Constructing) {
      release_external_runtime_resources();
    } else if (state == LifecycleState::Ready ||
               state == LifecycleState::Closing) {
      quarantine_external_runtime();
    }
    return;
  }

  try {
    close();
  } catch (...) {
    if (lifecycle_state_.load() == LifecycleState::Closed) {
      return;
    }
    try {
      close();
    } catch (...) {
      if (lifecycle_state_.load() == LifecycleState::Closed) {
        return;
      }
      log_error(tt::LogOp, "kvcache_manager: ServerRuntimeCore teardown failed "
                           "twice; quarantining device resources until reset");
      quarantine_failed_teardown();
    }
  }
}

void ServerRuntimeCore::launch_external_runtime() {
  TT_FATAL(host_policy_mode_ == HostPolicyMode::External,
           "launch_external_runtime requires external host policy");
  TT_FATAL(lifecycle_state_.load() == LifecycleState::Constructing,
           "external ServerRuntimeCore runtime can launch only from its "
           "prepared state");
  external_seen_ = test_initial_counter_;
  external_commit_issued_ = test_initial_counter_;
  external_evict_issued_ = test_initial_counter_;
  external_host_io_cancelled_.store(false, std::memory_order_relaxed);
  external_poll_buffer_.assign(
      protocol::kMoveReqSemSize + protocol::kMoveReqSlotBytes, 0);

  try {
    launch_server_kernel();
  } catch (...) {
    if (server_launch_may_have_started_) {
      lifecycle_state_.store(LifecycleState::Quarantined);
      quarantine_failed_teardown();
    }
    throw;
  }
  lifecycle_state_.store(LifecycleState::Ready);
}

void ServerRuntimeCore::terminate_external_runtime() {
  TT_FATAL(host_policy_mode_ == HostPolicyMode::External,
           "terminate_external_runtime requires external host policy");
  const LifecycleState state = lifecycle_state_.load();
  if (state == LifecycleState::Closed || state == LifecycleState::Quarantined) {
    return;
  }
  TT_FATAL(state == LifecycleState::Ready || state == LifecycleState::Closing,
           "external ServerRuntimeCore runtime cannot terminate from its "
           "current lifecycle state");
  lifecycle_state_.store(LifecycleState::Closing);
  terminate_server_kernel();
}

void ServerRuntimeCore::release_external_runtime_resources() {
  TT_FATAL(host_policy_mode_ == HostPolicyMode::External,
           "external resource release requires external host policy");
  const LifecycleState state = lifecycle_state_.load();
  TT_FATAL(state == LifecycleState::Constructing ||
               state == LifecycleState::Closing ||
               state == LifecycleState::Closed,
           "external ServerRuntimeCore runtime resources cannot be released "
           "while its device workload may be running");
  if (state == LifecycleState::Closed) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(host_values_mutex_);
    TT_FATAL(
        external_host_leases_.empty(),
        "external ServerRuntimeCore runtime still owns {} mapped host leases",
        external_host_leases_.size());
    TT_FATAL(external_pool_units_in_use_ == 0,
             "external ServerRuntimeCore runtime still owns {} mapped "
             "host-pool units",
             external_pool_units_in_use_);
  }

  preserve_static_intermesh_kernel_resources();
  workload_.reset();
  static_intermesh_mux_service_.reset();
  host_values_.clear();
  pool_free_.clear();
  pool_host_ptr_ = nullptr;
  pool_units_total_ = 0;
  pool_meta_host_ptr_ = nullptr;
  pool_meta_read_off_ = 0;
  pool_meta_generations_.clear();

  stage_arena_.reset();
  stage_host_ptr_ = nullptr;
  stage_completion_host_ptr_ = nullptr;
  get_completion_host_ptr_ = nullptr;
  get_ingress_host_ptr_ = nullptr;
  stage_noc_ok_ = false;
  scalar_get_pipeline_ = {};

  get_value_cb_buf_.reset();
  residency_lru_buf_.reset();
  server_l1_buf_.reset();
  dram_value_buf_.reset();
  dram_put_host_queue_buf_.reset();
  dram_directory_buf_.reset();
  dram_hashmap_buf_.reset();
  dram_heap_metadata_buf_.reset();
  failed_teardown_holder_.reset();

  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    active_slots.erase(RegistryKey{server_submesh_, worker_group_});
  }
  lifecycle_state_.store(LifecycleState::Closed);
}

void ServerRuntimeCore::quarantine_external_runtime() noexcept {
  if (host_policy_mode_ != HostPolicyMode::External) {
    return;
  }
  external_host_io_cancelled_.store(true, std::memory_order_release);
  server_workload_running_.store(false, std::memory_order_release);
  detach_external_host_leases();
  lifecycle_state_.store(LifecycleState::Quarantined);
  quarantine_failed_teardown();
}

void ServerRuntimeCore::check_ready() const {
  if (lifecycle_state_.load() != LifecycleState::Ready ||
      !server_workload_running_.load(std::memory_order_acquire)) {
    TT_THROW(
        "ServerRuntimeCore is not ready (closing, quarantined, or closed)");
  }
  throw_if_tiering_worker_failed();
}

bool ServerRuntimeCore::has_tiering_worker_failed() const noexcept {
  return tiering_worker_failed_.load(std::memory_order_acquire);
}

void ServerRuntimeCore::record_tiering_worker_failure(
    std::exception_ptr failure) noexcept {
  try {
    std::lock_guard<std::mutex> lock(tiering_worker_failure_mutex_);
    if (tiering_worker_failure_ == nullptr) {
      tiering_worker_failure_ = std::move(failure);
    }
  } catch (...) {
    // Preserve the failure signal even if storing its diagnostic unexpectedly
    // fails.
  }
  tiering_worker_failed_.store(true, std::memory_order_release);
}

std::exception_ptr ServerRuntimeCore::get_tiering_worker_failure() const {
  if (!has_tiering_worker_failed()) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(tiering_worker_failure_mutex_);
  return tiering_worker_failure_;
}

void ServerRuntimeCore::throw_if_tiering_worker_failed() const {
  if (!has_tiering_worker_failed()) {
    return;
  }
  if (const auto failure = get_tiering_worker_failure(); failure != nullptr) {
    std::rethrow_exception(failure);
  }
  TT_THROW("kvcache_manager tiering worker failed");
}

bool ServerRuntimeCore::is_ready() const {
  return lifecycle_state_.load() == LifecycleState::Ready &&
         server_workload_running_.load(std::memory_order_acquire) &&
         !has_tiering_worker_failed();
}

void ServerRuntimeCore::preserve_static_intermesh_kernel_resources() {
  if (static_intermesh_mux_service_ == nullptr ||
      failed_teardown_holder_ == nullptr) {
    return;
  }

  auto resources = std::shared_ptr<FailedTeardownResources>(
      failed_teardown_holder_.release());
  resources->server_l1 = std::move(server_l1_buf_);
  resources->get_value_cb = std::move(get_value_cb_buf_);
  resources->get_ingress_l1 = std::move(scalar_get_pipeline_.ingress.l1);
  resources->get_admission_l1 = std::move(scalar_get_pipeline_.admission.l1);
  resources->residency_lru = std::move(residency_lru_buf_);
  resources->dram_value = std::move(dram_value_buf_);
  resources->dram_put_host_queue = std::move(dram_put_host_queue_buf_);
  resources->dram_directory = std::move(dram_directory_buf_);
  resources->dram_hashmap = std::move(dram_hashmap_buf_);
  resources->dram_heap_metadata = std::move(dram_heap_metadata_buf_);
  resources->stage_arena = std::move(stage_arena_);
  retain_static_intermesh_mux_resources(std::move(resources));
}

ServerDescriptor ServerRuntimeCore::make_descriptor_unchecked() const {
  const auto node = server_submesh_->get_fabric_node_id(
      tt::tt_metal::distributed::MeshCoordinate(0, 0));
  tt::tt_metal::IDevice *device = server_submesh_->get_device(
      tt::tt_metal::distributed::MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr, "server_submesh has no chip at (0,0)");
  // The descriptor remains authoritative after ESTABLISH. The fixed bootstrap
  // contract uses the harvesting-independent translated coordinate of this
  // shard's logical worker core on supported Wormhole systems.
  const auto server_virtual =
      device->worker_core_from_logical_core(kServerCore);
  const auto dram_grid = device->dram_grid_size();
  return ServerDescriptor{
      .fabric_mesh_id = static_cast<uint32_t>(*node.mesh_id),
      .fabric_chip_id = static_cast<uint32_t>(node.chip_id),
      // The actual reserve address (reserve_l1 asserts it equals the allocator
      // base) -- the authoritative value, not a re-derivation.
      .l1_base = server_l1_base_,
      .noc_x = static_cast<uint32_t>(server_virtual.x),
      .noc_y = static_cast<uint32_t>(server_virtual.y),
      .ring_slots = ring_slots_,
      .value_ring_depth = value_ring_depth_,
      .max_value_bytes = max_value_bytes_,
      .max_key_bytes = max_key_len_,
      .max_clients = protocol::kMaxClients,
      .reserved = 0,
      .num_dram_channels = static_cast<uint32_t>(device->num_dram_channels()),
      .dram_grid_x = static_cast<uint32_t>(dram_grid.x),
      .dram_grid_y = static_cast<uint32_t>(dram_grid.y),
      .wire_layout_version = protocol::kWireLayoutVersion,
  };
}

ServerDescriptor ServerRuntimeCore::make_descriptor() const {
  TT_FATAL(is_ready(), "make_descriptor requires a launched, healthy "
                       "ServerRuntimeCore: addresses and wire config are "
                       "resolved at launch, and a descriptor from a failed "
                       "ServerRuntimeCore would describe a wire nobody serves");
  return make_descriptor_unchecked();
}

bool ServerRuntimeCore::test_quarantine_holds_submesh(
    MeshDevice *server_submesh) {
  for (auto *resources = failed_teardown_head_.load(std::memory_order_acquire);
       resources != nullptr; resources = resources->next) {
    if (resources->server_submesh.get() == server_submesh) {
      return true;
    }
  }
  return false;
}

void ServerRuntimeCore::release_test_quarantined_servers() {
  // Tests call this only after injected failures that run after a successful
  // Finish. Production quarantines are never released because their persistent
  // kernel state is unknown.
  auto *resources =
      failed_teardown_head_.exchange(nullptr, std::memory_order_acq_rel);
  while (resources != nullptr) {
    auto *next = resources->next;
    if (resources->safe_to_release_for_test) {
      {
        std::lock_guard<std::mutex> lock(registry_mutex);
        active_slots.erase(
            RegistryKey{resources->registry_key, resources->worker_group});
      }
      delete resources;
    } else {
      auto *head = failed_teardown_head_.load(std::memory_order_relaxed);
      do {
        resources->next = head;
      } while (!failed_teardown_head_.compare_exchange_weak(
          head, resources, std::memory_order_release,
          std::memory_order_relaxed));
    }
    resources = next;
  }
}

void ServerRuntimeCore::quarantine_failed_teardown() noexcept {
  worker_stop_.store(true, std::memory_order_relaxed);
  if (tiering_worker_ != nullptr) {
    if (tiering_worker_->joinable()) {
      tiering_worker_->join();
    }
    tiering_worker_.reset();
  }

  // close() may finish ownership release and then rethrow a saved worker
  // failure. Such an error is diagnostic, not a second teardown failure, and
  // there is no emergency holder left to quarantine.
  if (failed_teardown_holder_ == nullptr) {
    return;
  }

  // Deliberately process-lifetime storage: active_slots still rejects reuse,
  // and these allocations keep the submesh and every address visible to the
  // possibly live kernel valid until device reset. Construction preallocates
  // this holder before launch, so this destructor path never allocates.
  auto *resources = failed_teardown_holder_.release();
  resources->workload = std::move(workload_);
  resources->static_intermesh_mux_service =
      std::move(static_intermesh_mux_service_);
  resources->server_l1 = std::move(server_l1_buf_);
  resources->get_value_cb = std::move(get_value_cb_buf_);
  resources->get_ingress_l1 = std::move(scalar_get_pipeline_.ingress.l1);
  resources->get_admission_l1 = std::move(scalar_get_pipeline_.admission.l1);
  resources->residency_lru = std::move(residency_lru_buf_);
  resources->dram_value = std::move(dram_value_buf_);
  resources->dram_put_host_queue = std::move(dram_put_host_queue_buf_);
  resources->dram_directory = std::move(dram_directory_buf_);
  resources->dram_hashmap = std::move(dram_hashmap_buf_);
  resources->dram_heap_metadata = std::move(dram_heap_metadata_buf_);
  resources->stage_arena = std::move(stage_arena_);
  resources->safe_to_release_for_test = test_quarantine_safe_to_release_;

  auto *head = failed_teardown_head_.load(std::memory_order_relaxed);
  do {
    resources->next = head;
  } while (!failed_teardown_head_.compare_exchange_weak(
      head, resources, std::memory_order_release, std::memory_order_relaxed));
}

void ServerRuntimeCore::close() {
  auto operation_lock = acquire_host_operation();
  bool drain_before_teardown = false;
  bool worker_failed = false;
  std::exception_ptr worker_failure;
  const LifecycleState state = lifecycle_state_.load();
  if (state == LifecycleState::Quarantined || state == LifecycleState::Closed) {
    return;
  }
  TT_FATAL(state == LifecycleState::Ready || state == LifecycleState::Closing,
           "ServerRuntimeCore cannot close from its current lifecycle state");

  if (state == LifecycleState::Ready) {
    lifecycle_state_.store(LifecycleState::Closing);
    drain_before_teardown = true;
  }

  if (drain_before_teardown && !has_tiering_worker_failed()) {
    // Preserve-before-reuse PUTs may need the live host worker to complete
    // eviction handshakes. Deployment must quiesce endpoint Clients first;
    // failure restores a usable ServerRuntimeCore.
    try {
      drain_inflight_puts();
    } catch (...) {
      if (!has_tiering_worker_failed()) {
        if (lifecycle_state_.load() == LifecycleState::Closing) {
          lifecycle_state_.store(LifecycleState::Ready);
        }
        throw;
      }
    }
  }

  // Finish may throw after the termination signal landed. Closing rejects data
  // APIs but preserves the workload, worker, buffers, and registry so a later
  // close() can retry from this point.
  terminate_server_kernel();

  worker_stop_.store(true, std::memory_order_relaxed);
  if (tiering_worker_ != nullptr) {
    if (tiering_worker_->joinable()) {
      tiering_worker_->join();
    }
    tiering_worker_.reset();
  }
  worker_failed = has_tiering_worker_failed();
  worker_failure = get_tiering_worker_failure();

  // Check pool accounting after the worker is joined, then release values
  // before their mapping.
  {
    std::lock_guard<std::mutex> lk(host_values_mutex_);
    if (pool_host_ptr_ != nullptr) {
      uint32_t accounted = 0;
      for (const auto &[free_unit, free_units] : pool_free_) {
        (void)free_unit;
        accounted += free_units;
      }
      for (const auto &[slot, hv] : host_values_) {
        (void)slot;
        accounted += hv.pool_units;
      }
      if (accounted != pool_units_total_) {
        log_warning(tt::LogOp,
                    "kvcache_manager: host-value pool leaked — {} of {} units "
                    "accounted for",
                    accounted, pool_units_total_);
      }
    }
    host_values_.clear();
    pool_free_.clear();
    pool_host_ptr_ = nullptr;
    pool_units_total_ = 0;
    pool_meta_host_ptr_ = nullptr;
    pool_meta_read_off_ = 0;
    pool_meta_generations_.clear();
  }

  // Drop this shard's ownership after its worker joins. The endpoint arena
  // remains mapped until the last shard or quarantine holder releases it.
  preserve_static_intermesh_kernel_resources();
  stage_arena_.reset();
  stage_host_ptr_ = nullptr;
  stage_completion_host_ptr_ = nullptr;
  get_completion_host_ptr_ = nullptr;
  get_ingress_host_ptr_ = nullptr;
  stage_noc_ok_ = false;
  scalar_get_pipeline_ = {};

  // Release every device allocation while the Python wrapper may remain alive.
  // The kernel has exited and the host worker is joined, so no device or host
  // path can retain these addresses.
  get_value_cb_buf_.reset();
  residency_lru_buf_.reset();
  server_l1_buf_.reset();
  dram_value_buf_.reset();
  dram_put_host_queue_buf_.reset();
  dram_directory_buf_.reset();
  dram_hashmap_buf_.reset();
  dram_heap_metadata_buf_.reset();
  static_intermesh_mux_service_.reset();
  failed_teardown_holder_.reset();

  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    active_slots.erase(RegistryKey{server_submesh_, worker_group_});
  }
  lifecycle_state_.store(LifecycleState::Closed);

  if (worker_failure != nullptr) {
    std::rethrow_exception(worker_failure);
  }
  if (worker_failed) {
    TT_THROW("kvcache_manager tiering worker failed");
  }
}

} // namespace kvcache_manager::detail::server_runtime
