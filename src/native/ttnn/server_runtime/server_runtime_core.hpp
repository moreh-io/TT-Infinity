// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "server_runtime/ttnn_dependencies.hpp"

namespace tt::tt_metal {
class IDevice;
class HostBuffer;
namespace experimental {
class PinnedMemory;
} // namespace experimental
} // namespace tt::tt_metal

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshWorkload;
class MeshBuffer;
} // namespace tt::tt_metal::distributed

namespace kvcache_manager::detail::server_runtime {

class ServerDeviceRuntime;
class ServerHostLease;
struct SharedHostArena;
struct ServerHostRequest;
struct ServerReceivedValue;
struct ServerSupplyMetrics;

struct MoveProfile {
  uint64_t to_host_count = 0;
  uint64_t to_host_worker_ns = 0;
  uint64_t to_host_complete_ns = 0;
  uint64_t to_host_read_calls = 0;
  uint64_t to_host_read_bytes = 0;
  uint64_t to_host_read_passes = 0;
  uint64_t to_host_read_retries = 0;
  uint64_t to_host_read_io_ns = 0;
  uint64_t to_host_scatter_ns = 0;
  uint64_t to_host_device_pushes = 0;
  uint64_t to_device_count = 0;
  uint64_t to_device_worker_ns = 0;
  uint64_t to_device_complete_ns = 0;
};

struct AllocatorProfile {
  uint64_t allocation_calls = 0;
  uint64_t allocation_successes = 0;
  uint64_t allocation_failures = 0;
  uint64_t victim_scans = 0;
  uint64_t victims_evicted = 0;
  uint64_t lease_calls = 0;
  uint64_t lease_successes = 0;
  uint64_t lease_failures = 0;
  uint64_t lease_victim_scans = 0;
  uint64_t lease_victims_evicted = 0;
  uint64_t scan_cycles = 0;
  uint64_t lease_scan_cycles = 0;
  uint64_t allocation_cycles = 0;
  uint32_t clock_mhz = 0;
};

struct CacheState {
  bool found = false;
  bool device_resident = false;
  bool host_backed = false;
};

/// KV cache server (the NIC chip) attached to a (1,1) submesh of a caller-owned
/// parent mesh. Runs a persistent op-dispatch kernel. The device directory owns
/// catalog metadata and residency; after commit, the host backing store owns
/// the authoritative value bytes. The caller owns the parent mesh;
/// ServerRuntimeCore keeps an emergency strong reference to server_submesh and
/// reserves L1 from its allocator. See
/// ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §1, §2, §8 for role,
/// topology, enumeration, and lifecycle contracts.
class ServerRuntimeCore {
public:
  /// Preconditions (TT_FATAL): server_submesh is non-null, shape (1,1), a real
  /// submesh (has a parent mesh), and not already backing a live
  /// ServerRuntimeCore. The caller-side L1 invariants (first-alloc base,
  /// exclusive submesh allocator, quiesce before close) are not
  /// machine-enforced; see
  /// ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §2, §8.
  /// `static_intermesh_t3k_server=true` requires FABRIC_2D, places the
  /// ServerRuntimeCore on the configured T3K static endpoint, opens one static
  /// connection, and maps Galaxy fabric chip IDs 0..31 to preseeded routes in
  /// `static_intermesh_remote_mesh_id`.
  explicit ServerRuntimeCore(
      tt::tt_metal::distributed::MeshDevice *server_submesh,
      bool static_intermesh_t3k_server = false,
      uint32_t static_intermesh_remote_mesh_id = 0, uint32_t worker_group = 0);

  ~ServerRuntimeCore();

  ServerRuntimeCore(const ServerRuntimeCore &) = delete;
  ServerRuntimeCore &operator=(const ServerRuntimeCore &) = delete;
  ServerRuntimeCore(ServerRuntimeCore &&) = delete;
  ServerRuntimeCore &operator=(ServerRuntimeCore &&) = delete;

  // Host-direct debug/test API — NOT a production transport; the wire
  // Client::get/exists/remove methods are the end-user path. These methods
  // read, or in the remove case invalidate, the authoritative device DRAM
  // directory over PCIe. The caller must quiesce all endpoint Clients because
  // this process-local mutex cannot serialize remote traffic. See
  // ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §6.
  std::optional<std::vector<uint8_t>> get(const std::string &key);
  bool exists(const std::string &key);
  bool remove(const std::string &key);
  CacheState cache_state(const std::string &key);

  /// Wait until every PUT issued before this call has completed host
  /// write-through and the directory/hash commit. The Client-facing durable
  /// boundary is Client::flush(), which uses Fabric. The caller must quiesce
  /// endpoint Clients before using this host-direct diagnostic.
  void flush();
  /// Debug/benchmark completion sequence for host-tiering worker transactions,
  /// including write-through, host-served GET, and explicit admission. No-op
  /// MOVE requests do not advance it.
  uint64_t moves_completed() const;

  /// Cumulative server-local tiering timings. The to-device fields include
  /// explicit admission and host-served GET. Values remain zero unless
  /// KVM_SERVER_PROFILE=1.
  MoveProfile move_profile() const;

  /// Cumulative BRISC allocator-pressure counters. Values remain zero unless
  /// KVM_SERVER_ALLOCATOR_PROFILE=1. Read only while request submission is
  /// quiescent.
  AllocatorProfile allocator_profile() const;

  /// Debug/test only: how many U-units of the host-value pool are free, and how
  /// many host-tier values currently live in it. Whether a value sits in the
  /// pool or in a heap vector is invisible from the outside otherwise -- both
  /// serve byte-exact -- so a test that means to exercise one of the two paths
  /// has no other way to confirm it did. Both are 0 when there is no pool. See
  /// ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §5.
  std::pair<uint32_t, uint32_t> pool_stats();
  /// Debug/test-only raw device issued and committed PUT counters.
  std::pair<uint32_t, uint32_t> put_drain_progress();
  bool is_ready() const;

  /// ServerRuntimeCore-owned wire configuration. Kept as a host diagnostic;
  /// Client construction receives the same fields only through Fabric
  /// ESTABLISH.
  ServerDescriptor make_descriptor() const;

  void close();

  void drain_inflight_puts(); // block until the current device-issued PUT
                              // frontier commits

private:
  friend struct ServerTestAccess;
  friend class ServerDeviceRuntime;
  friend class ServerHostLease;

  struct FailedTeardownResources;
  struct StaticIntermeshMuxService;
  enum class TestFault : uint8_t {
    None,
    FailLaunchAfterEnqueue,
    FailTieringWorker
  };
  struct TestOptions {
    TestFault fault = TestFault::None;
    bool delay_establish_ack = false;
    bool drop_first_establish_response = false;
    bool delay_first_put_commit = false;
    uint32_t initial_counter = 0;
    uint32_t terminate_failure_count = 0;
  };
  struct GetAdmissionProgress {
    bool enabled = false;
    uint32_t ready = 0;
    uint32_t source_done = 0;
    uint32_t durable_done = 0;
  };
  enum class LifecycleState : uint8_t {
    Constructing,
    Ready,
    Closing,
    Quarantined,
    Closed
  };
  enum class HostPolicyMode : uint8_t { Internal, External };

  struct ScalarGetMode {
    bool mapped_completion = false;
    bool mapped_ingress = false;
  };

  struct FabricFreeWorker {
    bool enabled = false;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> l1;
    uint32_t l1_base = 0;
    uint32_t termination_addr = 0;
  };

  struct ScalarGetPipeline {
    bool mapped_completion_enabled = false;
    FabricFreeWorker ingress;
    FabricFreeWorker admission;
  };

  static std::atomic<FailedTeardownResources *> failed_teardown_head_;

  /// Test-only snapshot of the dedicated GET-admission worker's cumulative
  /// handoff counters. Read only while scalar request submission is quiescent.
  GetAdmissionProgress test_get_admission_progress() const;
  /// Test-only final mapped-ingress selection after resource reservation.
  bool test_mapped_get_ingress_enabled() const noexcept;
  // Inspect and safely release only injected-failure quarantines. Production
  // failures remain quarantined until process exit and device reset.
  static bool test_quarantine_holds_submesh(
      tt::tt_metal::distributed::MeshDevice *server_submesh);
  static void release_test_quarantined_servers();

  ServerRuntimeCore(tt::tt_metal::distributed::MeshDevice *server_submesh,
                    TestOptions test_options);
  ServerRuntimeCore(tt::tt_metal::distributed::MeshDevice *server_submesh,
                    TestOptions test_options, bool static_intermesh_t3k_server,
                    uint32_t static_intermesh_remote_mesh_id,
                    uint32_t worker_group = 0,
                    HostPolicyMode host_policy_mode = HostPolicyMode::Internal);

  tt::tt_metal::distributed::MeshDevice *server_submesh_ = nullptr;
  // Explicit role selection; independent host-local control planes may use
  // identical mesh IDs.
  bool static_intermesh_t3k_server_ = false;
  uint32_t static_intermesh_remote_mesh_id_ = 0;
  uint32_t worker_group_ = 0;
  HostPolicyMode host_policy_mode_ = HostPolicyMode::Internal;
  std::atomic<LifecycleState> lifecycle_state_{LifecycleState::Constructing};
  std::atomic<bool> server_workload_running_{false};
  bool server_launch_may_have_started_ = false;
  bool server_termination_signalled_ = false;

  std::unique_ptr<FailedTeardownResources> failed_teardown_holder_;

  // L1 carve-out reserve (held for the ServerRuntimeCore's lifetime);
  // server_l1_base_ is the base for every layout.hpp offset.
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> server_l1_buf_;
  uint32_t server_l1_base_ = 0;

  // GET value handoff CB backing buffer (BRISC reader -> NCRISC sender), placed
  // top-down and aliased via set_globally_allocated_address so it does not
  // clash with the manual carve-out. Held for the ServerRuntimeCore's lifetime
  // (the CB references its address).
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> get_value_cb_buf_;

  // BRISC-private exact LRU links, one prev/next pair per directory slot. This
  // is a separate top-down L1 allocation so runtime max_keys does not alter the
  // shared carve-out offsets.
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> residency_lru_buf_;
  uint32_t residency_lru_base_ = 0;

  // Kernel resources owned by ServerRuntimeCore.
  std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> workload_;
  std::shared_ptr<StaticIntermeshMuxService> static_intermesh_mux_service_;
  uint32_t termination_addr_ = 0;
  // ServerRuntimeCore kernel core. Submesh boundaries isolate the chip, so no
  // SubDeviceManager.
  const tt::tt_metal::CoreCoord kServerCore;
  const tt::tt_metal::CoreCoord kGetIngressCore;
  const tt::tt_metal::CoreCoord kGetAdmissionCore;

  void check_ready() const;
  bool has_tiering_worker_failed() const noexcept;
  void record_tiering_worker_failure(std::exception_ptr failure) noexcept;
  std::exception_ptr get_tiering_worker_failure() const;
  void throw_if_tiering_worker_failed() const;
  ScalarGetMode scalar_get_mode() const noexcept;
  void prepare_mapped_get_completion();
  uint32_t wait_for_mapped_get_completion();
  void submit_mapped_get_ingress(uint32_t ring_slot, uint32_t client_token,
                                 uint32_t seq, uint32_t key_len,
                                 const uint32_t *key_words,
                                 uint32_t client_dram_base,
                                 uint32_t logical_page_size,
                                 uint32_t output_capacity_bytes,
                                 uint32_t completion_l1_addr);
  void reserve_l1();
  void reserve_get_workers();
  void reserve_auxiliary_worker_l1(FabricFreeWorker &worker,
                                   tt::tt_metal::CoreCoord core, uint32_t bytes,
                                   const char *role_name);
  void reserve_dram();
  void reserve_stage_buffer();
  void
  prepare_server_kernel(const config::ServerDirectoryPolicy &directory_policy);
  void launch_server_kernel();
  void launch_internal_worker_and_server();
  void launch_external_runtime();
  void terminate_external_runtime();
  void release_external_runtime_resources();
  void quarantine_external_runtime() noexcept;
  void preserve_static_intermesh_kernel_resources();
  void retain_static_intermesh_mux_resources(std::shared_ptr<void> resources);
  void detach_external_host_leases() noexcept;
  ServerDescriptor make_descriptor_unchecked() const;
  void terminate_server_kernel();
  std::unique_lock<std::mutex> acquire_host_operation();
  void quarantine_failed_teardown() noexcept;

  std::optional<ServerHostRequest> poll_external_host_request();
  ServerReceivedValue receive_external_value(const ServerHostRequest &request,
                                             bool prefer_mapped_lease);
  ServerSupplyMetrics
  supply_external_value(const ServerHostRequest &request,
                        const ServerHostLease *lease,
                        std::span<const uint8_t> heap_bytes);
  void complete_external_host_request(const ServerHostRequest &request);
  void cancel_external_host_io() noexcept;
  void release_external_host_lease(ServerHostLease &lease) noexcept;
  void validate_external_request(const ServerHostRequest &request) const;

  // ServerRuntimeCore DRAM buffers: a buddy-managed value heap and
  // max_keys_-entry pending-write FIFO. See
  // ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §5.
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dram_value_buf_;
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>
      dram_put_host_queue_buf_;
  uint32_t dram_value_base_ = 0;
  uint32_t dram_put_host_queue_base_ = 0;

  // Device directory owns catalog metadata; value_byte_off independently names
  // a cache block. See ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md
  // §4 and §5.
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dram_directory_buf_;
  uint32_t dram_directory_base_ = 0;
  // Auxiliary key-hash index; the directory remains authoritative.
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dram_hashmap_buf_;
  uint32_t dram_hashmap_base_ = 0;
  uint32_t num_buckets_ = 0;
  // Buddy allocator metadata: one block-state/free-list entry per device-cache
  // heap unit.
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>
      dram_heap_metadata_buf_;
  uint32_t dram_heap_metadata_base_ = 0;
  uint32_t max_key_len_ = 0;
  uint32_t dir_entry_bytes_ = 0;

  // Directory staging, mailbox, semaphore, and epoch offsets within the
  // ServerRuntimeCore L1 reserve.
  uint32_t dir_scan_staging_offset_ = 0;
  uint32_t dir_req_mailbox_offset_ = 0;
  uint32_t dir_req_sem_offset_ = 0;
  uint32_t dir_done_sem_offset_ = 0;
  uint32_t dir_resp_mailbox_offset_ = 0;
  uint32_t dir_epoch_counter_offset_ = 0;

  // BRISC staging for publishing a pending entry and later patching its host
  // generation. Distinct from LRU scan staging.
  uint32_t dir_commit_entry_staging_offset_ = 0;

  // The ServerRuntimeCore resolves runtime ring depths once and the Client
  // inherits them. All derived offsets must use the runtime layout helpers; see
  // ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §3 and §4.
  uint32_t ring_slots_ = 0;
  uint32_t value_ring_depth_ = 0;

  // Runtime carve-out size (reserve_l1, page-rounded); bounded by
  // kServerL1ReserveMaxBytes and the physical worker-L1 budget.
  uint32_t server_l1_reserve_bytes_ = 0;

  // Fixed-prefix offsets re-derived from the runtime ring depth at launch.
  uint32_t ring_sem_base_offset_ = 0;
  uint32_t peer_lookup_table_offset_ = 0;
  uint32_t host_any_get_done_monitor_offset_ = 0;
  uint32_t get_credit_sem_offset_ = 0;

  // Independent catalog/FIFO capacity and device-cache capacity axes.
  // See ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §5.
  uint32_t max_keys_ = 0;
  uint32_t value_heap_bytes_ = 0;
  uint32_t num_units_ = 0;

  // Single-value cap (KVM_COMMON_MAX_VALUE_BYTES): bounds PUT value_len. The
  // heap minimum is ceil(cap / U) * U; max_get_chunks_ = ceil(cap / U) bounds
  // the GET chunk count.
  uint32_t max_value_bytes_ = 0;
  uint32_t max_get_chunks_ = 0;

  // PUT landed/stream-progress, readback, and durable-commit publication
  // offsets.
  uint32_t direct_put_done_offset_ = 0;
  uint32_t put_readback_scratch_offset_ = 0;
  uint32_t commit_issued_offset_ = 0;
  uint32_t commit_done_offset_ = 0;

  // BRISC-owned buddy allocator state (max order plus per-order free-list
  // heads).
  uint32_t heap_allocator_offset_ = 0;

  // Host-tiering worker request/completion channels at the end of the
  // ServerRuntimeCore L1 reserve. See
  // ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §6.
  uint32_t move_req_sem_offset_ = 0;
  uint32_t move_req_offset_ = 0;
  uint32_t commit_req_sem_offset_ = 0;
  uint32_t commit_req_offset_ = 0;
  // The single move request slot is reusable only when move_req_sem ==
  // move_processed.
  uint32_t move_processed_offset_ = 0;
  // Per-host-serve count of chunks ready in the staging window.
  uint32_t stage_progress_offset_ = 0;
  // BRISC source-consumed count; fallback workers wait here instead of
  // mapped-host completion.
  uint32_t promote_done_offset_ = 0;
  // Mapped write-through device-push destination and completion handshake.
  uint32_t evict_ack_sem_offset_ = 0;
  uint32_t evict_ack_offset_ = 0;
  uint32_t evict_done_offset_ = 0;
  uint32_t hash_probe_staging_offset_ = 0;
  uint32_t heap_metadata_staging_offset_ = 0;
  // BRISC publishes cold-start completion before NCRISC begins serving.
  uint32_t cold_start_done_offset_ = 0;
  uint32_t allocator_profile_offset_ = 0;

  // One authoritative host backing value. Pool values are directly
  // NoC-readable; heap-vector values are copied through the staging window for
  // host serve or cache admission.
  struct HostValue {
    std::vector<uint8_t> heap; // empty iff pool-resident
    uint32_t len = 0;
    uint32_t pool_unit =
        0; // U-unit offset into the pool; meaningful iff pool_units > 0
    uint32_t pool_units = 0; // 0 = heap-resident
    uint32_t generation =
        0; // pool metadata generation; 0 for heap-resident values
  };

  // Authoritative host backing store, keyed by directory slot.
  // See ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §5.
  // host_values_mutex_ also guards the pool free list.
  std::unordered_map<uint32_t, HostValue> host_values_;
  std::mutex host_values_mutex_;

  // Bytes of a host-tier value, wherever it lives. Valid while
  // host_values_mutex_ is held.
  const uint8_t *host_value_bytes(const HostValue &hv) const;

  // Host-value pool: the NoC-mapped region past the staging window (see
  // reserve_stage_buffer). Pool residency lets host-source operations hand the
  // device an offset instead of copying; capacity is therefore an optimization
  // and never a correctness condition. U-unit extents, first-fit with neighbour
  // coalescing; pool_free_ maps unit offset -> unit count and its extents are
  // disjoint and never adjacent.
  uint8_t *pool_host_ptr_ = nullptr; // pool base, host side (nullptr = no pool)
  uint32_t pool_read_off_ =
      0; // pool base as a byte offset from the staging window base
  uint32_t pool_units_total_ = 0; // for the close-time accounting check
  uint8_t *pool_meta_host_ptr_ =
      nullptr; // slot metadata base in the same NoC-mapped region
  uint32_t pool_meta_read_off_ =
      0; // metadata base as a staging-window byte offset
  std::vector<uint32_t> pool_meta_generations_;
  std::map<uint32_t, uint32_t> pool_free_;
  uint32_t external_pool_units_in_use_ = 0;
  std::unordered_set<ServerHostLease *> external_host_leases_;
  // Called with host_values_mutex_ held.
  std::optional<uint32_t> pool_alloc(uint32_t units);
  void pool_release(uint32_t unit, uint32_t units);
  void drop_host_value(uint32_t slot);
  void invalidate_pool_meta(uint32_t slot);
  void invalidate_pool_meta_if_generation(uint32_t slot, uint32_t generation);
  uint32_t publish_pool_meta(uint32_t slot, uint32_t read_off,
                             uint32_t total_len);

  // Endpoint-shared NoC-mapped host transfer arena. Each static worker group
  // owns a disjoint slice containing its staging window, completion lines,
  // pool metadata, and pool. A usable mapping enables NoC push/pull; otherwise
  // the worker uses targeted host/device copies. See
  // ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md §6.
  std::shared_ptr<SharedHostArena> stage_arena_;
  uint8_t *stage_host_ptr_ =
      nullptr; // worker-group slice base (worker memcpy destination)
  uint32_t stage_bytes_ =
      0; // staging capacity (>= max single value, U-rounded)
  volatile uint32_t *stage_completion_host_ptr_ =
      nullptr; // mapped-host durable host-source sequence
  uint32_t stage_completion_offset_ = 0; // byte offset from stage_host_ptr_
  volatile uint32_t *get_completion_host_ptr_ =
      nullptr;                         // mapped-host scalar GET status+1
  uint32_t get_completion_offset_ = 0; // byte offset from stage_host_ptr_
  uint8_t *get_ingress_host_ptr_ =
      nullptr;                      // double-buffered host ingress descriptors
  uint32_t get_ingress_offset_ = 0; // byte offset from stage_host_ptr_
  uint32_t get_ingress_generation_ =
      0; // last host-published descriptor generation
  uint32_t stage_pcie_xy_enc_ =
      0; // server RT arg: PCIe core NoC encoding of the host staging buffer
  uint32_t stage_addr_lo_ = 0; // server RT arg: host NoC addr low 32
  uint32_t stage_addr_hi_ = 0; // server RT arg: host NoC addr high 32
  bool stage_noc_ok_ = false;  // usable server-side NoC mapping
  ScalarGetPipeline scalar_get_pipeline_;

  // Host-tiering worker thread (started at launch end, joined at close): polls
  // move_req, manages host-value lifetime, and coordinates mapped device
  // push/pull or targeted-copy fallbacks.
  std::atomic<uint64_t> moves_completed_{0};
  std::unique_ptr<std::thread> tiering_worker_;
  std::atomic<bool> worker_stop_{false};
  std::atomic<bool> external_host_io_cancelled_{false};
  std::atomic<bool> tiering_worker_failed_{false};
  mutable std::mutex tiering_worker_failure_mutex_;
  std::exception_ptr tiering_worker_failure_;
  void tiering_worker_loop();
  // Device-to-host fallback and mapped device-push destination helpers.
  void stage_value_device_to_host(uint32_t slot, uint32_t value_byte_off,
                                  uint32_t total_len);
  struct DevicePushDestination {
    std::optional<uint32_t> pool_unit;
    uint32_t read_off = 0;
    uint32_t generation = 0;
  };
  DevicePushDestination reserve_device_push_destination(uint32_t slot,
                                                        uint32_t total_len);
  void publish_device_pushed_value(uint32_t slot, uint32_t total_len,
                                   const DevicePushDestination &destination);
  void stage_value_host_to_device(uint32_t slot, uint32_t value_byte_off,
                                  uint32_t total_len, uint32_t stage_read_off);
  // Feed a host-served GET from an in-place mapped value or the streaming
  // staging window. See ttnn/cpp/ttnn/operations/kvcache_manager/docs/design.md
  // §6.
  void stage_host_value_for_stream(tt::tt_metal::IDevice *device, uint32_t slot,
                                   uint32_t total_len,
                                   uint32_t stage_progress_addr,
                                   uint32_t stage_read_off);
  void publish_stage_progress(tt::tt_metal::IDevice *device,
                              uint32_t stage_progress_addr,
                              uint32_t staged_chunks);
  // The commit_req stage read offset for a host-source operation on this slot
  // (0 = must be copied into the window).
  uint32_t promote_read_off(uint32_t slot);
  // Targeted per-unit PCIe read of just this value's heap units (not a
  // whole-heap read), with a double-read stability guard. Shared by host-direct
  // ServerRuntimeCore::get and stage_value_device_to_host.
  std::vector<uint8_t> read_value_units(uint32_t value_byte_off,
                                        uint32_t total_len);
  struct ValueReadProfile {
    uint64_t calls = 0;
    uint64_t bytes = 0;
    uint64_t passes = 0;
    uint64_t retries = 0;
    uint64_t io_ns = 0;
    uint64_t scatter_ns = 0;
  };
  // Same read, but into caller-provided storage so write-through can land
  // straight in a pool extent with no intermediate vector. `dst` must hold
  // whole U units (the PCIe read is per unit, so the last unit needs a full
  // unit of room even when total_len does not fill it).
  void read_value_units_into(std::span<uint8_t> dst, uint32_t value_byte_off,
                             uint32_t total_len,
                             ValueReadProfile *profile = nullptr);

  // KVM_SERVER_PROFILE enables host-clock PUT drain and sync-commit timing.
  bool kvcache_profile_ = false;
  // Construction-time snapshots guard compatibility and configure both
  // ServerRuntimeCore kernels.
  bool allocator_profile_enabled_ = false;
  config::ProfileDetail profile_detail_{};
  std::atomic<uint64_t> drain_wait_ns_{0};
  std::atomic<uint64_t> drain_wait_count_{0};
  std::atomic<uint64_t> sync_commit_wait_ns_{0};
  std::atomic<uint64_t> sync_commit_wait_count_{0};

  // Host-source operation per-phase breakdown, including explicit admission and
  // host-served GET.
  std::atomic<uint64_t> promote_count_{0};
  std::atomic<uint64_t> promote_reqread_ns_{0}; // sem-observe + move_req slot
  std::atomic<uint64_t> promote_gather_ns_{
      0}; // strided host memcpy into the per-bank scratch
  std::atomic<uint64_t> promote_dma_ns_{
      0}; // WriteToDeviceDRAMChannel (host->device DMA)
  // Split host streaming copy time from progress-publication time.
  std::atomic<uint64_t> promote_stage_copy_ns_{0};
  std::atomic<uint64_t> promote_stage_pub_ns_{0};
  std::atomic<uint64_t> promote_commit_ns_{
      0}; // commit_req slot + semaphore publication
  std::atomic<uint64_t> promote_completion_wait_ns_{0};
  std::atomic<uint64_t> promote_completion_l1_fallback_count_{0};

  // ServerRuntimeCore-local tiering scopes. worker_ns ends at publication;
  // complete_ns ends at durability.
  std::atomic<uint64_t> move_to_host_count_{0};
  std::atomic<uint64_t> move_to_host_worker_ns_{0};
  std::atomic<uint64_t> move_to_host_complete_ns_{0};
  std::atomic<uint64_t> move_to_host_read_calls_{0};
  std::atomic<uint64_t> move_to_host_read_bytes_{0};
  std::atomic<uint64_t> move_to_host_read_passes_{0};
  std::atomic<uint64_t> move_to_host_read_retries_{0};
  std::atomic<uint64_t> move_to_host_read_io_ns_{0};
  std::atomic<uint64_t> move_to_host_scatter_ns_{0};
  std::atomic<uint64_t> move_to_host_device_pushes_{0};
  std::atomic<uint64_t> write_through_reqread_ns_{0};
  std::atomic<uint64_t> write_through_reserve_ns_{0};
  std::atomic<uint64_t> write_through_ack_ns_{0};
  std::atomic<uint64_t> write_through_push_wait_ns_{0};
  std::atomic<uint64_t> write_through_publish_ns_{0};
  std::atomic<uint64_t> write_through_processed_ns_{0};
  std::atomic<uint64_t> move_to_device_count_{0};
  std::atomic<uint64_t> move_to_device_worker_ns_{0};
  std::atomic<uint64_t> move_to_device_complete_ns_{0};

  // Private regression-test hooks; production construction keeps them inactive.
  bool test_delay_establish_ack_ = false;
  bool test_drop_first_establish_response_ = false;
  // The BRISC defers only the first ready PUT commit while continuing to
  // service lookups.
  uint32_t test_first_put_commit_delay_cycles_ = 0;
  // Private lifecycle-test hook.
  uint32_t test_terminate_failures_remaining_ = 0;
  // Private fault-injection / near-wrap hooks. Production defaults keep all
  // inactive.
  bool test_fail_launch_after_enqueue_ = false;
  bool test_fail_tiering_worker_ = false;
  uint32_t test_initial_counter_ = 0;
  uint32_t session_generation_seed_ = 1;
  bool test_quarantine_safe_to_release_ = false;

  // Idle host-worker and targeted-copy fallback poll interval
  // (KVM_SERVER_HOST_POLL_US, default 10).
  uint32_t poll_us_ = 10;

  struct ExternalHostRequestState {
    enum class Phase : uint8_t {
      Pending,
      TransferInProgress,
      TransferComplete
    };

    bool active = false;
    Phase phase = Phase::Pending;
    uint32_t sequence = 0;
    uint32_t op = 0;
    uint32_t slot = 0;
    uint32_t value_byte_off = 0;
    uint32_t total_len = 0;
    uint32_t key_len = 0;
    std::vector<uint8_t> key;
  };
  ExternalHostRequestState external_host_request_;
  uint32_t external_seen_ = 0;
  uint32_t external_commit_issued_ = 0;
  uint32_t external_evict_issued_ = 0;
  std::vector<uint8_t> external_poll_buffer_;

  // Serializes process-local host diagnostics and teardown. Endpoint Clients
  // are coordinated by the device protocol and must be externally quiesced
  // before host-direct diagnostics or close.
  std::mutex host_operation_mutex_;
};

} // namespace kvcache_manager::detail::server_runtime
