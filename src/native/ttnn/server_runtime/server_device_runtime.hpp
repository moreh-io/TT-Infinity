// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "server_runtime/ttnn_dependencies.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace kvcache_manager::detail::server_runtime {

class ServerDeviceRuntime;
class ServerHostLease;

struct ServerDeviceRuntimeDeleter {
  void operator()(ServerDeviceRuntime *runtime) const noexcept;
};

struct ServerHostLeaseDeleter {
  void operator()(ServerHostLease *lease) const noexcept;
};

using ServerDeviceRuntimeHandle =
    std::unique_ptr<ServerDeviceRuntime, ServerDeviceRuntimeDeleter>;
using ServerHostLeaseHandle =
    std::unique_ptr<ServerHostLease, ServerHostLeaseDeleter>;

enum class ServerDeviceRuntimeState : uint8_t {
  Prepared,
  Running,
  Closing,
  Terminated,
  Quarantined
};

enum class ServerHostRequestKind : uint8_t {
  WriteThrough,
  RemoveBacking,
  PromoteToDevice,
  ServeGet
};

struct ServerDeviceRuntimeConfig {
  tt::tt_metal::distributed::MeshDevice *server_submesh = nullptr;
  bool static_intermesh_t3k_server = false;
  uint32_t static_intermesh_remote_mesh_id = 0;
  uint32_t worker_group = 0;
  bool test_delay_first_establish_response = false;
  bool test_drop_first_establish_response = false;
  bool test_delay_first_put_commit = false;
};

struct ServerHostRequest {
  uint32_t sequence = 0;
  ServerHostRequestKind kind = ServerHostRequestKind::WriteThrough;
  uint32_t slot = 0;
  uint32_t value_len = 0;
  uint32_t device_value_offset = 0;
  uint64_t poll_ns = 0;
};

struct ServerDeviceReadMetrics {
  uint64_t calls = 0;
  uint64_t bytes = 0;
  uint64_t passes = 0;
  uint64_t retries = 0;
  uint64_t io_ns = 0;
  uint64_t scatter_ns = 0;
};

struct ServerReceivedValue {
  ServerHostLeaseHandle lease;
  std::vector<uint8_t> bytes;
  ServerDeviceReadMetrics read_metrics{};
  bool used_device_push = false;
  uint64_t reserve_ns = 0;
  uint64_t ack_ns = 0;
  uint64_t push_wait_ns = 0;
};

struct ServerSupplyMetrics {
  uint64_t gather_ns = 0;
  uint64_t dma_ns = 0;
  uint64_t stage_copy_ns = 0;
  uint64_t stage_publish_ns = 0;
  uint64_t commit_ns = 0;
  uint64_t completion_wait_ns = 0;
  bool used_completion_l1_fallback = false;
};

struct ServerRuntimeAllocatorProfile {
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

ServerDeviceRuntimeHandle
create_server_device_runtime(const ServerDeviceRuntimeConfig &config);
void server_device_runtime_launch(ServerDeviceRuntime &runtime);
ServerDeviceRuntimeState
server_device_runtime_state(const ServerDeviceRuntime &runtime) noexcept;
bool server_device_runtime_is_ready(
    const ServerDeviceRuntime &runtime) noexcept;

ServerDescriptor
server_device_runtime_make_descriptor(const ServerDeviceRuntime &runtime);

std::optional<ServerHostRequest>
server_device_runtime_poll_host_request(ServerDeviceRuntime &runtime);
ServerReceivedValue
server_device_runtime_receive_value(ServerDeviceRuntime &runtime,
                                    const ServerHostRequest &request,
                                    bool prefer_mapped_lease);
ServerSupplyMetrics server_device_runtime_supply_value(
    ServerDeviceRuntime &runtime, const ServerHostRequest &request,
    const ServerHostLease *lease, std::span<const uint8_t> heap_bytes);
void server_device_runtime_complete_host_request(
    ServerDeviceRuntime &runtime, const ServerHostRequest &request);
void server_device_runtime_cancel_host_io(
    ServerDeviceRuntime &runtime) noexcept;

void server_device_runtime_drain_inflight_puts(ServerDeviceRuntime &runtime);
std::pair<uint32_t, uint32_t>
server_device_runtime_put_drain_progress(ServerDeviceRuntime &runtime);
ServerRuntimeAllocatorProfile
server_device_runtime_allocator_profile(const ServerDeviceRuntime &runtime);
void server_device_runtime_terminate(ServerDeviceRuntime &runtime);

} // namespace kvcache_manager::detail::server_runtime
