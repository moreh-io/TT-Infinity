// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#include "fake_backend.hpp"

#include <atomic>
#include <optional>
#include <span>
#include <stdexcept>

namespace kvcache_manager::test {

void FakeServerScript::enqueue(
    const kvcache_manager::detail::server_runtime::ServerHostRequest &request,
    std::vector<uint8_t> received_bytes) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!received_bytes.empty()) {
    received_values.insert_or_assign(request.sequence,
                                     std::move(received_bytes));
  }
  requests.push_back(request);
}

bool FakeServerScript::wait_for_completed(size_t count,
                                          std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex);
  return completion_cv.wait_for(lock, timeout, [this, count] {
    return completed_sequences.size() >= count;
  });
}

std::optional<std::vector<uint8_t>>
FakeServerScript::supplied_bytes(uint32_t sequence) const {
  std::lock_guard<std::mutex> lock(mutex);
  const auto value = supplied_values.find(sequence);
  if (value == supplied_values.end()) {
    return std::nullopt;
  }
  return value->second;
}

FakeBackendState &fake_backend_state() {
  static FakeBackendState state;
  return state;
}

void reset_fake_backend() { fake_backend_state() = FakeBackendState{}; }

} // namespace kvcache_manager::test

namespace ttnn::operations::kvcache_manager::backend {

uint32_t client_operations_abi_version() noexcept {
  return ::kvcache_manager::test::fake_backend_state()
      .client_operations_abi_version;
}

} // namespace ttnn::operations::kvcache_manager::backend

namespace kvcache_manager::detail::server_runtime {

namespace test = ::kvcache_manager::test;

class ServerDeviceRuntime {
public:
  explicit ServerDeviceRuntime(
      std::shared_ptr<test::FakeServerScript> server_script)
      : script(std::move(server_script)) {}

  std::atomic<ServerDeviceRuntimeState> state{
      ServerDeviceRuntimeState::Prepared};
  std::atomic<bool> cancelled{false};
  std::shared_ptr<test::FakeServerScript> script;
};

class ServerHostLease {};

void ServerDeviceRuntimeDeleter::operator()(
    ServerDeviceRuntime *runtime) const noexcept {
  auto &state = test::fake_backend_state();
  ++state.server_destroy_count;
  const auto runtime_state = runtime->state.load(std::memory_order_acquire);
  if (runtime_state == ServerDeviceRuntimeState::Running ||
      runtime_state == ServerDeviceRuntimeState::Closing) {
    ++state.server_unterminated_destroy_count;
  }
  delete runtime;
}

void ServerHostLeaseDeleter::operator()(ServerHostLease *lease) const noexcept {
  delete lease;
}

ServerDeviceRuntimeHandle
create_server_device_runtime(const ServerDeviceRuntimeConfig &config) {
  auto &state = test::fake_backend_state();
  ++state.server_create_count;
  state.server_mesh = config.server_submesh;
  state.static_intermesh_t3k_server = config.static_intermesh_t3k_server;
  state.static_intermesh_remote_mesh_id =
      config.static_intermesh_remote_mesh_id;
  state.server_worker_group = config.worker_group;
  return ServerDeviceRuntimeHandle(
      new ServerDeviceRuntime(state.server_script));
}

void server_device_runtime_launch(ServerDeviceRuntime &runtime) {
  test::fake_backend_state().calls.emplace_back("server_runtime_launch");
  if (test::fake_backend_state().fail_server_launch) {
    runtime.state.store(ServerDeviceRuntimeState::Quarantined,
                        std::memory_order_release);
    throw std::runtime_error("injected Server runtime launch failure");
  }
  runtime.state.store(ServerDeviceRuntimeState::Running,
                      std::memory_order_release);
}

ServerDeviceRuntimeState
server_device_runtime_state(const ServerDeviceRuntime &runtime) noexcept {
  return runtime.state.load(std::memory_order_acquire);
}

bool server_device_runtime_is_ready(
    const ServerDeviceRuntime &runtime) noexcept {
  return server_device_runtime_state(runtime) ==
         ServerDeviceRuntimeState::Running;
}

ServerDescriptor
server_device_runtime_make_descriptor(const ServerDeviceRuntime &) {
  return {};
}

std::optional<ServerHostRequest>
server_device_runtime_poll_host_request(ServerDeviceRuntime &runtime) {
  if (runtime.cancelled.load(std::memory_order_acquire)) {
    return std::nullopt;
  }
  std::lock_guard<std::mutex> lock(runtime.script->mutex);
  if (runtime.script->requests.empty()) {
    return std::nullopt;
  }
  auto request = runtime.script->requests.front();
  runtime.script->requests.pop_front();
  return request;
}

ServerReceivedValue
server_device_runtime_receive_value(ServerDeviceRuntime &runtime,
                                    const ServerHostRequest &request, bool) {
  ServerReceivedValue result;
  std::lock_guard<std::mutex> lock(runtime.script->mutex);
  const auto value = runtime.script->received_values.find(request.sequence);
  if (value != runtime.script->received_values.end()) {
    result.bytes = value->second;
  }
  return result;
}

ServerSupplyMetrics server_device_runtime_supply_value(
    ServerDeviceRuntime &runtime, const ServerHostRequest &request,
    const ServerHostLease *, std::span<const uint8_t> bytes) {
  std::lock_guard<std::mutex> lock(runtime.script->mutex);
  runtime.script->supplied_values.insert_or_assign(
      request.sequence, std::vector<uint8_t>(bytes.begin(), bytes.end()));
  return {};
}

void server_device_runtime_complete_host_request(
    ServerDeviceRuntime &runtime, const ServerHostRequest &request) {
  {
    std::lock_guard<std::mutex> lock(runtime.script->mutex);
    runtime.script->completed_sequences.push_back(request.sequence);
  }
  runtime.script->completion_cv.notify_all();
}

void server_device_runtime_cancel_host_io(
    ServerDeviceRuntime &runtime) noexcept {
  runtime.cancelled.store(true, std::memory_order_release);
  test::fake_backend_state().calls.emplace_back("server_runtime_cancel");
}

void server_device_runtime_drain_inflight_puts(ServerDeviceRuntime &) {
  test::fake_backend_state().calls.emplace_back("server_runtime_drain");
}

std::pair<uint32_t, uint32_t>
server_device_runtime_put_drain_progress(ServerDeviceRuntime &) {
  return {0, 0};
}

ServerRuntimeAllocatorProfile
server_device_runtime_allocator_profile(const ServerDeviceRuntime &) {
  test::fake_backend_state().calls.emplace_back(
      "server_runtime_allocator_profile");
  return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
}

void server_device_runtime_terminate(ServerDeviceRuntime &runtime) {
  test::fake_backend_state().calls.emplace_back("server_runtime_terminate");
  runtime.state.store(ServerDeviceRuntimeState::Closing,
                      std::memory_order_release);
  if (test::fake_backend_state().server_terminate_failures > 0) {
    --test::fake_backend_state().server_terminate_failures;
    throw std::runtime_error("injected Server runtime terminate failure");
  }
  runtime.state.store(ServerDeviceRuntimeState::Terminated,
                      std::memory_order_release);
}

} // namespace kvcache_manager::detail::server_runtime
