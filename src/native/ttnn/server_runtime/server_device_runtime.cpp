// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#include "server_runtime/server_device_runtime_impl.hpp"

#include <exception>
#include <memory>

#include <tt_stl/assert.hpp>

#include "server_runtime/server_runtime_core.hpp"

namespace kvcache_manager::detail::server_runtime {

ServerHostLease::ServerHostLease(
    ::kvcache_manager::detail::server_runtime::ServerRuntimeCore *owner,
    uint32_t slot, uint32_t pool_unit, uint32_t pool_units,
    uint32_t read_offset, uint32_t generation, uint32_t value_len) noexcept
    : owner_(owner), slot_(slot), pool_unit_(pool_unit),
      pool_units_(pool_units), read_offset_(read_offset),
      generation_(generation), value_len_(value_len) {}

ServerHostLease::~ServerHostLease() noexcept {
  if (owner_ != nullptr) {
    owner_->release_external_host_lease(*this);
  }
}

ServerDeviceRuntime::ServerDeviceRuntime(
    const ServerDeviceRuntimeConfig &config) {
  TT_FATAL(config.server_submesh != nullptr,
           "ServerDeviceRuntime server_submesh must be non-null");
  server_ = std::unique_ptr<
      ::kvcache_manager::detail::server_runtime::ServerRuntimeCore>(
      new ::kvcache_manager::detail::server_runtime::ServerRuntimeCore(
          config.server_submesh,
          ::kvcache_manager::detail::server_runtime::ServerRuntimeCore::
              TestOptions{
                  .delay_establish_ack =
                      config.test_delay_first_establish_response,
                  .drop_first_establish_response =
                      config.test_drop_first_establish_response,
                  .delay_first_put_commit = config.test_delay_first_put_commit,
              },
          config.static_intermesh_t3k_server,
          config.static_intermesh_remote_mesh_id, config.worker_group,
          ::kvcache_manager::detail::server_runtime::ServerRuntimeCore::
              HostPolicyMode::External));
}

ServerDeviceRuntime::~ServerDeviceRuntime() noexcept {
  if (server_ == nullptr) {
    return;
  }
  try {
    const auto current = state_.load(std::memory_order_acquire);
    if (current == ServerDeviceRuntimeState::Prepared ||
        current == ServerDeviceRuntimeState::Terminated) {
      server_->release_external_runtime_resources();
    } else if (current == ServerDeviceRuntimeState::Running ||
               current == ServerDeviceRuntimeState::Closing) {
      server_->quarantine_external_runtime();
      state_.store(ServerDeviceRuntimeState::Quarantined,
                   std::memory_order_release);
    }
  } catch (...) {
    server_->quarantine_external_runtime();
    state_.store(ServerDeviceRuntimeState::Quarantined,
                 std::memory_order_release);
  }
}

void ServerDeviceRuntime::launch() {
  TT_FATAL(state_.load(std::memory_order_acquire) ==
               ServerDeviceRuntimeState::Prepared,
           "ServerDeviceRuntime can launch only from Prepared");
  try {
    server_->launch_external_runtime();
  } catch (...) {
    if (server_->lifecycle_state_.load() ==
        ::kvcache_manager::detail::server_runtime::ServerRuntimeCore::
            LifecycleState::Quarantined) {
      state_.store(ServerDeviceRuntimeState::Quarantined,
                   std::memory_order_release);
    }
    throw;
  }
  state_.store(ServerDeviceRuntimeState::Running, std::memory_order_release);
}

ServerDeviceRuntimeState ServerDeviceRuntime::state() const noexcept {
  return state_.load(std::memory_order_acquire);
}

bool ServerDeviceRuntime::is_ready() const noexcept {
  return state() == ServerDeviceRuntimeState::Running && server_->is_ready();
}

ServerDescriptor ServerDeviceRuntime::make_descriptor() const {
  TT_FATAL(state() == ServerDeviceRuntimeState::Running,
           "ServerDeviceRuntime descriptor requires Running state");
  return server_->make_descriptor();
}

void ServerDeviceRuntime::cancel_host_io() noexcept {
  server_->cancel_external_host_io();
}

void ServerDeviceRuntime::drain_inflight_puts() {
  const auto current = state();
  TT_FATAL(current == ServerDeviceRuntimeState::Running ||
               current == ServerDeviceRuntimeState::Closing,
           "ServerDeviceRuntime PUT drain requires Running or Closing state");
  server_->drain_inflight_puts();
}

std::pair<uint32_t, uint32_t> ServerDeviceRuntime::put_drain_progress() {
  TT_FATAL(state() == ServerDeviceRuntimeState::Running,
           "ServerDeviceRuntime PUT progress requires Running state");
  return server_->put_drain_progress();
}

ServerRuntimeAllocatorProfile ServerDeviceRuntime::allocator_profile() const {
  TT_FATAL(state() == ServerDeviceRuntimeState::Running,
           "ServerDeviceRuntime allocator profile requires Running");
  const auto profile = server_->allocator_profile();
  return {
      .allocation_calls = profile.allocation_calls,
      .allocation_successes = profile.allocation_successes,
      .allocation_failures = profile.allocation_failures,
      .victim_scans = profile.victim_scans,
      .victims_evicted = profile.victims_evicted,
      .lease_calls = profile.lease_calls,
      .lease_successes = profile.lease_successes,
      .lease_failures = profile.lease_failures,
      .lease_victim_scans = profile.lease_victim_scans,
      .lease_victims_evicted = profile.lease_victims_evicted,
      .scan_cycles = profile.scan_cycles,
      .lease_scan_cycles = profile.lease_scan_cycles,
      .allocation_cycles = profile.allocation_cycles,
      .clock_mhz = profile.clock_mhz,
  };
}

void ServerDeviceRuntime::terminate() {
  const auto current = state();
  if (current == ServerDeviceRuntimeState::Terminated ||
      current == ServerDeviceRuntimeState::Quarantined) {
    return;
  }
  TT_FATAL(current == ServerDeviceRuntimeState::Running ||
               current == ServerDeviceRuntimeState::Closing,
           "ServerDeviceRuntime can terminate only from Running or Closing");
  state_.store(ServerDeviceRuntimeState::Closing, std::memory_order_release);
  server_->terminate_external_runtime();
  server_->cancel_external_host_io();
  state_.store(ServerDeviceRuntimeState::Terminated, std::memory_order_release);
}

void ServerDeviceRuntimeDeleter::operator()(
    ServerDeviceRuntime *runtime) const noexcept {
  delete runtime;
}
void ServerHostLeaseDeleter::operator()(ServerHostLease *lease) const noexcept {
  delete lease;
}

ServerDeviceRuntimeHandle
create_server_device_runtime(const ServerDeviceRuntimeConfig &config) {
  return ServerDeviceRuntimeHandle(new ServerDeviceRuntime(config));
}

void server_device_runtime_launch(ServerDeviceRuntime &runtime) {
  runtime.launch();
}

ServerDeviceRuntimeState
server_device_runtime_state(const ServerDeviceRuntime &runtime) noexcept {
  return runtime.state();
}

bool server_device_runtime_is_ready(
    const ServerDeviceRuntime &runtime) noexcept {
  return runtime.is_ready();
}

ServerDescriptor
server_device_runtime_make_descriptor(const ServerDeviceRuntime &runtime) {
  return runtime.make_descriptor();
}

void server_device_runtime_cancel_host_io(
    ServerDeviceRuntime &runtime) noexcept {
  runtime.cancel_host_io();
}

void server_device_runtime_drain_inflight_puts(ServerDeviceRuntime &runtime) {
  runtime.drain_inflight_puts();
}

std::pair<uint32_t, uint32_t>
server_device_runtime_put_drain_progress(ServerDeviceRuntime &runtime) {
  return runtime.put_drain_progress();
}

ServerRuntimeAllocatorProfile
server_device_runtime_allocator_profile(const ServerDeviceRuntime &runtime) {
  return runtime.allocator_profile();
}

void server_device_runtime_terminate(ServerDeviceRuntime &runtime) {
  runtime.terminate();
}

} // namespace kvcache_manager::detail::server_runtime
