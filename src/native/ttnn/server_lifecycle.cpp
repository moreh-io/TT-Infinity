// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#include "server_impl.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

#include <ttnn/operations/kvcache_manager/common/config.hpp>

namespace kvcache_manager {
namespace {

std::mutex registry_mutex;
using ServerRegistryKey =
    std::pair<tt::tt_metal::distributed::MeshDevice *, uint32_t>;
std::set<ServerRegistryKey> active_servers;

uint32_t resolve_poll_us() {
  constexpr uint32_t kDefaultPollUs = 10;
  constexpr uint32_t kMaxPollUs = 1'000'000;
  const auto parsed = ttnn::operations::kvcache_manager::config::read_u64_env(
      "KVM_SERVER_HOST_POLL_US");
  if (!parsed.has_value()) {
    return kDefaultPollUs;
  }
  if (*parsed == 0) {
    throw std::runtime_error("KVM_SERVER_HOST_POLL_US must be positive");
  }
  return static_cast<uint32_t>(std::min<uint64_t>(*parsed, kMaxPollUs));
}

} // namespace

Server::Impl::Impl(tt::tt_metal::distributed::MeshDevice *server_submesh,
                   bool static_intermesh_t3k_server,
                   uint32_t static_intermesh_remote_mesh_id,
                   uint32_t worker_group,
                   bool test_delay_first_establish_response,
                   bool test_drop_first_establish_response,
                   bool test_delay_first_put_commit)
    : server_submesh_(server_submesh), worker_group_(worker_group) {
  if (server_submesh_ == nullptr) {
    throw std::invalid_argument("server_submesh must be non-null");
  }

  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (!active_servers
             .insert(ServerRegistryKey{server_submesh_, worker_group_})
             .second) {
      throw std::runtime_error(
          "a Server is already active on this submesh worker group");
    }
    registry_registered_ = true;
  }

  try {
    profile_enabled_ = ttnn::operations::kvcache_manager::config::read_bool_env(
                           "KVM_SERVER_PROFILE")
                           .value_or(false);
    poll_us_ = resolve_poll_us();
    runtime_ = detail::server_runtime::create_server_device_runtime({
        .server_submesh = server_submesh_,
        .static_intermesh_t3k_server = static_intermesh_t3k_server,
        .static_intermesh_remote_mesh_id = static_intermesh_remote_mesh_id,
        .worker_group = worker_group_,
        .test_delay_first_establish_response =
            test_delay_first_establish_response,
        .test_drop_first_establish_response =
            test_drop_first_establish_response,
        .test_delay_first_put_commit = test_delay_first_put_commit,
    });

    constexpr uint32_t kWorkerWaiting = 0;
    constexpr uint32_t kWorkerRun = 1;
    constexpr uint32_t kWorkerCancel = 2;
    auto start_gate = std::make_shared<std::atomic<uint32_t>>(kWorkerWaiting);
    auto worker = std::make_unique<std::thread>([this, start_gate] {
      uint32_t action = kWorkerWaiting;
      while ((action = start_gate->load(std::memory_order_acquire)) ==
             kWorkerWaiting) {
        start_gate->wait(kWorkerWaiting, std::memory_order_acquire);
      }
      if (action == kWorkerRun) {
        tiering_worker_loop();
      }
    });

    try {
      detail::server_runtime::server_device_runtime_launch(*runtime_);
    } catch (...) {
      start_gate->store(kWorkerCancel, std::memory_order_release);
      start_gate->notify_one();
      worker->join();
      throw;
    }

    tiering_worker_ = std::move(worker);
    lifecycle_state_ = LifecycleState::Ready;
    start_gate->store(kWorkerRun, std::memory_order_release);
    start_gate->notify_one();
  } catch (...) {
    unregister_server();
    throw;
  }
}

Server::Impl::~Impl() noexcept {
  try {
    close();
  } catch (...) {
    try {
      close();
    } catch (...) {
      force_quarantine();
    }
  }
}

bool Server::Impl::is_ready() const {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  return lifecycle_state_ == LifecycleState::Ready &&
         !tiering_worker_failed_.load(std::memory_order_acquire) &&
         runtime_ != nullptr &&
         detail::server_runtime::server_device_runtime_is_ready(*runtime_);
}

uint64_t Server::Impl::moves_completed() const {
  return moves_completed_.load(std::memory_order_acquire);
}

MoveProfile Server::Impl::move_profile() const {
  return {
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

AllocatorProfile Server::Impl::allocator_profile() const {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if (lifecycle_state_ != LifecycleState::Ready) {
    throw std::runtime_error("Server is not ready");
  }
  throw_if_tiering_worker_failed();
  const auto profile =
      detail::server_runtime::server_device_runtime_allocator_profile(
          checked_runtime());
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

std::pair<uint32_t, uint32_t> Server::Impl::test_put_drain_progress() {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if (lifecycle_state_ != LifecycleState::Ready) {
    throw std::runtime_error("Server is not ready");
  }
  throw_if_tiering_worker_failed();
  return detail::server_runtime::server_device_runtime_put_drain_progress(
      checked_runtime());
}

void Server::Impl::close() {
  std::unique_lock<std::mutex> lock(lifecycle_mutex_);
  if (lifecycle_state_ == LifecycleState::Closed ||
      lifecycle_state_ == LifecycleState::Quarantined) {
    return;
  }
  if (lifecycle_state_ != LifecycleState::Ready &&
      lifecycle_state_ != LifecycleState::Closing) {
    throw std::runtime_error("Server cannot close from its current state");
  }

  if (lifecycle_state_ == LifecycleState::Ready) {
    lifecycle_state_ = LifecycleState::Closing;
    if (!tiering_worker_failed_.load(std::memory_order_acquire)) {
      try {
        detail::server_runtime::server_device_runtime_drain_inflight_puts(
            checked_runtime());
      } catch (...) {
        if (!tiering_worker_failed_.load(std::memory_order_acquire)) {
          lifecycle_state_ = LifecycleState::Ready;
        }
        throw;
      }
    }
  }

  if (tiering_worker_failed_.load(std::memory_order_acquire)) {
    const auto failure = tiering_worker_failure();
    lock.unlock();
    force_quarantine();
    if (failure != nullptr) {
      std::rethrow_exception(failure);
    }
    throw std::runtime_error("Server tiering worker failed");
  }

  detail::server_runtime::server_device_runtime_terminate(checked_runtime());
  worker_stop_.store(true, std::memory_order_release);
  detail::server_runtime::server_device_runtime_cancel_host_io(
      checked_runtime());
  stop_and_join_worker();
  clear_host_values();
  runtime_.reset();
  lifecycle_state_ = LifecycleState::Closed;
  unregister_server();
}

detail::server_runtime::ServerDeviceRuntime &Server::Impl::checked_runtime() {
  if (runtime_ == nullptr) {
    throw std::runtime_error("Server device runtime is unavailable");
  }
  return *runtime_;
}

const detail::server_runtime::ServerDeviceRuntime &
Server::Impl::checked_runtime() const {
  if (runtime_ == nullptr) {
    throw std::runtime_error("Server device runtime is unavailable");
  }
  return *runtime_;
}

void Server::Impl::record_tiering_worker_failure(
    std::exception_ptr failure) noexcept {
  {
    std::lock_guard<std::mutex> lock(tiering_worker_failure_mutex_);
    if (tiering_worker_failure_ == nullptr) {
      tiering_worker_failure_ = failure;
    }
  }
  tiering_worker_failed_.store(true, std::memory_order_release);
}

std::exception_ptr Server::Impl::tiering_worker_failure() const noexcept {
  std::lock_guard<std::mutex> lock(tiering_worker_failure_mutex_);
  return tiering_worker_failure_;
}

void Server::Impl::throw_if_tiering_worker_failed() const {
  if (!tiering_worker_failed_.load(std::memory_order_acquire)) {
    return;
  }
  const auto failure = tiering_worker_failure();
  if (failure != nullptr) {
    std::rethrow_exception(failure);
  }
  throw std::runtime_error("Server tiering worker failed");
}

void Server::Impl::stop_and_join_worker() noexcept {
  worker_stop_.store(true, std::memory_order_release);
  if (tiering_worker_ != nullptr) {
    if (tiering_worker_->joinable()) {
      tiering_worker_->join();
    }
    tiering_worker_.reset();
  }
}

void Server::Impl::clear_host_values() noexcept {
  std::lock_guard<std::mutex> lock(host_values_mutex_);
  host_values_.clear();
}

void Server::Impl::force_quarantine() noexcept {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if (lifecycle_state_ == LifecycleState::Closed ||
      lifecycle_state_ == LifecycleState::Quarantined) {
    return;
  }
  worker_stop_.store(true, std::memory_order_release);
  if (runtime_ != nullptr) {
    detail::server_runtime::server_device_runtime_cancel_host_io(*runtime_);
  }
  stop_and_join_worker();
  clear_host_values();
  runtime_.reset();
  lifecycle_state_ = LifecycleState::Quarantined;
  unregister_server();
}

void Server::Impl::unregister_server() noexcept {
  if (!registry_registered_) {
    return;
  }
  std::lock_guard<std::mutex> lock(registry_mutex);
  active_servers.erase(ServerRegistryKey{server_submesh_, worker_group_});
  registry_registered_ = false;
}

} // namespace kvcache_manager
