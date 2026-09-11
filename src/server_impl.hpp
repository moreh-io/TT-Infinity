// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "backend_bridge.hpp"
#include "kvcache_manager/server.hpp"

namespace kvcache_manager {

class __attribute__((visibility("hidden"))) Server::Impl {
public:
  Impl(tt::tt_metal::distributed::MeshDevice *server_submesh,
       bool static_intermesh_t3k_server,
       uint32_t static_intermesh_remote_mesh_id, uint32_t worker_group,
       bool test_delay_first_establish_response = false,
       bool test_drop_first_establish_response = false,
       bool test_delay_first_put_commit = false);
  ~Impl() noexcept;

  [[nodiscard]] bool is_ready() const;
  [[nodiscard]] uint64_t moves_completed() const;
  [[nodiscard]] MoveProfile move_profile() const;
  [[nodiscard]] AllocatorProfile allocator_profile() const;
  [[nodiscard]] std::pair<uint32_t, uint32_t> test_put_drain_progress();
  void close();

private:
  enum class LifecycleState : uint8_t {
    Constructing,
    Ready,
    Closing,
    Quarantined,
    Closed
  };

  struct HostValue {
    uint32_t len = 0;
    detail::server_runtime::ServerHostLeaseHandle lease;
    std::vector<uint8_t> bytes;
  };

  void tiering_worker_loop();
  void
  handle_host_request(const detail::server_runtime::ServerHostRequest &request);
  void record_tiering_worker_failure(std::exception_ptr failure) noexcept;
  [[nodiscard]] std::exception_ptr tiering_worker_failure() const noexcept;
  void throw_if_tiering_worker_failed() const;
  void stop_and_join_worker() noexcept;
  void clear_host_values() noexcept;
  void force_quarantine() noexcept;
  void unregister_server() noexcept;

  [[nodiscard]] detail::server_runtime::ServerDeviceRuntime &checked_runtime();
  [[nodiscard]] const detail::server_runtime::ServerDeviceRuntime &
  checked_runtime() const;

  tt::tt_metal::distributed::MeshDevice *server_submesh_ = nullptr;
  uint32_t worker_group_ = 0;
  mutable std::mutex lifecycle_mutex_;
  LifecycleState lifecycle_state_ = LifecycleState::Constructing;
  detail::server_runtime::ServerDeviceRuntimeHandle runtime_;
  std::unique_ptr<std::thread> tiering_worker_;
  std::atomic<bool> worker_stop_{false};
  std::atomic<bool> tiering_worker_failed_{false};
  mutable std::mutex tiering_worker_failure_mutex_;
  std::exception_ptr tiering_worker_failure_;

  std::mutex host_values_mutex_;
  std::unordered_map<uint32_t, HostValue> host_values_;

  bool registry_registered_ = false;
  bool profile_enabled_ = false;
  uint32_t poll_us_ = 10;

  std::atomic<uint64_t> moves_completed_{0};
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
  std::atomic<uint64_t> move_to_device_count_{0};
  std::atomic<uint64_t> move_to_device_worker_ns_{0};
  std::atomic<uint64_t> move_to_device_complete_ns_{0};
};

} // namespace kvcache_manager
