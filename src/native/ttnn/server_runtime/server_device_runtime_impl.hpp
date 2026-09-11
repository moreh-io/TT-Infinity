// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include "server_runtime/server_device_runtime.hpp"

namespace kvcache_manager::detail::server_runtime {

class ServerRuntimeCore;

class __attribute__((visibility("hidden"))) ServerHostLease {
public:
  ServerHostLease(
      ::kvcache_manager::detail::server_runtime::ServerRuntimeCore *owner,
      uint32_t slot, uint32_t pool_unit, uint32_t pool_units,
      uint32_t read_offset, uint32_t generation, uint32_t value_len) noexcept;
  ~ServerHostLease() noexcept;

  ServerHostLease(const ServerHostLease &) = delete;
  ServerHostLease &operator=(const ServerHostLease &) = delete;

  [[nodiscard]] uint32_t read_offset() const noexcept { return read_offset_; }
  [[nodiscard]] uint32_t value_len() const noexcept { return value_len_; }
  [[nodiscard]] const ::kvcache_manager::detail::server_runtime::
      ServerRuntimeCore *
      owner() const noexcept {
    return owner_;
  }

private:
  friend class ::kvcache_manager::detail::server_runtime::ServerRuntimeCore;

  ::kvcache_manager::detail::server_runtime::ServerRuntimeCore *owner_ =
      nullptr;
  uint32_t slot_ = 0;
  uint32_t pool_unit_ = 0;
  uint32_t pool_units_ = 0;
  uint32_t read_offset_ = 0;
  uint32_t generation_ = 0;
  uint32_t value_len_ = 0;
};

class __attribute__((visibility("hidden"))) ServerDeviceRuntime {
public:
  explicit ServerDeviceRuntime(const ServerDeviceRuntimeConfig &config);
  ~ServerDeviceRuntime() noexcept;

  ServerDeviceRuntime(const ServerDeviceRuntime &) = delete;
  ServerDeviceRuntime &operator=(const ServerDeviceRuntime &) = delete;

  void launch();
  [[nodiscard]] ServerDeviceRuntimeState state() const noexcept;
  [[nodiscard]] bool is_ready() const noexcept;
  [[nodiscard]] ServerDescriptor make_descriptor() const;
  [[nodiscard]] std::optional<ServerHostRequest> poll_host_request();
  [[nodiscard]] ServerReceivedValue
  receive_value(const ServerHostRequest &request, bool prefer_mapped_lease);
  [[nodiscard]] ServerSupplyMetrics
  supply_value(const ServerHostRequest &request, const ServerHostLease *lease,
               std::span<const uint8_t> heap_bytes);
  void complete_host_request(const ServerHostRequest &request);
  void cancel_host_io() noexcept;
  void drain_inflight_puts();
  [[nodiscard]] std::pair<uint32_t, uint32_t> put_drain_progress();
  [[nodiscard]] ServerRuntimeAllocatorProfile allocator_profile() const;
  void terminate();

private:
  std::unique_ptr<::kvcache_manager::detail::server_runtime::ServerRuntimeCore>
      server_;
  std::atomic<ServerDeviceRuntimeState> state_{
      ServerDeviceRuntimeState::Prepared};
};

} // namespace kvcache_manager::detail::server_runtime
