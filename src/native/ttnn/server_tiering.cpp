// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#include "server_impl.hpp"

#include <chrono>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>

namespace kvcache_manager {

void Server::Impl::tiering_worker_loop() {
  try {
    while (!worker_stop_.load(std::memory_order_acquire)) {
      auto request =
          detail::server_runtime::server_device_runtime_poll_host_request(
              checked_runtime());
      if (!request.has_value()) {
        if (!worker_stop_.load(std::memory_order_acquire)) {
          std::this_thread::sleep_for(std::chrono::microseconds(poll_us_));
        }
        continue;
      }
      handle_host_request(*request);
    }
  } catch (...) {
    if (!worker_stop_.load(std::memory_order_acquire)) {
      record_tiering_worker_failure(std::current_exception());
    }
  }
}

void Server::Impl::handle_host_request(
    const detail::server_runtime::ServerHostRequest &request) {
  using RequestKind = detail::server_runtime::ServerHostRequestKind;

  const auto move_started = std::chrono::steady_clock::now();
  auto worker_finished = move_started;
  detail::server_runtime::ServerReceivedValue received;
  detail::server_runtime::ServerSupplyMetrics supplied;

  switch (request.kind) {
  case RequestKind::WriteThrough: {
    // Retire the previous generation before the runtime publishes a new
    // mapped destination for this directory slot.
    {
      std::lock_guard<std::mutex> lock(host_values_mutex_);
      host_values_.erase(request.slot);
    }

    received = detail::server_runtime::server_device_runtime_receive_value(
        checked_runtime(), request, /*prefer_mapped_lease=*/true);
    if ((received.lease != nullptr) == (!received.bytes.empty())) {
      throw std::runtime_error(
          "write-through returned an invalid host backing representation");
    }

    HostValue value{
        .len = request.value_len,
        .lease = std::move(received.lease),
        .bytes = std::move(received.bytes),
    };
    {
      std::lock_guard<std::mutex> lock(host_values_mutex_);
      host_values_.insert_or_assign(request.slot, std::move(value));
    }
    worker_finished = std::chrono::steady_clock::now();
    break;
  }
  case RequestKind::RemoveBacking: {
    std::lock_guard<std::mutex> lock(host_values_mutex_);
    host_values_.erase(request.slot);
    worker_finished = std::chrono::steady_clock::now();
    break;
  }
  case RequestKind::PromoteToDevice:
  case RequestKind::ServeGet: {
    std::lock_guard<std::mutex> lock(host_values_mutex_);
    const auto value = host_values_.find(request.slot);
    if (value == host_values_.end()) {
      throw std::runtime_error(
          "device requested a host backing that manager does not own");
    }
    if (value->second.len != request.value_len) {
      throw std::runtime_error(
          "device request length does not match manager host backing");
    }
    const std::span<const uint8_t> bytes =
        value->second.lease == nullptr
            ? std::span<const uint8_t>(value->second.bytes)
            : std::span<const uint8_t>{};
    supplied = detail::server_runtime::server_device_runtime_supply_value(
        checked_runtime(), request, value->second.lease.get(), bytes);
    worker_finished = std::chrono::steady_clock::now();
    break;
  }
  }

  detail::server_runtime::server_device_runtime_complete_host_request(
      checked_runtime(), request);
  const auto move_finished = std::chrono::steady_clock::now();
  moves_completed_.fetch_add(1, std::memory_order_release);

  if (!profile_enabled_) {
    return;
  }

  const uint64_t worker_ns =
      request.poll_ns + std::chrono::duration_cast<std::chrono::nanoseconds>(
                            worker_finished - move_started)
                            .count();
  const uint64_t complete_ns =
      request.poll_ns + std::chrono::duration_cast<std::chrono::nanoseconds>(
                            move_finished - move_started)
                            .count();
  if (request.kind == RequestKind::WriteThrough) {
    move_to_host_count_.fetch_add(1, std::memory_order_relaxed);
    move_to_host_worker_ns_.fetch_add(worker_ns, std::memory_order_relaxed);
    move_to_host_complete_ns_.fetch_add(complete_ns, std::memory_order_relaxed);
    move_to_host_read_calls_.fetch_add(received.read_metrics.calls,
                                       std::memory_order_relaxed);
    move_to_host_read_bytes_.fetch_add(received.read_metrics.bytes,
                                       std::memory_order_relaxed);
    move_to_host_read_passes_.fetch_add(received.read_metrics.passes,
                                        std::memory_order_relaxed);
    move_to_host_read_retries_.fetch_add(received.read_metrics.retries,
                                         std::memory_order_relaxed);
    move_to_host_read_io_ns_.fetch_add(received.read_metrics.io_ns,
                                       std::memory_order_relaxed);
    move_to_host_scatter_ns_.fetch_add(received.read_metrics.scatter_ns,
                                       std::memory_order_relaxed);
    if (received.used_device_push) {
      move_to_host_device_pushes_.fetch_add(1, std::memory_order_relaxed);
    }
  } else if (request.kind == RequestKind::PromoteToDevice ||
             request.kind == RequestKind::ServeGet) {
    move_to_device_count_.fetch_add(1, std::memory_order_relaxed);
    move_to_device_worker_ns_.fetch_add(worker_ns, std::memory_order_relaxed);
    move_to_device_complete_ns_.fetch_add(complete_ns,
                                          std::memory_order_relaxed);
  }

  (void)supplied;
}

} // namespace kvcache_manager
