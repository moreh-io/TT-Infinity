// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#include "client_impl.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>

#include "backend_bridge.hpp"

namespace kvcache_manager {

using tt::tt_metal::Tensor;
using namespace native_kvm;

backend::ClientOperationsConfig Client::Impl::make_operations_config() const {
  return {
      .client_mesh = client_submesh_,
      .client_coord = client_coord_,
      .deployment_server_node = deployment_server_node_,
      .client_value_ring_depth = value_ring_depth_,
      .route_word_to_server = route_word_to_server_,
      .fabric_link_idx = fabric_link_idx_,
      .static_intermesh_lane = static_intermesh_lane_,
      .static_intermesh_galaxy_to_t3k = static_intermesh_galaxy_to_t3k_,
      .profile_enabled = kvcache_profile_,
      .put_phase_profile_enabled = put_phase_profile_enabled_,
      .profile_detail = static_cast<uint8_t>(profile_detail_),
      .test_fault = backend::ClientOperationsTestFault::None,
  };
}

void Client::Impl::sync_operations_quarantine() noexcept {
  if (operations_ == nullptr ||
      backend::client_operations_state(*operations_) !=
          backend::ClientOperationsState::Quarantined ||
      lifecycle_state_ == LifecycleState::Quarantined) {
    return;
  }

  lifecycle_state_ = LifecycleState::Quarantined;
  quarantine_resources();
}

backend::ClientOperations &Client::Impl::checked_operations() const {
  TT_FATAL(lifecycle_state_ == LifecycleState::Established &&
               operations_ != nullptr,
           "Client is not established (closed or quarantined)");
  return *operations_;
}

std::vector<uint8_t> Client::Impl::ping(std::span<const uint8_t> payload) {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  auto &operations = checked_operations();
  try {
    auto echoed = backend::client_operations_ping(operations, payload);
    sync_operations_quarantine();
    return echoed;
  } catch (...) {
    sync_operations_quarantine();
    throw;
  }
}

void Client::Impl::put(const std::string &key, const Tensor &value_tensor,
                       uint32_t value_len_bytes, bool sync_commit) {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  auto &operations = checked_operations();
  try {
    backend::client_operations_put(operations, key, value_tensor,
                                   value_len_bytes, sync_commit);
    sync_operations_quarantine();
  } catch (...) {
    sync_operations_quarantine();
    throw;
  }
}

void Client::Impl::put_batch(const std::vector<std::string> &keys,
                             const std::vector<Tensor> &value_tensors,
                             const std::vector<uint32_t> &value_len_bytes,
                             bool sync_commit) {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  auto &operations = checked_operations();
  try {
    backend::client_operations_put_batch(operations, keys, value_tensors,
                                         value_len_bytes, sync_commit);
    sync_operations_quarantine();
  } catch (...) {
    sync_operations_quarantine();
    throw;
  }
}

PutProfile Client::Impl::put_profile() const {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  const auto profile =
      backend::client_operations_put_profile(checked_operations());
  return {
      .count = profile.count,
      .prepare_ns = profile.prepare_ns,
      .program_ns = profile.program_ns,
      .enqueue_ns = profile.enqueue_ns,
      .landed_wait_ns = profile.landed_wait_ns,
      .commit_wait_ns = profile.commit_wait_ns,
      .total_ns = profile.total_ns,
  };
}

void Client::Impl::flush() {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  auto &operations = checked_operations();
  try {
    backend::client_operations_flush(operations);
    sync_operations_quarantine();
  } catch (...) {
    sync_operations_quarantine();
    throw;
  }
}

bool Client::Impl::get(const std::string &key, const Tensor &out_tensor) {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  auto &operations = checked_operations();
  try {
    const bool found =
        backend::client_operations_get(operations, key, out_tensor);
    sync_operations_quarantine();
    return found;
  } catch (...) {
    sync_operations_quarantine();
    throw;
  }
}

std::vector<bool>
Client::Impl::get_parallel(const std::vector<Impl *> &clients,
                           const std::vector<std::string> &keys,
                           const Tensor &out_tensor) {
  TT_FATAL(!clients.empty(), "get_parallel: clients must not be empty");
  TT_FATAL(clients.size() == keys.size(),
           "get_parallel: clients and keys must have equal length");

  std::vector<Impl *> lock_order = clients;
  for (std::size_t i = 0; i < lock_order.size(); ++i) {
    TT_FATAL(lock_order[i] != nullptr, "get_parallel: client {} is null", i);
  }
  std::sort(lock_order.begin(), lock_order.end(), std::less<Impl *>{});
  TT_FATAL(std::adjacent_find(lock_order.begin(), lock_order.end()) ==
               lock_order.end(),
           "get_parallel: clients must be distinct");

  std::vector<std::unique_lock<std::mutex>> lifecycle_locks;
  lifecycle_locks.reserve(lock_order.size());
  for (Impl *client : lock_order) {
    lifecycle_locks.emplace_back(client->lifecycle_mutex_);
  }

  std::vector<backend::ClientOperations *> operations;
  operations.reserve(clients.size());
  for (Impl *client : clients) {
    operations.push_back(&client->checked_operations());
  }

  try {
    auto found =
        backend::client_operations_get_parallel(operations, keys, out_tensor);
    for (Impl *client : clients) {
      client->sync_operations_quarantine();
    }
    return found;
  } catch (...) {
    for (Impl *client : clients) {
      client->sync_operations_quarantine();
    }
    throw;
  }
}

bool Client::Impl::exists(const std::string &key) {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  auto &operations = checked_operations();
  try {
    const bool found = backend::client_operations_exists(operations, key);
    sync_operations_quarantine();
    return found;
  } catch (...) {
    sync_operations_quarantine();
    throw;
  }
}

bool Client::Impl::remove(const std::string &key) {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  auto &operations = checked_operations();
  try {
    const bool removed = backend::client_operations_remove(operations, key);
    sync_operations_quarantine();
    return removed;
  } catch (...) {
    sync_operations_quarantine();
    throw;
  }
}

void Client::Impl::move_to_host(const std::string &key) {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  auto &operations = checked_operations();
  try {
    backend::client_operations_move_to_host(operations, key);
    sync_operations_quarantine();
  } catch (...) {
    sync_operations_quarantine();
    throw;
  }
}

void Client::Impl::move_to_device(const std::string &key) {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  auto &operations = checked_operations();
  try {
    backend::client_operations_move_to_device(operations, key);
    sync_operations_quarantine();
  } catch (...) {
    sync_operations_quarantine();
    throw;
  }
}

} // namespace kvcache_manager
