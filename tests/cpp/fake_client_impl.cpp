// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#include "client_impl.hpp"

#include <algorithm>

#include "backend_bridge.hpp"
#include "fake_backend.hpp"

namespace kvcache_manager {

struct Client::Impl::QuarantineResources {};

Client::Impl::Impl(
    tt::tt_metal::distributed::MeshDevice *client_mesh,
    std::optional<tt::tt_fabric::FabricNodeId> server_node,
    const std::optional<tt::tt_metal::distributed::MeshCoordinate>
        &client_coord,
    bool static_intermesh_galaxy_to_t3k, uint32_t server_worker_group,
    uint32_t static_intermesh_lane) {
  detail::verify_client_operations_abi();
  auto &state = test::fake_backend_state();
  ++state.client_create_count;
  state.client_mesh = client_mesh;
  state.client_server_node = std::move(server_node);
  state.client_coord = client_coord;
  state.static_intermesh_galaxy_to_t3k = static_intermesh_galaxy_to_t3k;
  state.client_server_worker_group = server_worker_group;
  state.static_intermesh_lane = static_intermesh_lane;
  lifecycle_state_ = LifecycleState::Established;
}

Client::Impl::~Impl() noexcept {
  close();
  ++test::fake_backend_state().client_destroy_count;
}

bool Client::Impl::is_established() const {
  test::fake_backend_state().calls.emplace_back("client_is_established");
  return lifecycle_state_ == LifecycleState::Established;
}

std::vector<uint8_t> Client::Impl::ping(std::span<const uint8_t> payload) {
  test::fake_backend_state().calls.emplace_back("client_ping");
  return {payload.begin(), payload.end()};
}

void Client::Impl::put(const std::string &key,
                       const tt::tt_metal::Tensor &value_tensor,
                       uint32_t value_len_bytes, bool sync_commit) {
  auto &state = test::fake_backend_state();
  state.calls.emplace_back("client_put");
  state.key = key;
  state.tensor = &value_tensor;
  state.value_len_bytes = value_len_bytes;
  state.sync_commit = sync_commit;
}

void Client::Impl::put_batch(
    const std::vector<std::string> &keys,
    const std::vector<tt::tt_metal::Tensor> &value_tensors,
    const std::vector<uint32_t> &value_len_bytes, bool sync_commit) {
  auto &state = test::fake_backend_state();
  state.calls.emplace_back("client_put_batch");
  state.keys = keys;
  state.tensors.clear();
  std::transform(value_tensors.begin(), value_tensors.end(),
                 std::back_inserter(state.tensors),
                 [](const auto &tensor) { return &tensor; });
  state.value_lengths = value_len_bytes;
  state.sync_commit = sync_commit;
}

PutProfile Client::Impl::put_profile() const {
  test::fake_backend_state().calls.emplace_back("client_put_profile");
  return {1, 2, 3, 4, 5, 6, 7};
}

void Client::Impl::flush() {
  test::fake_backend_state().calls.emplace_back("client_flush");
}

bool Client::Impl::get(const std::string &key,
                       const tt::tt_metal::Tensor &out_tensor) {
  auto &state = test::fake_backend_state();
  state.calls.emplace_back("client_get");
  state.key = key;
  state.tensor = &out_tensor;
  return true;
}

std::vector<bool>
Client::Impl::get_parallel(const std::vector<Impl *> &clients,
                           const std::vector<std::string> &keys,
                           const tt::tt_metal::Tensor &out_tensor) {
  auto &state = test::fake_backend_state();
  state.calls.emplace_back("client_get_parallel");
  state.keys = keys;
  state.tensor = &out_tensor;
  return std::vector<bool>(clients.size(), true);
}

bool Client::Impl::exists(const std::string &key) {
  auto &state = test::fake_backend_state();
  state.calls.emplace_back("client_exists");
  state.key = key;
  return false;
}

bool Client::Impl::remove(const std::string &key) {
  auto &state = test::fake_backend_state();
  state.calls.emplace_back("client_remove");
  state.key = key;
  return true;
}

void Client::Impl::move_to_host(const std::string &key) {
  auto &state = test::fake_backend_state();
  state.calls.emplace_back("client_move_to_host");
  state.key = key;
}

void Client::Impl::move_to_device(const std::string &key) {
  auto &state = test::fake_backend_state();
  state.calls.emplace_back("client_move_to_device");
  state.key = key;
}

void Client::Impl::close() {
  if (lifecycle_state_ == LifecycleState::Closed) {
    return;
  }
  test::fake_backend_state().calls.emplace_back("client_close");
  lifecycle_state_ = LifecycleState::Closed;
}

} // namespace kvcache_manager
