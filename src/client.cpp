// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#include "kvcache_manager/client.hpp"

#include <utility>

#include "client_impl.hpp"

namespace kvcache_manager {

Client::Client(
    tt::tt_metal::distributed::MeshDevice *client_mesh,
    std::optional<tt::tt_fabric::FabricNodeId> server_node,
    std::optional<tt::tt_metal::distributed::MeshCoordinate> client_coord,
    bool static_intermesh_galaxy_to_t3k, uint32_t server_worker_group,
    uint32_t static_intermesh_lane)
    : impl_(std::make_unique<Impl>(client_mesh, std::move(server_node),
                                   client_coord, static_intermesh_galaxy_to_t3k,
                                   server_worker_group,
                                   static_intermesh_lane)) {}

Client::~Client() noexcept = default;

bool Client::is_established() const { return impl_->is_established(); }

std::vector<uint8_t> Client::ping(std::span<const uint8_t> payload) {
  return impl_->ping(payload);
}

void Client::put(const std::string &key,
                 const tt::tt_metal::Tensor &value_tensor,
                 uint32_t value_len_bytes, bool sync_commit) {
  impl_->put(key, value_tensor, value_len_bytes, sync_commit);
}

void Client::put_batch(const std::vector<std::string> &keys,
                       const std::vector<tt::tt_metal::Tensor> &value_tensors,
                       const std::vector<uint32_t> &value_len_bytes,
                       bool sync_commit) {
  impl_->put_batch(keys, value_tensors, value_len_bytes, sync_commit);
}

PutProfile Client::put_profile() const { return impl_->put_profile(); }

void Client::flush() { impl_->flush(); }

bool Client::get(const std::string &key,
                 const tt::tt_metal::Tensor &out_tensor) {
  return impl_->get(key, out_tensor);
}

std::vector<bool> Client::get_parallel(const std::vector<Client *> &clients,
                                       const std::vector<std::string> &keys,
                                       const tt::tt_metal::Tensor &out_tensor) {
  std::vector<Impl *> implementations;
  implementations.reserve(clients.size());
  for (Client *client : clients) {
    implementations.push_back(client == nullptr ? nullptr
                                                : client->impl_.get());
  }
  return Impl::get_parallel(implementations, keys, out_tensor);
}

bool Client::exists(const std::string &key) { return impl_->exists(key); }

bool Client::remove(const std::string &key) { return impl_->remove(key); }

void Client::move_to_host(const std::string &key) { impl_->move_to_host(key); }

void Client::move_to_device(const std::string &key) {
  impl_->move_to_device(key);
}

void Client::close() { impl_->close(); }

} // namespace kvcache_manager
