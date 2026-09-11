// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#include "kvcache_manager/server.hpp"

#include "server_impl.hpp"

namespace kvcache_manager {

Server::Server(tt::tt_metal::distributed::MeshDevice *server_submesh,
               bool static_intermesh_t3k_server,
               uint32_t static_intermesh_remote_mesh_id, uint32_t worker_group)
    : impl_(std::make_unique<Impl>(server_submesh, static_intermesh_t3k_server,
                                   static_intermesh_remote_mesh_id,
                                   worker_group)) {}

Server::Server(tt::tt_metal::distributed::MeshDevice *server_submesh,
               TestOptions test_options)
    : impl_(std::make_unique<Impl>(server_submesh, false, 0, 0,
                                   test_options.delay_first_establish_response,
                                   test_options.drop_first_establish_response,
                                   test_options.delay_first_put_commit)) {}

Server::~Server() noexcept = default;

bool Server::is_ready() const { return impl_->is_ready(); }

uint64_t Server::moves_completed() const { return impl_->moves_completed(); }

MoveProfile Server::move_profile() const { return impl_->move_profile(); }

AllocatorProfile Server::allocator_profile() const {
  return impl_->allocator_profile();
}

std::pair<uint32_t, uint32_t> Server::test_put_drain_progress() {
  return impl_->test_put_drain_progress();
}

void Server::close() { impl_->close(); }

} // namespace kvcache_manager
