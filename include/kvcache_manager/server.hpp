// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include "kvcache_manager/export.hpp"
#include "kvcache_manager/profiles.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace kvcache_manager {

struct ServerTestAccess;

class KVCACHE_MANAGER_API Server {
public:
  explicit Server(tt::tt_metal::distributed::MeshDevice *server_submesh,
                  bool static_intermesh_t3k_server = false,
                  uint32_t static_intermesh_remote_mesh_id = 0,
                  uint32_t worker_group = 0);
  ~Server() noexcept;

  Server(const Server &) = delete;
  Server &operator=(const Server &) = delete;
  Server(Server &&) = delete;
  Server &operator=(Server &&) = delete;

  [[nodiscard]] bool is_ready() const;
  [[nodiscard]] uint64_t moves_completed() const;
  [[nodiscard]] MoveProfile move_profile() const;
  [[nodiscard]] AllocatorProfile allocator_profile() const;
  void close();

private:
  friend struct ServerTestAccess;

  struct TestOptions {
    bool delay_first_establish_response = false;
    bool drop_first_establish_response = false;
    bool delay_first_put_commit = false;
  };

  Server(tt::tt_metal::distributed::MeshDevice *server_submesh,
         TestOptions test_options);
  [[nodiscard]] std::pair<uint32_t, uint32_t> test_put_drain_progress();

  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace kvcache_manager
