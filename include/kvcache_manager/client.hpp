// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "kvcache_manager/export.hpp"
#include "kvcache_manager/profiles.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace kvcache_manager {

class KVCACHE_MANAGER_API Client {
public:
  Client(tt::tt_metal::distributed::MeshDevice *client_mesh,
         std::optional<tt::tt_fabric::FabricNodeId> server_node,
         std::optional<tt::tt_metal::distributed::MeshCoordinate> client_coord =
             std::nullopt,
         bool static_intermesh_galaxy_to_t3k = false,
         uint32_t server_worker_group = 0, uint32_t static_intermesh_lane = 0);
  ~Client() noexcept;

  Client(const Client &) = delete;
  Client &operator=(const Client &) = delete;
  Client(Client &&) = delete;
  Client &operator=(Client &&) = delete;

  [[nodiscard]] bool is_established() const;
  [[nodiscard]] std::vector<uint8_t> ping(std::span<const uint8_t> payload);
  void put(const std::string &key, const tt::tt_metal::Tensor &value_tensor,
           uint32_t value_len_bytes, bool sync_commit = false);
  void put_batch(const std::vector<std::string> &keys,
                 const std::vector<tt::tt_metal::Tensor> &value_tensors,
                 const std::vector<uint32_t> &value_len_bytes,
                 bool sync_commit = false);
  [[nodiscard]] PutProfile put_profile() const;
  void flush();
  [[nodiscard]] bool get(const std::string &key,
                         const tt::tt_metal::Tensor &out_tensor);
  [[nodiscard]] static std::vector<bool>
  get_parallel(const std::vector<Client *> &clients,
               const std::vector<std::string> &keys,
               const tt::tt_metal::Tensor &out_tensor);
  [[nodiscard]] bool exists(const std::string &key);
  [[nodiscard]] bool remove(const std::string &key);
  void move_to_host(const std::string &key);
  void move_to_device(const std::string &key);
  void close();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace kvcache_manager
