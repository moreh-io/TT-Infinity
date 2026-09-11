// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "server_runtime/server_device_runtime.hpp"
#include <ttnn/operations/kvcache_manager/client/client_operations.hpp>

namespace kvcache_manager::test {

struct FakeServerScript {
  void enqueue(
      const kvcache_manager::detail::server_runtime::ServerHostRequest &request,
      std::vector<uint8_t> received_bytes = {});
  [[nodiscard]] bool wait_for_completed(
      size_t count,
      std::chrono::milliseconds timeout = std::chrono::seconds(2));
  [[nodiscard]] std::optional<std::vector<uint8_t>>
  supplied_bytes(uint32_t sequence) const;

  mutable std::mutex mutex;
  std::condition_variable completion_cv;
  std::deque<kvcache_manager::detail::server_runtime::ServerHostRequest>
      requests;
  std::map<uint32_t, std::vector<uint8_t>> received_values;
  std::map<uint32_t, std::vector<uint8_t>> supplied_values;
  std::vector<uint32_t> completed_sequences;
};

struct FakeBackendState {
  uint32_t client_operations_abi_version =
      ttnn::operations::kvcache_manager::backend::kClientOperationsAbiVersion;
  int server_create_count = 0;
  int server_destroy_count = 0;
  int server_unterminated_destroy_count = 0;
  int client_create_count = 0;
  int client_destroy_count = 0;
  tt::tt_metal::distributed::MeshDevice *server_mesh = nullptr;
  tt::tt_metal::distributed::MeshDevice *client_mesh = nullptr;
  bool static_intermesh_t3k_server = false;
  bool fail_server_launch = false;
  uint32_t server_terminate_failures = 0;
  uint32_t static_intermesh_remote_mesh_id = 0;
  uint32_t server_worker_group = 0;
  bool static_intermesh_galaxy_to_t3k = false;
  uint32_t client_server_worker_group = 0;
  uint32_t static_intermesh_lane = 0;
  std::optional<tt::tt_metal::distributed::MeshCoordinate> client_coord;
  std::optional<tt::tt_fabric::FabricNodeId> client_server_node;
  std::vector<std::string> calls;
  std::string key;
  const tt::tt_metal::Tensor *tensor = nullptr;
  uint32_t value_len_bytes = 0;
  bool sync_commit = false;
  std::vector<std::string> keys;
  std::vector<const tt::tt_metal::Tensor *> tensors;
  std::vector<uint32_t> value_lengths;
  std::shared_ptr<FakeServerScript> server_script =
      std::make_shared<FakeServerScript>();
};

FakeBackendState &fake_backend_state();
void reset_fake_backend();

} // namespace kvcache_manager::test
