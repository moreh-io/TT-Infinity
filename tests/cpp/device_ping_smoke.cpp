// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "kvcache_manager/client.hpp"
#include "kvcache_manager/server.hpp"

namespace {

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using Tensor = tt::tt_metal::Tensor;

constexpr uint32_t kValueBytes = 4288;

Tensor make_device_tensor(const std::shared_ptr<MeshDevice> &mesh,
                          const std::vector<uint8_t> &data) {
  if (data.size() != kValueBytes) {
    throw std::runtime_error("device smoke tensor must be one KV chunk");
  }
  const tt::tt_metal::TensorSpec spec(
      ttnn::Shape{1, kValueBytes},
      tt::tt_metal::TensorLayout(
          tt::tt_metal::DataType::UINT8, tt::tt_metal::Layout::ROW_MAJOR,
          tt::tt_metal::MemoryConfig{
              tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
              tt::tt_metal::BufferType::DRAM}));
  return Tensor::from_vector(data, spec).to_device(mesh.get());
}

std::vector<uint8_t> read_device_tensor(const Tensor &tensor) {
  const auto shards = ttnn::distributed::get_device_tensors(tensor.cpu());
  if (shards.size() != 1) {
    throw std::runtime_error("device smoke expected one output shard");
  }
  return shards.front().to_vector<uint8_t>();
}

void close_mesh_noexcept(std::shared_ptr<MeshDevice> &mesh) noexcept {
  if (mesh == nullptr) {
    return;
  }
  try {
    ttnn::distributed::close_mesh_device(mesh);
  } catch (...) {
  }
  mesh.reset();
}

void cleanup(std::unique_ptr<kvcache_manager::Client> &client,
             std::unique_ptr<kvcache_manager::Server> &server,
             std::shared_ptr<MeshDevice> &client_submesh,
             std::shared_ptr<MeshDevice> &server_submesh,
             std::shared_ptr<MeshDevice> &parent_mesh,
             bool fabric_enabled) noexcept {
  client.reset();
  server.reset();

  if (parent_mesh != nullptr) {
    try {
      parent_mesh->quiesce_devices();
    } catch (...) {
    }
  }
  close_mesh_noexcept(client_submesh);
  close_mesh_noexcept(server_submesh);
  close_mesh_noexcept(parent_mesh);

  if (fabric_enabled) {
    try {
      tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    } catch (...) {
    }
  }
}

} // namespace

int main() {
  std::shared_ptr<MeshDevice> parent_mesh;
  std::shared_ptr<MeshDevice> client_submesh;
  std::shared_ptr<MeshDevice> server_submesh;
  std::unique_ptr<kvcache_manager::Server> server;
  std::unique_ptr<kvcache_manager::Client> client;
  bool fabric_enabled = false;

  try {
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    fabric_enabled = true;
    parent_mesh = ttnn::distributed::open_mesh_device(
        ttnn::distributed::MeshShape{2, 1}, DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreConfig{});
    client_submesh =
        parent_mesh->create_submesh(ttnn::distributed::MeshShape{1, 1},
                                    ttnn::distributed::MeshCoordinate{0, 0});
    server_submesh =
        parent_mesh->create_submesh(ttnn::distributed::MeshShape{1, 1},
                                    ttnn::distributed::MeshCoordinate{1, 0});

    server = std::make_unique<kvcache_manager::Server>(server_submesh.get());
    if (!server->is_ready()) {
      throw std::runtime_error("server did not become ready");
    }
    const auto server_node = server_submesh->get_fabric_node_id(
        ttnn::distributed::MeshCoordinate{0, 0});
    client = std::make_unique<kvcache_manager::Client>(client_submesh.get(),
                                                       server_node);
    if (!client->is_established()) {
      throw std::runtime_error("client did not establish");
    }

    const std::array<uint8_t, 5> payload{1, 3, 5, 7, 9};
    if (client->ping(payload) !=
        std::vector<uint8_t>(payload.begin(), payload.end())) {
      throw std::runtime_error("Fabric ping payload mismatch");
    }

    {
      std::vector<uint8_t> expected(kValueBytes);
      for (uint32_t index = 0; index < kValueBytes; ++index) {
        expected[index] = static_cast<uint8_t>((index * 17 + 3) & 0xff);
      }
      auto value = make_device_tensor(client_submesh, expected);
      auto output = make_device_tensor(client_submesh,
                                       std::vector<uint8_t>(kValueBytes, 0));

      client->put("cpp-device-smoke", value, kValueBytes,
                  /*sync_commit=*/false);
      client->flush();
      if (!client->exists("cpp-device-smoke")) {
        throw std::runtime_error("Fabric EXISTS missed committed value");
      }
      if (!client->get("cpp-device-smoke", output) ||
          read_device_tensor(output) != expected) {
        throw std::runtime_error("Fabric device GET payload mismatch");
      }

      client->move_to_host("cpp-device-smoke");
      auto host_backed_output = make_device_tensor(
          client_submesh, std::vector<uint8_t>(kValueBytes, 0));
      if (!client->get("cpp-device-smoke", host_backed_output) ||
          read_device_tensor(host_backed_output) != expected) {
        throw std::runtime_error("Fabric host-backed GET payload mismatch");
      }

      client->move_to_device("cpp-device-smoke");
      auto promoted_output = make_device_tensor(
          client_submesh, std::vector<uint8_t>(kValueBytes, 0));
      if (!client->get("cpp-device-smoke", promoted_output) ||
          read_device_tensor(promoted_output) != expected) {
        throw std::runtime_error("Fabric promoted GET payload mismatch");
      }
      if (!client->remove("cpp-device-smoke") ||
          client->exists("cpp-device-smoke")) {
        throw std::runtime_error("Fabric REMOVE did not erase value");
      }
    }

    client->close();
    client.reset();
    server->close();
    server.reset();
    parent_mesh->quiesce_devices();
    close_mesh_noexcept(client_submesh);
    close_mesh_noexcept(server_submesh);
    close_mesh_noexcept(parent_mesh);
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    fabric_enabled = false;
  } catch (const std::exception &error) {
    cleanup(client, server, client_submesh, server_submesh, parent_mesh,
            fabric_enabled);
    std::cerr << error.what() << '\n';
    return 1;
  } catch (...) {
    cleanup(client, server, client_submesh, server_submesh, parent_mesh,
            fabric_enabled);
    std::cerr << "unknown device smoke failure\n";
    return 1;
  }

  std::cout << "C++ device ping smoke passed\n";
  return 0;
}
