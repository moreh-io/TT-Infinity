// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "kvcache_manager/client.hpp"
#include "kvcache_manager/server.hpp"

namespace {

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using Tensor = tt::tt_metal::Tensor;
using namespace std::chrono_literals;

constexpr uint32_t kChunkBytes = 4288;
constexpr uint32_t kGalaxyEndpointChip = 24;
constexpr uint32_t kT3kEndpointChip = 1;
constexpr uint32_t kMeshColumns = 4;

enum class Role { Server, Client };

struct Arguments {
  Role role = Role::Server;
  std::filesystem::path stop_file;
  uint32_t link_index = 1;
  uint32_t remote_mesh_id = 0;
  uint32_t source_chip = kGalaxyEndpointChip;
  uint32_t iterations = 3;
  uint32_t payload_bytes = 16 * 1024;
};

volatile std::sig_atomic_t stop_requested = 0;

void request_stop(int) { stop_requested = 1; }

[[noreturn]] void usage_error(const std::string &message) {
  throw std::invalid_argument(
      message + "\nusage: kvcache_manager_cpp_static_intermesh_smoke "
                "server --stop-file FILE [--link-index N] "
                "[--remote-mesh-id N]\n"
                "   or: kvcache_manager_cpp_static_intermesh_smoke "
                "client [--link-index N] [--source-chip N] "
                "[--iterations N] [--payload-bytes N]");
}

uint32_t parse_u32(std::string_view text, std::string_view option) {
  uint64_t value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() ||
      value > std::numeric_limits<uint32_t>::max()) {
    usage_error(std::string(option) + " requires an unsigned 32-bit integer");
  }
  return static_cast<uint32_t>(value);
}

Arguments parse_arguments(int argc, char **argv) {
  if (argc < 2) {
    usage_error("missing role");
  }

  Arguments args;
  const std::string_view role(argv[1]);
  if (role == "server") {
    args.role = Role::Server;
  } else if (role == "client") {
    args.role = Role::Client;
  } else {
    usage_error("role must be server or client");
  }

  for (int index = 2; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (index + 1 >= argc) {
      usage_error(std::string(option) + " requires a value");
    }
    const std::string_view value(argv[++index]);
    if (option == "--stop-file") {
      args.stop_file = value;
    } else if (option == "--link-index") {
      args.link_index = parse_u32(value, option);
    } else if (option == "--remote-mesh-id") {
      args.remote_mesh_id = parse_u32(value, option);
    } else if (option == "--source-chip") {
      args.source_chip = parse_u32(value, option);
    } else if (option == "--iterations") {
      args.iterations = parse_u32(value, option);
    } else if (option == "--payload-bytes") {
      args.payload_bytes = parse_u32(value, option);
    } else {
      usage_error("unknown option: " + std::string(option));
    }
  }

  if (args.role == Role::Server && args.stop_file.empty()) {
    usage_error("server requires --stop-file");
  }
  if (args.iterations == 0 || args.payload_bytes == 0) {
    usage_error("iterations and payload bytes must be positive");
  }
  return args;
}

tt::tt_fabric::FabricRouterConfig router_config(Role role,
                                                uint32_t link_index) {
  tt::tt_fabric::FabricRouterConfig config;
  config.intermesh_vc0_only = true;
  config.static_intermesh_endpoint_link_index = link_index;
  if (role == Role::Server) {
    config.static_intermesh_endpoint_role =
        tt::tt_fabric::StaticIntermeshEndpointRole::T3K_ENDPOINT;
    config.static_intermesh_endpoint_physical_chip = kT3kEndpointChip;
    config.static_intermesh_peer_handshake_address = 0x11000;
    config.static_intermesh_peer_receiver_base_address = 0x2FD60;
    config.static_intermesh_peer_receiver_num_slots = 11;
  } else {
    config.static_intermesh_endpoint_role =
        tt::tt_fabric::StaticIntermeshEndpointRole::GALAXY_GATEWAY;
    config.static_intermesh_endpoint_physical_chip = kGalaxyEndpointChip;
    config.static_intermesh_peer_handshake_address = 0x18000;
    config.static_intermesh_peer_receiver_base_address = 0x29CE0;
    config.static_intermesh_peer_receiver_num_slots = 16;
  }
  return config;
}

void enable_fabric(Role role, uint32_t link_index) {
  tt::tt_fabric::SetFabricConfig(
      tt::tt_fabric::FabricConfig::FABRIC_2D,
      tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
      std::nullopt, tt::tt_fabric::FabricTensixConfig::DISABLED,
      tt::tt_fabric::FabricUDMMode::DISABLED,
      tt::tt_fabric::FabricManagerMode::DEFAULT,
      router_config(role, link_index));
}

std::shared_ptr<MeshDevice> open_parent_mesh(Role role) {
  const auto shape =
      role == Role::Server
          ? tt::tt_metal::distributed::MeshShape{2, kMeshColumns}
          : tt::tt_metal::distributed::MeshShape{8, kMeshColumns};
  return ttnn::distributed::open_mesh_device(
      shape, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1,
      tt::tt_metal::DispatchCoreConfig{});
}

std::shared_ptr<MeshDevice>
endpoint_submesh(const std::shared_ptr<MeshDevice> &parent,
                 uint32_t physical_chip) {
  const auto device_ids = parent->get_device_ids();
  for (size_t index = 0; index < device_ids.size(); ++index) {
    if (static_cast<uint32_t>(device_ids[index]) != physical_chip) {
      continue;
    }
    return parent->create_submesh(
        tt::tt_metal::distributed::MeshShape{1, 1},
        tt::tt_metal::distributed::MeshCoordinate{
            static_cast<uint32_t>(index / kMeshColumns),
            static_cast<uint32_t>(index % kMeshColumns)});
  }
  throw std::runtime_error("physical endpoint chip is not in the opened mesh");
}

uint32_t padded_value_bytes(uint32_t payload_bytes) {
  const uint64_t padded =
      (static_cast<uint64_t>(payload_bytes) + kChunkBytes - 1) / kChunkBytes *
      kChunkBytes;
  if (padded > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("padded payload exceeds uint32 range");
  }
  return static_cast<uint32_t>(padded);
}

Tensor make_device_tensor(const std::shared_ptr<MeshDevice> &mesh,
                          const std::vector<uint8_t> &data,
                          uint32_t payload_bytes) {
  if (data.size() != payload_bytes) {
    throw std::runtime_error(
        "tensor data length does not match payload length");
  }
  const uint32_t padded_bytes = padded_value_bytes(payload_bytes);
  std::vector<uint8_t> padded(padded_bytes);
  std::copy(data.begin(), data.end(), padded.begin());
  const tt::tt_metal::TensorSpec spec(
      ttnn::Shape{padded_bytes / kChunkBytes, kChunkBytes},
      tt::tt_metal::TensorLayout(
          tt::tt_metal::DataType::UINT8, tt::tt_metal::Layout::ROW_MAJOR,
          tt::tt_metal::MemoryConfig{
              tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
              tt::tt_metal::BufferType::DRAM}));
  return Tensor::from_vector(padded, spec).to_device(mesh.get());
}

Tensor make_zero_tensor(const std::shared_ptr<MeshDevice> &mesh,
                        uint32_t payload_bytes) {
  return make_device_tensor(mesh, std::vector<uint8_t>(payload_bytes, 0),
                            payload_bytes);
}

std::vector<uint8_t> read_device_tensor(const Tensor &tensor,
                                        uint32_t payload_bytes) {
  const auto shards = ttnn::distributed::get_device_tensors(tensor.cpu());
  if (shards.size() != 1) {
    throw std::runtime_error("internode smoke expected one output shard");
  }
  auto bytes = shards.front().to_vector<uint8_t>();
  if (bytes.size() < payload_bytes) {
    throw std::runtime_error("internode smoke output is shorter than payload");
  }
  bytes.resize(payload_bytes);
  return bytes;
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
             std::shared_ptr<MeshDevice> &endpoint_mesh,
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
  close_mesh_noexcept(endpoint_mesh);
  close_mesh_noexcept(parent_mesh);
  if (fabric_enabled) {
    try {
      tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    } catch (...) {
    }
  }
}

void run_server(const Arguments &args) {
  bool fabric_enabled = false;
  std::shared_ptr<MeshDevice> parent_mesh;
  std::shared_ptr<MeshDevice> server_submesh;
  std::unique_ptr<kvcache_manager::Server> server;
  std::unique_ptr<kvcache_manager::Client> no_client;

  try {
    enable_fabric(Role::Server, args.link_index);
    fabric_enabled = true;
    parent_mesh = open_parent_mesh(Role::Server);
    server_submesh = endpoint_submesh(parent_mesh, kT3kEndpointChip);
    server = std::make_unique<kvcache_manager::Server>(
        server_submesh.get(), /*static_intermesh_t3k_server=*/true,
        args.remote_mesh_id);
    if (!server->is_ready()) {
      throw std::runtime_error("server did not become ready");
    }

    std::cout << "SERVER_READY" << std::endl;

    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
    while (stop_requested == 0) {
      std::error_code error;
      if (std::filesystem::exists(args.stop_file, error)) {
        break;
      }
      if (error) {
        throw std::runtime_error("failed to inspect server stop file");
      }
      std::this_thread::sleep_for(100ms);
    }

    server->close();
    server.reset();
    parent_mesh->quiesce_devices();
    close_mesh_noexcept(server_submesh);
    close_mesh_noexcept(parent_mesh);
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    fabric_enabled = false;
    std::cout << "SERVER_STOPPED" << std::endl;
  } catch (...) {
    cleanup(no_client, server, server_submesh, parent_mesh, fabric_enabled);
    throw;
  }
}

void verify_get(kvcache_manager::Client &client,
                const std::shared_ptr<MeshDevice> &client_submesh,
                const std::string &key, const std::vector<uint8_t> &expected,
                std::string_view stage) {
  const auto payload_bytes = static_cast<uint32_t>(expected.size());
  auto output = make_zero_tensor(client_submesh, payload_bytes);
  if (!client.get(key, output)) {
    throw std::runtime_error(std::string(stage) + " GET missed");
  }
  if (read_device_tensor(output, payload_bytes) != expected) {
    throw std::runtime_error(std::string(stage) + " GET payload mismatch");
  }
}

void run_client(const Arguments &args) {
  bool fabric_enabled = false;
  std::shared_ptr<MeshDevice> parent_mesh;
  std::shared_ptr<MeshDevice> client_submesh;
  std::unique_ptr<kvcache_manager::Client> client;
  std::unique_ptr<kvcache_manager::Server> no_server;

  try {
    enable_fabric(Role::Client, args.link_index);
    fabric_enabled = true;
    parent_mesh = open_parent_mesh(Role::Client);
    client_submesh = endpoint_submesh(parent_mesh, args.source_chip);
    std::cout << "CLIENT_FABRIC_READY" << std::endl;

    client = std::make_unique<kvcache_manager::Client>(
        client_submesh.get(), std::nullopt, std::nullopt,
        /*static_intermesh_galaxy_to_t3k=*/true);
    if (!client->is_established()) {
      throw std::runtime_error("client did not establish");
    }

    const std::array<uint8_t, 64> ping_payload = [] {
      std::array<uint8_t, 64> payload{};
      for (uint32_t index = 0; index < payload.size(); ++index) {
        payload[index] = static_cast<uint8_t>(index);
      }
      return payload;
    }();
    if (client->ping(ping_payload) !=
        std::vector<uint8_t>(ping_payload.begin(), ping_payload.end())) {
      throw std::runtime_error("PING response did not match the request");
    }
    std::cout << "PASS establish+ping" << std::endl;

    for (uint32_t iteration = 0; iteration < args.iterations; ++iteration) {
      const std::string key =
          "static-intermesh-smoke-" + std::to_string(iteration);
      std::vector<uint8_t> expected(args.payload_bytes);
      for (uint32_t offset = 0; offset < args.payload_bytes; ++offset) {
        expected[offset] = static_cast<uint8_t>((iteration + offset) & 0xff);
      }
      auto value =
          make_device_tensor(client_submesh, expected, args.payload_bytes);
      const bool sync_commit = iteration % 2 == 0;
      client->put(key, value, args.payload_bytes, sync_commit);
      if (!sync_commit) {
        client->flush();
      }
      if (!client->exists(key)) {
        throw std::runtime_error("EXISTS missed the committed PUT");
      }
      verify_get(*client, client_submesh, key, expected, "device-backed");

      client->move_to_host(key);
      verify_get(*client, client_submesh, key, expected, "host-backed");
      client->move_to_device(key);
      verify_get(*client, client_submesh, key, expected, "promoted");

      if (!client->remove(key) || client->exists(key)) {
        throw std::runtime_error("REMOVE/EXISTS result mismatch");
      }
      std::cout << "PASS iteration=" << iteration
                << " bytes=" << args.payload_bytes
                << " put=" << (sync_commit ? "committed" : "landed+flush")
                << " move=host+device remove=ok" << std::endl;
    }

    client->close();
    client.reset();
    parent_mesh->quiesce_devices();
    close_mesh_noexcept(client_submesh);
    close_mesh_noexcept(parent_mesh);
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    fabric_enabled = false;
  } catch (...) {
    cleanup(client, no_server, client_submesh, parent_mesh, fabric_enabled);
    throw;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto args = parse_arguments(argc, argv);
    if (args.role == Role::Server) {
      run_server(args);
    } else {
      run_client(args);
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "unknown static intermesh smoke failure\n";
    return 1;
  }
  return 0;
}
