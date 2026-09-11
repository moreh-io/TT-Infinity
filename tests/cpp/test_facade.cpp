// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "fake_backend.hpp"
#include "kvcache_manager/client.hpp"
#include "kvcache_manager/server.hpp"

namespace {

void check(bool condition, const char *expression, int line) {
  if (!condition) {
    throw std::runtime_error("line " + std::to_string(line) +
                             ": check failed: " + expression);
  }
}

#define CHECK(expression)                                                      \
  check(static_cast<bool>(expression), #expression, __LINE__)

using kvcache_manager::test::fake_backend_state;
using kvcache_manager::test::reset_fake_backend;

void test_type_contract() {
  static_assert(!std::is_copy_constructible_v<kvcache_manager::Server>);
  static_assert(!std::is_move_constructible_v<kvcache_manager::Server>);
  static_assert(!std::is_copy_constructible_v<kvcache_manager::Client>);
  static_assert(!std::is_move_constructible_v<kvcache_manager::Client>);
}

void test_client_operations_abi_guard() {
  reset_fake_backend();
  auto &state = fake_backend_state();
  state.client_operations_abi_version =
      ttnn::operations::kvcache_manager::backend::kClientOperationsAbiVersion +
      1;

  bool threw = false;
  try {
    kvcache_manager::Client client(
        reinterpret_cast<tt::tt_metal::distributed::MeshDevice *>(0x900),
        std::nullopt, std::nullopt, true);
  } catch (const std::runtime_error &error) {
    threw = std::string(error.what()).find("loaded ABI") != std::string::npos;
  }

  CHECK(threw);
  CHECK(state.client_create_count == 0);
}

void test_server_facade() {
  reset_fake_backend();
  auto *mesh =
      reinterpret_cast<tt::tt_metal::distributed::MeshDevice *>(0x1000);

  {
    kvcache_manager::Server server(mesh, true, 7, 3);
    auto &state = fake_backend_state();
    CHECK(state.server_create_count == 1);
    CHECK(state.server_mesh == mesh);
    CHECK(state.static_intermesh_t3k_server);
    CHECK(state.static_intermesh_remote_mesh_id == 7);
    CHECK(state.server_worker_group == 3);
    CHECK(server.is_ready());
    CHECK(server.moves_completed() == 0);

    const auto move_profile = server.move_profile();
    CHECK(move_profile.to_host_count == 0);
    CHECK(move_profile.to_host_device_pushes == 0);
    CHECK(move_profile.to_device_complete_ns == 0);

    const auto allocator_profile = server.allocator_profile();
    CHECK(allocator_profile.allocation_calls == 1);
    CHECK(allocator_profile.allocation_cycles == 13);
    CHECK(allocator_profile.clock_mhz == 14);

    server.close();
    CHECK(state.calls == std::vector<std::string>({
                             "server_runtime_launch",
                             "server_runtime_allocator_profile",
                             "server_runtime_drain",
                             "server_runtime_terminate",
                             "server_runtime_cancel",
                         }));
  }

  CHECK(fake_backend_state().server_destroy_count == 1);
}

void test_server_host_backing_policy() {
  using RequestKind =
      kvcache_manager::detail::server_runtime::ServerHostRequestKind;

  reset_fake_backend();
  auto *mesh =
      reinterpret_cast<tt::tt_metal::distributed::MeshDevice *>(0x1100);
  auto script = fake_backend_state().server_script;
  const std::vector<uint8_t> expected{3, 1, 4, 1, 5};

  {
    kvcache_manager::Server server(mesh);
    script->enqueue(
        {
            .sequence = 1,
            .kind = RequestKind::WriteThrough,
            .slot = 7,
            .value_len = static_cast<uint32_t>(expected.size()),
            .device_value_offset = 0,
            .poll_ns = 10,
        },
        expected);
    CHECK(script->wait_for_completed(1));

    script->enqueue({
        .sequence = 2,
        .kind = RequestKind::ServeGet,
        .slot = 7,
        .value_len = static_cast<uint32_t>(expected.size()),
        .device_value_offset = 0,
        .poll_ns = 10,
    });
    CHECK(script->wait_for_completed(2));
    CHECK(script->supplied_bytes(2) == expected);

    script->enqueue({
        .sequence = 3,
        .kind = RequestKind::RemoveBacking,
        .slot = 7,
    });
    CHECK(script->wait_for_completed(3));
    for (uint32_t attempt = 0; attempt < 1000 && server.moves_completed() != 3;
         ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(server.moves_completed() == 3);
    server.close();
  }

  CHECK(fake_backend_state().server_destroy_count == 1);
}

void test_server_launch_failure_releases_manager_registry() {
  reset_fake_backend();
  auto *mesh =
      reinterpret_cast<tt::tt_metal::distributed::MeshDevice *>(0x1200);
  fake_backend_state().fail_server_launch = true;

  bool threw = false;
  try {
    kvcache_manager::Server server(mesh);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  CHECK(threw);
  CHECK(fake_backend_state().server_create_count == 1);
  CHECK(fake_backend_state().server_destroy_count == 1);

  fake_backend_state().fail_server_launch = false;
  {
    kvcache_manager::Server server(mesh);
    server.close();
  }
  CHECK(fake_backend_state().server_create_count == 2);
  CHECK(fake_backend_state().server_destroy_count == 2);
}

void test_server_close_retries_terminate() {
  reset_fake_backend();
  auto *mesh =
      reinterpret_cast<tt::tt_metal::distributed::MeshDevice *>(0x1300);
  fake_backend_state().server_terminate_failures = 1;

  {
    kvcache_manager::Server server(mesh);
    bool threw = false;
    try {
      server.close();
    } catch (const std::runtime_error &) {
      threw = true;
    }
    CHECK(threw);
    CHECK(!server.is_ready());
    CHECK(fake_backend_state().server_destroy_count == 0);

    server.close();
  }
  CHECK(fake_backend_state().server_destroy_count == 1);
  CHECK(fake_backend_state().server_unterminated_destroy_count == 0);
}

void test_server_worker_failure_quarantines_runtime() {
  using RequestKind =
      kvcache_manager::detail::server_runtime::ServerHostRequestKind;

  reset_fake_backend();
  auto *mesh =
      reinterpret_cast<tt::tt_metal::distributed::MeshDevice *>(0x1400);
  auto script = fake_backend_state().server_script;

  {
    kvcache_manager::Server server(mesh);
    script->enqueue({
        .sequence = 1,
        .kind = RequestKind::ServeGet,
        .slot = 99,
        .value_len = 16,
    });
    for (uint32_t attempt = 0; attempt < 1000 && server.is_ready(); ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(!server.is_ready());

    bool threw = false;
    try {
      server.close();
    } catch (const std::runtime_error &) {
      threw = true;
    }
    CHECK(threw);
    server.close();
  }

  CHECK(fake_backend_state().server_destroy_count == 1);
  CHECK(fake_backend_state().server_unterminated_destroy_count == 1);
}

void test_client_facade() {
  reset_fake_backend();
  auto *mesh =
      reinterpret_cast<tt::tt_metal::distributed::MeshDevice *>(0x2000);
  const tt::tt_fabric::FabricNodeId server_node(tt::tt_fabric::MeshId{21}, 22);
  const tt::tt_metal::distributed::MeshCoordinate coordinate(2, 3);
  tt::tt_metal::Tensor tensor{};
  std::vector<tt::tt_metal::Tensor> tensors;
  tensors.reserve(2);
  tensors.emplace_back();
  tensors.emplace_back();

  {
    kvcache_manager::Client client(mesh, server_node, coordinate, false, 4, 1);
    auto &state = fake_backend_state();
    CHECK(state.client_create_count == 1);
    CHECK(state.client_mesh == mesh);
    CHECK(state.client_server_node == server_node);
    CHECK(state.client_coord == coordinate);
    CHECK(!state.static_intermesh_galaxy_to_t3k);
    CHECK(state.client_server_worker_group == 4);
    CHECK(state.static_intermesh_lane == 1);
    CHECK(client.is_established());

    const std::array<uint8_t, 3> payload{4, 5, 6};
    CHECK(client.ping(payload) == std::vector<uint8_t>({4, 5, 6}));

    client.put("put", tensor, 64, true);
    CHECK(state.key == "put");
    CHECK(state.tensor == &tensor);
    CHECK(state.value_len_bytes == 64);
    CHECK(state.sync_commit);

    const std::vector<std::string> keys{"a", "b"};
    const std::vector<uint32_t> lengths{32, 64};
    client.put_batch(keys, tensors, lengths, false);
    CHECK(state.keys == keys);
    CHECK(state.tensors == std::vector<const tt::tt_metal::Tensor *>(
                               {&tensors[0], &tensors[1]}));
    CHECK(state.value_lengths == lengths);
    CHECK(!state.sync_commit);

    const auto put_profile = client.put_profile();
    CHECK(put_profile.count == 1);
    CHECK(put_profile.commit_wait_ns == 6);
    CHECK(put_profile.total_ns == 7);

    client.flush();
    CHECK(client.get("get", tensor));
    CHECK(state.key == "get");
    CHECK(kvcache_manager::Client::get_parallel(
              {&client}, {"parallel"}, tensor) == std::vector<bool>({true}));
    CHECK(state.keys == std::vector<std::string>({"parallel"}));
    CHECK(!client.exists("missing"));
    CHECK(state.key == "missing");
    CHECK(client.remove("remove"));
    CHECK(state.key == "remove");
    client.move_to_host("host");
    CHECK(state.key == "host");
    client.move_to_device("device");
    CHECK(state.key == "device");
    client.close();

    CHECK(state.calls == std::vector<std::string>({
                             "client_is_established",
                             "client_ping",
                             "client_put",
                             "client_put_batch",
                             "client_put_profile",
                             "client_flush",
                             "client_get",
                             "client_get_parallel",
                             "client_exists",
                             "client_remove",
                             "client_move_to_host",
                             "client_move_to_device",
                             "client_close",
                         }));
  }

  CHECK(fake_backend_state().client_destroy_count == 1);
}

} // namespace

int main() {
  try {
    test_type_contract();
    test_client_operations_abi_guard();
    test_server_facade();
    test_server_host_backing_policy();
    test_server_launch_failure_releases_manager_registry();
    test_server_close_retries_terminate();
    test_server_worker_failure_quarantines_runtime();
    test_client_facade();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  std::cout << "C++ facade tests passed\n";
  return 0;
}
