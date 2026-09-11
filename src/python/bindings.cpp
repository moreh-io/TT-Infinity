// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "kvcache_manager/client.hpp"
#include "kvcache_manager/profiles.hpp"
#include "kvcache_manager/server.hpp"

#if NB_VERSION_MAJOR != 2 || NB_VERSION_MINOR != 12 || NB_VERSION_PATCH != 0
#error "kvcache-manager must use the nanobind 2.12.0 ABI used by TTNN"
#endif

namespace nb = nanobind;

namespace kvcache_manager {

struct ServerTestAccess {
  static std::unique_ptr<Server> create_delay_first_establish_response(
      tt::tt_metal::distributed::MeshDevice *server_submesh) {
    return std::unique_ptr<Server>(
        new Server(server_submesh, Server::TestOptions{
                                       .delay_first_establish_response = true,
                                   }));
  }

  static std::unique_ptr<Server> create_drop_first_establish_response(
      tt::tt_metal::distributed::MeshDevice *server_submesh) {
    return std::unique_ptr<Server>(
        new Server(server_submesh, Server::TestOptions{
                                       .drop_first_establish_response = true,
                                   }));
  }

  static std::unique_ptr<Server> create_delay_first_put_commit(
      tt::tt_metal::distributed::MeshDevice *server_submesh) {
    return std::unique_ptr<Server>(
        new Server(server_submesh, Server::TestOptions{
                                       .delay_first_put_commit = true,
                                   }));
  }

  static std::pair<uint32_t, uint32_t> put_drain_progress(Server &server) {
    return server.test_put_drain_progress();
  }
};

} // namespace kvcache_manager

namespace {

using Client = kvcache_manager::Client;
using FabricNodeId = tt::tt_fabric::FabricNodeId;
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using Server = kvcache_manager::Server;
using Tensor = tt::tt_metal::Tensor;

std::string to_key(nb::handle value) {
  if (nb::isinstance<nb::str>(value)) {
    return nb::cast<std::string>(value);
  }
  if (nb::isinstance<nb::bytes>(value)) {
    const auto bytes = nb::cast<nb::bytes>(value);
    return std::string(bytes.c_str(), bytes.size());
  }
  throw nb::type_error("key must be str or bytes");
}

nb::dict to_dict(const kvcache_manager::MoveProfile &profile) {
  nb::dict result;
  result["to_host_count"] = profile.to_host_count;
  result["to_host_worker_ns"] = profile.to_host_worker_ns;
  result["to_host_complete_ns"] = profile.to_host_complete_ns;
  result["to_host_read_calls"] = profile.to_host_read_calls;
  result["to_host_read_bytes"] = profile.to_host_read_bytes;
  result["to_host_read_passes"] = profile.to_host_read_passes;
  result["to_host_read_retries"] = profile.to_host_read_retries;
  result["to_host_read_io_ns"] = profile.to_host_read_io_ns;
  result["to_host_scatter_ns"] = profile.to_host_scatter_ns;
  result["to_host_device_pushes"] = profile.to_host_device_pushes;
  result["to_device_count"] = profile.to_device_count;
  result["to_device_worker_ns"] = profile.to_device_worker_ns;
  result["to_device_complete_ns"] = profile.to_device_complete_ns;
  return result;
}

nb::dict to_dict(const kvcache_manager::AllocatorProfile &profile) {
  nb::dict result;
  result["allocation_calls"] = profile.allocation_calls;
  result["allocation_successes"] = profile.allocation_successes;
  result["allocation_failures"] = profile.allocation_failures;
  result["victim_scans"] = profile.victim_scans;
  result["victims_evicted"] = profile.victims_evicted;
  result["lease_calls"] = profile.lease_calls;
  result["lease_successes"] = profile.lease_successes;
  result["lease_failures"] = profile.lease_failures;
  result["lease_victim_scans"] = profile.lease_victim_scans;
  result["lease_victims_evicted"] = profile.lease_victims_evicted;
  result["scan_cycles"] = profile.scan_cycles;
  result["lease_scan_cycles"] = profile.lease_scan_cycles;
  result["allocation_cycles"] = profile.allocation_cycles;
  result["clock_mhz"] = profile.clock_mhz;
  return result;
}

nb::dict to_dict(const kvcache_manager::PutProfile &profile) {
  nb::dict result;
  result["count"] = profile.count;
  result["prepare_ns"] = profile.prepare_ns;
  result["program_ns"] = profile.program_ns;
  result["enqueue_ns"] = profile.enqueue_ns;
  result["landed_wait_ns"] = profile.landed_wait_ns;
  result["commit_wait_ns"] = profile.commit_wait_ns;
  result["total_ns"] = profile.total_ns;
  return result;
}

void require_ttnn_types() {
  nb::module_::import_("ttnn");

  if (!nb::type<MeshDevice>().is_valid() ||
      !nb::type<MeshCoordinate>().is_valid() ||
      !nb::type<FabricNodeId>().is_valid() || !nb::type<Tensor>().is_valid()) {
    throw nb::import_error(
        "kvcache_manager._native requires a matching TTNN nanobind runtime");
  }
}

void bind_server(nb::module_ &module) {
  nb::class_<Server>(module, "Server")
      .def(nb::init<MeshDevice *, bool, uint32_t, uint32_t>(),
           nb::arg("server_submesh"),
           nb::arg("static_intermesh_t3k_server") = false,
           nb::arg("static_intermesh_remote_mesh_id") = 0,
           nb::arg("worker_group") = 0, nb::keep_alive<1, 2>())
      .def_static("_test_create_drop_first_establish_response",
                  &kvcache_manager::ServerTestAccess::
                      create_drop_first_establish_response,
                  nb::arg("server_submesh").noconvert(), nb::keep_alive<0, 1>())
      .def_static("_test_create_delay_first_establish_response",
                  &kvcache_manager::ServerTestAccess::
                      create_delay_first_establish_response,
                  nb::arg("server_submesh").noconvert(), nb::keep_alive<0, 1>())
      .def_static(
          "_test_create_delay_first_put_commit",
          &kvcache_manager::ServerTestAccess::create_delay_first_put_commit,
          nb::arg("server_submesh").noconvert(), nb::keep_alive<0, 1>())
      .def("_test_put_drain_progress",
           &kvcache_manager::ServerTestAccess::put_drain_progress)
      .def("is_ready", &Server::is_ready)
      .def("close", &Server::close)
      .def("moves_completed", &Server::moves_completed)
      .def("move_profile",
           [](const Server &self) { return to_dict(self.move_profile()); })
      .def("allocator_profile", [](const Server &self) {
        return to_dict(self.allocator_profile());
      });
}

void bind_client(nb::module_ &module) {
  nb::class_<Client>(module, "Client")
      .def(
          "__init__",
          [](Client *self, MeshDevice *client_mesh,
             std::optional<FabricNodeId> server_node,
             const std::optional<MeshCoordinate> &client_coord,
             bool static_intermesh_galaxy_to_t3k, uint32_t server_worker_group,
             uint32_t static_intermesh_lane) {
            new (self) Client(client_mesh, std::move(server_node), client_coord,
                              static_intermesh_galaxy_to_t3k,
                              server_worker_group, static_intermesh_lane);
          },
          nb::arg("client_mesh"), nb::arg("server_node") = nb::none(),
          nb::arg("client_coord") = nb::none(),
          nb::arg("static_intermesh_galaxy_to_t3k") = false,
          nb::arg("server_worker_group") = 0,
          nb::arg("static_intermesh_lane") = 0, nb::keep_alive<1, 2>())
      .def("is_established", &Client::is_established)
      .def("close", &Client::close)
      .def(
          "ping",
          [](Client &self, const nb::bytes &payload) {
            const auto *data =
                reinterpret_cast<const uint8_t *>(payload.c_str());
            const auto result =
                self.ping(std::span<const uint8_t>(data, payload.size()));
            return nb::bytes(reinterpret_cast<const char *>(result.data()),
                             result.size());
          },
          nb::arg("payload"))
      .def(
          "put",
          [](Client &self, nb::handle key, const Tensor &value_tensor,
             uint32_t value_len_bytes, bool sync_commit) {
            self.put(to_key(key), value_tensor, value_len_bytes, sync_commit);
          },
          nb::arg("key"), nb::arg("value_tensor"), nb::arg("value_len_bytes"),
          nb::arg("sync_commit") = false)
      .def(
          "put_batch",
          [](Client &self, const nb::list &keys,
             const std::vector<Tensor> &value_tensors,
             const std::vector<uint32_t> &value_len_bytes, bool sync_commit) {
            std::vector<std::string> normalized_keys;
            normalized_keys.reserve(keys.size());
            for (const nb::handle key : keys) {
              normalized_keys.push_back(to_key(key));
            }
            self.put_batch(normalized_keys, value_tensors, value_len_bytes,
                           sync_commit);
          },
          nb::arg("keys"), nb::arg("value_tensors"), nb::arg("value_len_bytes"),
          nb::arg("sync_commit") = false)
      .def("put_profile",
           [](const Client &self) { return to_dict(self.put_profile()); })
      .def("flush", &Client::flush)
      .def(
          "get",
          [](Client &self, nb::handle key, const Tensor &out_tensor) {
            return self.get(to_key(key), out_tensor);
          },
          nb::arg("key"), nb::arg("out_tensor"))
      .def_static(
          "get_parallel",
          [](const nb::list &clients, const nb::list &keys,
             const Tensor &out_tensor) {
            std::vector<Client *> normalized_clients;
            std::vector<std::string> normalized_keys;
            normalized_clients.reserve(clients.size());
            normalized_keys.reserve(keys.size());
            for (const nb::handle client : clients) {
              normalized_clients.push_back(nb::cast<Client *>(client));
            }
            for (const nb::handle key : keys) {
              normalized_keys.push_back(to_key(key));
            }
            nb::gil_scoped_release release;
            return Client::get_parallel(normalized_clients, normalized_keys,
                                        out_tensor);
          },
          nb::arg("clients"), nb::arg("keys"), nb::arg("out_tensor"))
      .def(
          "exists",
          [](Client &self, nb::handle key) { return self.exists(to_key(key)); },
          nb::arg("key"))
      .def(
          "delete",
          [](Client &self, nb::handle key) { return self.remove(to_key(key)); },
          nb::arg("key"))
      .def(
          "move_to_host",
          [](Client &self, nb::handle key) { self.move_to_host(to_key(key)); },
          nb::arg("key"))
      .def(
          "move_to_device",
          [](Client &self, nb::handle key) {
            self.move_to_device(to_key(key));
          },
          nb::arg("key"));
}

} // namespace

NB_MODULE(_native, module) {
  require_ttnn_types();
  bind_server(module);
  bind_client(module);
}
