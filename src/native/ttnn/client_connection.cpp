// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#include "client_impl.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/operations/kvcache_manager/common/config.hpp"
#include "ttnn/operations/kvcache_manager/common/fabric_helpers.hpp"
#include "ttnn/operations/kvcache_manager/common/protocol/layout.hpp"

#include "bootstrap_contract.hpp"

namespace kvcache_manager {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using namespace native_kvm;

namespace {

const std::string kEstablishKernelPath =
    "ttnn/cpp/ttnn/operations/kvcache_manager/client/device/kernels/dataflow/"
    "establish_kernel.cpp";
constexpr uint32_t kEstablishAttemptTimeoutMs = 250;
constexpr auto kEstablishHostWatchdogGrace = std::chrono::seconds(2);
constexpr auto kEstablishHostPollInterval = std::chrono::microseconds(100);

const char *establish_status_name(protocol::EstablishResponseStatus status) {
  switch (status) {
  case protocol::EstablishResponseStatus::Accepted:
    return "accepted";
  case protocol::EstablishResponseStatus::BootstrapVersionMismatch:
    return "bootstrap-version-mismatch";
  case protocol::EstablishResponseStatus::WireVersionMismatch:
    return "wire-version-mismatch";
  case protocol::EstablishResponseStatus::CapabilityMismatch:
    return "capability-mismatch";
  case protocol::EstablishResponseStatus::InvalidRequest:
    return "invalid-request";
  case protocol::EstablishResponseStatus::InvalidPeer:
    return "invalid-peer";
  case protocol::EstablishResponseStatus::DramGeometryMismatch:
    return "dram-geometry-mismatch";
  case protocol::EstablishResponseStatus::PeerBusy:
    return "peer-busy-reset-required";
  }
  return "unknown";
}

} // namespace

void Client::Impl::resolve_client_connection() {
  constexpr uint32_t kMaxNocCoordinate = (1u << 6) - 1;

  IDevice *client_device = client_submesh_->get_device(client_coord_);
  TT_FATAL(client_device != nullptr,
           "Client has no device at the selected coordinate");
  const auto worker_grid = client_submesh_->compute_with_storage_grid_size();
  TT_FATAL(server_worker_group_ < worker_grid.x && worker_grid.y >= 3,
           "KVM Server worker group {} requires logical cores ({},0..2), "
           "outside worker grid ({},{})",
           server_worker_group_, server_worker_group_, worker_grid.x,
           worker_grid.y);
  const CoreCoord server_core{server_worker_group_, 0};
  const CoreCoord server_virtual = detail::require_bootstrap_worker_noc(
      client_device, server_core, "kvcache_manager Client Server target");
  const CoreCoord client_virtual = detail::require_bootstrap_worker_noc(
      client_device, kClientCore, "kvcache_manager Client");
  server_noc_x_ = static_cast<uint32_t>(server_virtual.x);
  server_noc_y_ = static_cast<uint32_t>(server_virtual.y);
  TT_FATAL(server_noc_x_ <= kMaxNocCoordinate &&
               server_noc_y_ <= kMaxNocCoordinate,
           "derived Server bootstrap NoC coordinate ({}, {}) exceeds the "
           "6-bit Wormhole encoding",
           server_noc_x_, server_noc_y_);

  const uint32_t server_l1_base = static_cast<uint32_t>(
      client_submesh_->allocator()->get_base_allocator_addr(HalMemType::L1));
  server_bootstrap_l1_addr_ =
      server_l1_base + protocol::kBootstrapRequestBaseOffset;
  TT_FATAL(server_bootstrap_l1_addr_ != 0 &&
               server_bootstrap_l1_addr_ % 16 == 0,
           "derived Server bootstrap L1 address {} must be nonzero and "
           "16-byte aligned",
           server_bootstrap_l1_addr_);

  client_noc_x_ = static_cast<uint32_t>(client_virtual.x);
  client_noc_y_ = static_cast<uint32_t>(client_virtual.y);
  const auto client_node = client_submesh_->get_fabric_node_id(client_coord_);
  client_mesh_id_ = *client_node.mesh_id;
  client_chip_id_ = client_node.chip_id;
  TT_FATAL(
      client_chip_id_ < protocol::kBootstrapPeerSlots,
      "client chip_id {} is at or beyond the {}-slot Server bootstrap table",
      client_chip_id_, protocol::kBootstrapPeerSlots);

  if (static_intermesh_galaxy_to_t3k_) {
    TT_FATAL(!deployment_server_node_.has_value(),
             "static Galaxy-to-T3K mode must not receive a Server node");
    TT_FATAL(is_kvcache_2d_fabric(),
             "static Galaxy-to-T3K mode requires FABRIC_2D");
    const auto gateway_node =
        tt::tt_fabric::get_static_intermesh_endpoint_fabric_node_id();
    const uint32_t static_link_count =
        tt::tt_fabric::get_static_intermesh_endpoint_link_count();
    TT_FATAL(static_intermesh_lane_ < static_link_count,
             "static inter-mesh lane {} is outside configured link count {}",
             static_intermesh_lane_, static_link_count);
    TT_FATAL(client_node.mesh_id == gateway_node.mesh_id,
             "the Client (mesh {}) and configured static gateway (mesh {}) "
             "must be in the same local mesh",
             *client_node.mesh_id, *gateway_node.mesh_id);
    if (client_node != gateway_node) {
      TT_FATAL(is_kvcache_wire_reachable(client_node, gateway_node),
               "the Client (mesh {} chip {}) cannot reach the configured "
               "static gateway chip {}",
               *client_node.mesh_id, client_node.chip_id, gateway_node.chip_id);
      const auto gateway_links =
          kvcache_wire_link_indices(client_node, gateway_node);
      TT_FATAL(!gateway_links.empty(),
               "the Client (mesh {} chip {}) has no forwarding link plane "
               "toward static gateway chip {}",
               *client_node.mesh_id, client_node.chip_id, gateway_node.chip_id);
      fabric_link_idx_ =
          gateway_links[static_intermesh_lane_ % gateway_links.size()];
    }
    route_word_to_server_ = kvcache_unicast_route_word(gateway_node);

    const auto gateway_coord =
        tt::tt_fabric::get_fabric_mesh_coordinate(gateway_node);
    const auto client_fabric_coord =
        tt::tt_fabric::get_fabric_mesh_coordinate(client_node);
    const int32_t gateway_row = static_cast<int32_t>(gateway_coord[0]);
    const int32_t gateway_column = static_cast<int32_t>(gateway_coord[1]);
    const int32_t client_row = static_cast<int32_t>(client_fabric_coord[0]);
    const int32_t client_column = static_cast<int32_t>(client_fabric_coord[1]);
    const int32_t row_delta = client_row - gateway_row;
    const int32_t column_delta = client_column - gateway_column;
    const uint32_t north_south_hops =
        static_cast<uint32_t>(row_delta < 0 ? -row_delta : row_delta);
    const uint32_t east_west_hops =
        static_cast<uint32_t>(column_delta < 0 ? -column_delta : column_delta);
    TT_FATAL(client_node.chip_id <= 0x1Fu && *client_node.mesh_id <= 0xFFFFu &&
                 north_south_hops <= 0xFu && east_west_hops <= 0xFu,
             "static Galaxy return route does not fit the ESTABLISH encoding "
             "(mesh {}, chip {}, ns hops {}, ew hops {})",
             *client_node.mesh_id, client_node.chip_id, north_south_hops,
             east_west_hops);
    establish_return_route_ = protocol::pack_return_route(
        client_node.chip_id, *client_node.mesh_id, north_south_hops,
        east_west_hops, row_delta > 0 ? 1u : 0u, column_delta > 0 ? 1u : 0u);
    log_info(tt::LogOp,
             "kvcache_manager Client: static Galaxy-to-T3K ESTABLISH through "
             "gateway chip {} to Server worker group {}; return route to "
             "client chip {} is ns {} dir {}, ew {} dir {}",
             gateway_node.chip_id, server_worker_group_, client_node.chip_id,
             north_south_hops, row_delta > 0 ? "south" : "north",
             east_west_hops, column_delta > 0 ? "east" : "west");
    return;
  }

  TT_FATAL(deployment_server_node_.has_value(),
           "standard Fabric mode requires a Server FabricNodeId");
  const auto &server_node = *deployment_server_node_;
  TT_FATAL(client_node.mesh_id == server_node.mesh_id,
           "client/server mesh IDs differ (client {}, server {}); cross-mesh "
           "sessions require static Galaxy-to-T3K mode",
           *client_node.mesh_id, *server_node.mesh_id);
  TT_FATAL(client_node != server_node,
           "client and server must resolve to different Fabric nodes (both "
           "are mesh {} chip {})",
           *client_node.mesh_id, client_node.chip_id);
  TT_FATAL(is_kvcache_wire_reachable(client_node, server_node),
           "client (mesh {} chip {}) and server (mesh {} chip {}) are not "
           "Fabric-reachable",
           *client_node.mesh_id, client_node.chip_id, *server_node.mesh_id,
           server_node.chip_id);
  route_word_to_server_ = kvcache_unicast_route_word(server_node);
  fabric_link_idx_ = kvcache_wire_link_index(client_node, server_node);
}

std::map<std::string, std::string>
Client::Impl::client_kernel_defines(KvcacheKernelTarget target) const {
  const auto route_mode = static_intermesh_galaxy_to_t3k_
                              ? KvcacheRouteMode::StaticIntermeshGalaxyToT3k
                              : KvcacheRouteMode::Standard;
  return kvcache_kernel_defines(protocol::kRingNumSlotsDefault,
                                value_ring_depth_, target, profile_detail_,
                                std::nullopt, route_mode);
}

void Client::Impl::append_server_connection_rt_args(
    Program &program, std::vector<uint32_t> &rt_args) const {
  const auto client_node = client_submesh_->get_fabric_node_id(client_coord_);
  if (static_intermesh_galaxy_to_t3k_) {
    const auto gateway_node =
        tt::tt_fabric::get_static_intermesh_endpoint_fabric_node_id();
    if (client_node == gateway_node) {
      tt::tt_fabric::append_static_intermesh_lane_connection_rt_args(
          static_intermesh_lane_, program, kClientCore, rt_args);
    } else {
      tt::tt_fabric::append_fabric_connection_rt_args(client_node, gateway_node,
                                                      fabric_link_idx_, program,
                                                      kClientCore, rt_args);
    }
    return;
  }

  TT_FATAL(deployment_server_node_.has_value(),
           "standard Fabric mode has no Server node");
  tt::tt_fabric::append_fabric_connection_rt_args(
      client_node, *deployment_server_node_, fabric_link_idx_, program,
      kClientCore, rt_args);
}

void Client::Impl::reserve_bootstrap_l1() {
  auto allocate = [this](uint32_t bytes) {
    auto shard =
        ShardSpecBuffer(CoreRangeSet(CoreRange(kClientCore)),
                        /*shard_shape*/ {1, 1}, ShardOrientation::ROW_MAJOR,
                        /*page_shape*/ {1, 1},
                        /*tensor2d_shape_in_pages*/ {1, 1});
    DeviceLocalBufferConfig local{
        .page_size = bytes,
        .buffer_type = BufferType::L1,
        .sharding_args =
            BufferShardingArgs(shard, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
        .sub_device_id = std::nullopt,
    };
    ReplicatedBufferConfig global{.size = bytes};
    return MeshBuffer::create(global, local, client_submesh_);
  };

  establish_tx_buf_ = allocate(sizeof(protocol::EstablishRequest));
  establish_response_buf_ = allocate(sizeof(protocol::EstablishResponse));
  establish_sem_buf_ = allocate(16);
  establish_tx_base_ = static_cast<uint32_t>(establish_tx_buf_->address());
  establish_response_base_ =
      static_cast<uint32_t>(establish_response_buf_->address());
  establish_sem_base_ = static_cast<uint32_t>(establish_sem_buf_->address());
  TT_FATAL(establish_tx_base_ % 16 == 0 && establish_response_base_ % 16 == 0 &&
               establish_sem_base_ % 16 == 0,
           "ESTABLISH L1 buffers must be 16-byte aligned (tx={}, response={}, "
           "sem={})",
           establish_tx_base_, establish_response_base_, establish_sem_base_);
}

void Client::Impl::send_establish_request() {
  IDevice *client_device = client_submesh_->get_device(client_coord_);
  TT_FATAL(client_device != nullptr,
           "Client ESTABLISH has no device at the selected coordinate");
  const CoreCoord dram_grid = client_device->dram_grid_size();
  const protocol::EstablishRequest request{
      .magic = protocol::kBootstrapMagic,
      .bootstrap_version = protocol::kBootstrapProtocolVersion,
      .wire_layout_version = protocol::kWireLayoutVersion,
      .required_capabilities = protocol::kRequiredCapabilities,
      .client_mesh_id = client_mesh_id_,
      .client_chip_id = client_chip_id_,
      .client_noc_x = client_noc_x_,
      .client_noc_y = client_noc_y_,
      .response_addr = establish_response_base_,
      .response_sem_addr = establish_sem_base_,
      .nonce_low = establish_nonce_low_,
      .nonce_high = establish_nonce_high_,
      .client_l1_base = client_l1_base_,
      .recv_buffer_addr = recv_buf_base_,
      .client_value_ring_depth = value_ring_depth_,
      .return_route = establish_return_route_,
      .num_dram_channels =
          static_cast<uint32_t>(client_device->num_dram_channels()),
      .dram_grid_x = static_cast<uint32_t>(dram_grid.x),
      .dram_grid_y = static_cast<uint32_t>(dram_grid.y),
      .reserved_tail = 0,
  };

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(establish_timeout_ms_);
  while (std::chrono::steady_clock::now() < deadline) {
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
    const uint64_t remaining_cycles = std::max<uint64_t>(
        1, static_cast<uint64_t>(remaining.count()) *
               static_cast<uint64_t>(client_submesh_->get_clock_rate_mhz()) *
               1'000);
    if (launch_establish_client(
            request,
            std::min(establish_attempt_timeout_cycles_, remaining_cycles))) {
      return;
    }
  }

  TT_THROW("kvcache_manager ESTABLISH timed out after {} ms; the outcome is "
           "unknown and the device must be reset",
           establish_timeout_ms_);
}

bool Client::Impl::launch_establish_client(
    const protocol::EstablishRequest &request,
    uint64_t attempt_timeout_cycles) {
  Program program = CreateProgram();
  const CoreRange core_range(kClientCore, kClientCore);
  const KernelHandle kernel_id = CreateKernel(
      program, kEstablishKernelPath, core_range,
      DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                         .noc = NOC::RISCV_1_default,
                         .defines = client_kernel_defines(
                             KvcacheKernelTarget::ClientUnprofiled)});

  std::vector<uint32_t> rt_args = {
      establish_tx_base_,
      establish_sem_base_,
      server_bootstrap_l1_addr_,
      server_noc_x_,
      server_noc_y_,
      route_word_to_server_,
      client_chip_id_,
      static_cast<uint32_t>(attempt_timeout_cycles),
      static_cast<uint32_t>(attempt_timeout_cycles >> 32),
      establish_attempts_started_ == 0 ? 1u : 0u,
  };
  const auto *request_words = reinterpret_cast<const uint32_t *>(&request);
  rt_args.insert(rt_args.end(), request_words,
                 request_words + sizeof(request) / sizeof(uint32_t));
  append_server_connection_rt_args(program, rt_args);
  SetRuntimeArgs(program, kernel_id, kClientCore, rt_args);

  tt::tt_metal::detail::CompileProgram(client_submesh_, program);
  establish_workload_ = std::make_unique<MeshWorkload>();
  establish_workload_->add_program(chip_range_, std::move(program));

  auto &cq = client_submesh_->mesh_command_queue();
  establish_may_have_started_ = true;
  ++establish_attempts_started_;
  EnqueueMeshWorkload(cq, *establish_workload_, /*blocking=*/false);
  auto completion_event = cq.enqueue_record_event_to_host();
  const auto host_deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(kEstablishAttemptTimeoutMs) +
      kEstablishHostWatchdogGrace;
  while (!EventQuery(completion_event)) {
    TT_FATAL(std::chrono::steady_clock::now() < host_deadline,
             "kvcache_manager ESTABLISH workload exceeded its device deadline "
             "plus watchdog grace; the outcome is unknown and the device must "
             "be reset");
    std::this_thread::sleep_for(kEstablishHostPollInterval);
  }

  std::vector<uint8_t> status_bytes;
  ReadShard(cq, status_bytes, establish_sem_buf_, client_coord_,
            /*blocking=*/true);
  TT_FATAL(status_bytes.size() >= 2 * sizeof(uint32_t),
           "ESTABLISH status buffer is too small");
  uint32_t client_status = 0;
  std::memcpy(&client_status, status_bytes.data() + sizeof(uint32_t),
              sizeof(client_status));
  if (client_status ==
      static_cast<uint32_t>(protocol::EstablishClientStatus::Responded)) {
    return true;
  }
  TT_FATAL(client_status ==
               static_cast<uint32_t>(protocol::EstablishClientStatus::TimedOut),
           "ESTABLISH returned invalid client status {}", client_status);
  return false;
}

void Client::Impl::apply_establish_response() {
  std::vector<uint8_t> response_bytes;
  ReadShard(client_submesh_->mesh_command_queue(), response_bytes,
            establish_response_buf_, client_coord_,
            /*blocking=*/true);
  TT_FATAL(response_bytes.size() >= sizeof(protocol::EstablishResponse),
           "ESTABLISH response is too small: expected {}, got {}",
           sizeof(protocol::EstablishResponse), response_bytes.size());
  protocol::EstablishResponse response{};
  std::memcpy(&response, response_bytes.data(), sizeof(response));
  TT_FATAL(response.magic == protocol::kBootstrapMagic,
           "ESTABLISH response magic mismatch");
  TT_FATAL(response.bootstrap_version == protocol::kBootstrapProtocolVersion,
           "ESTABLISH response bootstrap version {} != client {}",
           response.bootstrap_version, protocol::kBootstrapProtocolVersion);
  TT_FATAL(response.nonce_low == establish_nonce_low_ &&
               response.nonce_high == establish_nonce_high_,
           "ESTABLISH response nonce does not match this Client request");
  const auto status =
      static_cast<protocol::EstablishResponseStatus>(response.status);
  if (status != protocol::EstablishResponseStatus::Accepted) {
    if (establish_attempts_started_ == 1) {
      establish_may_have_started_ = false;
    }
    TT_THROW("Server rejected ESTABLISH: {} ({})",
             establish_status_name(status), response.status);
  }
  TT_FATAL((response.capabilities & protocol::kRequiredCapabilities) ==
               protocol::kRequiredCapabilities,
           "Server ESTABLISH capabilities {:#x} do not cover required mask "
           "{:#x}",
           response.capabilities, protocol::kRequiredCapabilities);
  TT_FATAL(protocol::client_generation_from_token(response.client_token) != 0,
           "Server ESTABLISH returned generation-zero token {:#x}",
           response.client_token);
  TT_FATAL(
      protocol::client_id_from_token(response.client_token) == client_chip_id_,
      "Server ESTABLISH token ID {} does not match bootstrap peer chip {}",
      protocol::client_id_from_token(response.client_token), client_chip_id_);

  server_desc_ = response.descriptor;
  validate_server_descriptor(server_desc_);
  const auto descriptor_server_node = tt::tt_fabric::FabricNodeId(
      tt::tt_fabric::MeshId{server_desc_.fabric_mesh_id},
      server_desc_.fabric_chip_id);
  if (deployment_server_node_.has_value()) {
    TT_FATAL(descriptor_server_node == *deployment_server_node_,
             "ESTABLISH descriptor node (mesh {}, chip {}) differs from "
             "deployment node (mesh {}, chip {})",
             *descriptor_server_node.mesh_id, descriptor_server_node.chip_id,
             *deployment_server_node_->mesh_id,
             deployment_server_node_->chip_id);
  }
  TT_FATAL(server_desc_.l1_base + protocol::kBootstrapRequestBaseOffset ==
               server_bootstrap_l1_addr_,
           "Server L1 base {} violates the bootstrap ABI address {}",
           server_desc_.l1_base, server_bootstrap_l1_addr_);
  TT_FATAL(server_desc_.noc_x == server_noc_x_ &&
               server_desc_.noc_y == server_noc_y_,
           "Server NoC coordinate ({}, {}) violates the bootstrap ABI "
           "coordinate ({}, {})",
           server_desc_.noc_x, server_desc_.noc_y, server_noc_x_,
           server_noc_y_);
  TT_FATAL(server_desc_.value_ring_depth == value_ring_depth_,
           "Server value-ring depth {} differs from Client proposal {}",
           server_desc_.value_ring_depth, value_ring_depth_);

  client_token_ = response.client_token;
  backend::client_operations_configure_session(
      *operations_,
      {.server_descriptor = server_desc_, .client_token = client_token_});
  backend::client_operations_mark_established(*operations_);
  establish_may_have_started_ = false;
}

} // namespace kvcache_manager
