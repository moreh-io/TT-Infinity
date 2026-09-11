// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include <ttnn/operations/kvcache_manager/client/client_operations.hpp>
#include <ttnn/operations/kvcache_manager/common/config.hpp>
#include <ttnn/operations/kvcache_manager/common/fabric_helpers.hpp>
#include <ttnn/operations/kvcache_manager/common/protocol/bootstrap.hpp>
#include <ttnn/operations/kvcache_manager/common/protocol/packet.hpp>
#include <ttnn/operations/kvcache_manager/common/server_descriptor.hpp>

#include "kvcache_manager/client.hpp"

namespace tt::tt_metal::distributed {
class MeshBuffer;
}

namespace tt::tt_metal {
class Program;
}

namespace kvcache_manager {

namespace native_kvm = ttnn::operations::kvcache_manager;

class __attribute__((visibility("hidden"))) Client::Impl {
public:
  Impl(tt::tt_metal::distributed::MeshDevice *client_mesh,
       std::optional<tt::tt_fabric::FabricNodeId> server_node,
       const std::optional<tt::tt_metal::distributed::MeshCoordinate>
           &client_coord,
       bool static_intermesh_galaxy_to_t3k, uint32_t server_worker_group,
       uint32_t static_intermesh_lane);
  ~Impl() noexcept;

  [[nodiscard]] bool is_established() const;
  [[nodiscard]] std::vector<uint8_t> ping(std::span<const uint8_t> payload);
  void put(const std::string &key, const tt::tt_metal::Tensor &value_tensor,
           uint32_t value_len_bytes, bool sync_commit);
  void put_batch(const std::vector<std::string> &keys,
                 const std::vector<tt::tt_metal::Tensor> &value_tensors,
                 const std::vector<uint32_t> &value_len_bytes,
                 bool sync_commit);
  [[nodiscard]] PutProfile put_profile() const;
  void flush();
  [[nodiscard]] bool get(const std::string &key,
                         const tt::tt_metal::Tensor &out_tensor);
  [[nodiscard]] static std::vector<bool>
  get_parallel(const std::vector<Impl *> &clients,
               const std::vector<std::string> &keys,
               const tt::tt_metal::Tensor &out_tensor);
  [[nodiscard]] bool exists(const std::string &key);
  [[nodiscard]] bool remove(const std::string &key);
  void move_to_host(const std::string &key);
  void move_to_device(const std::string &key);
  void close();

private:
  enum class LifecycleState : uint8_t {
    Constructing,
    Established,
    Quarantined,
    Closed
  };

  void release_host_resources();
  struct QuarantineResources;
  void quarantine_resources() noexcept;
  void publish_quarantine() noexcept;
  void sync_operations_quarantine() noexcept;
  native_kvm::backend::ClientOperations &checked_operations() const;
  native_kvm::backend::ClientOperationsConfig make_operations_config() const;
  std::map<std::string, std::string>
  client_kernel_defines(native_kvm::KvcacheKernelTarget target) const;
  void append_server_connection_rt_args(tt::tt_metal::Program &program,
                                        std::vector<uint32_t> &rt_args) const;
  void reserve_bootstrap_l1();
  void resolve_client_connection();
  void send_establish_request();
  [[nodiscard]] bool
  launch_establish_client(const native_kvm::protocol::EstablishRequest &request,
                          uint64_t attempt_timeout_cycles);
  void apply_establish_response();
  void initialize_session();

  tt::tt_metal::distributed::MeshDevice *client_submesh_ = nullptr;
  bool static_intermesh_galaxy_to_t3k_ = false;
  uint32_t server_worker_group_ = 0;
  uint32_t static_intermesh_lane_ = 0;
  std::optional<tt::tt_fabric::FabricNodeId> deployment_server_node_;
  native_kvm::ServerDescriptor server_desc_{};

  static constexpr tt::tt_metal::CoreCoord kClientCore{0, 0};

  uint32_t client_l1_base_ = 0;

  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> establish_tx_buf_;
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>
      establish_response_buf_;
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> establish_sem_buf_;
  uint32_t establish_tx_base_ = 0;
  uint32_t establish_response_base_ = 0;
  uint32_t establish_sem_base_ = 0;

  uint32_t recv_buf_base_ = 0;

  std::shared_ptr<tt::tt_metal::distributed::MeshDevice> client_submesh_owner_;
  mutable std::mutex lifecycle_mutex_;
  native_kvm::backend::ClientOperationsHandle operations_;
  std::unique_ptr<QuarantineResources> quarantine_holder_;
  static std::atomic<QuarantineResources *> quarantine_head_;
  std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> establish_workload_;
  bool establish_may_have_started_ = false;

  uint32_t client_token_ = 0;
  uint32_t establish_nonce_low_ = 0;
  uint32_t establish_nonce_high_ = 0;

  tt::tt_metal::distributed::MeshCoordinate client_coord_{0, 0};

  uint32_t server_bootstrap_l1_addr_ = 0;
  uint32_t server_noc_x_ = 0;
  uint32_t server_noc_y_ = 0;
  uint32_t client_noc_x_ = 0;
  uint32_t client_noc_y_ = 0;
  uint32_t client_mesh_id_ = 0;
  uint32_t client_chip_id_ = 0;
  uint32_t route_word_to_server_ = 0;
  uint32_t establish_return_route_ = 0;
  uint32_t fabric_link_idx_ = 0;
  uint32_t establish_timeout_ms_ = 0;
  uint64_t establish_attempt_timeout_cycles_ = 0;
  uint32_t establish_attempts_started_ = 0;

  LifecycleState lifecycle_state_ = LifecycleState::Constructing;
  bool kvcache_profile_ = false;
  bool put_phase_profile_enabled_ = false;
  native_kvm::config::ProfileDetail profile_detail_{};
  uint32_t value_ring_depth_ = 0;

  tt::tt_metal::distributed::MeshCoordinateRange chip_range_{
      tt::tt_metal::distributed::MeshCoordinate(0, 0)};
};

} // namespace kvcache_manager
