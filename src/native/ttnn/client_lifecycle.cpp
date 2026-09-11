// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#include "client_impl.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "ttnn/operations/kvcache_manager/common/config.hpp"
#include "ttnn/operations/kvcache_manager/common/fabric_helpers.hpp"

#include "backend_bridge.hpp"

namespace kvcache_manager {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using namespace native_kvm;

namespace {

constexpr uint32_t kEstablishAttemptTimeoutMs = 250;

// Registry key = (client mesh raw pointer, client coordinate). A second Client
// on the same chip of the same mesh is rejected; different chips of one shared
// mesh are allowed.
using ClientRegistryKey = std::pair<MeshDevice *, MeshCoordinate>;
std::mutex client_registry_mutex;
std::set<ClientRegistryKey> active_clients;

std::atomic<uint64_t> establish_nonce_counter{1};

uint64_t make_establish_nonce() {
  const uint64_t now = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  const uint64_t counter =
      establish_nonce_counter.fetch_add(1, std::memory_order_relaxed);
  const uint64_t nonce = now ^ (counter * 0x9E3779B97F4A7C15ULL);
  return nonce == 0 ? 1 : nonce;
}

} // namespace

// An indeterminate tail may still access every field below. Each Client
// allocates this fixed holder before its first enqueue; moving into it and
// publishing it never allocates.
struct Client::Impl::QuarantineResources {
  std::shared_ptr<MeshDevice> submesh_owner;
  std::shared_ptr<MeshBuffer> establish_tx_buf;
  std::shared_ptr<MeshBuffer> establish_response_buf;
  std::shared_ptr<MeshBuffer> establish_sem_buf;
  std::unique_ptr<MeshWorkload> establish_workload;
  QuarantineResources *next = nullptr;
};

std::atomic<Client::Impl::QuarantineResources *> Client::Impl::quarantine_head_{
    nullptr};

void Client::Impl::quarantine_resources() noexcept {
  if (operations_ != nullptr) {
    backend::client_operations_quarantine(*operations_);
    operations_.reset();
  }
  auto &resources = *quarantine_holder_;
  resources.submesh_owner = std::move(client_submesh_owner_);
  resources.establish_tx_buf = std::move(establish_tx_buf_);
  resources.establish_response_buf = std::move(establish_response_buf_);
  resources.establish_sem_buf = std::move(establish_sem_buf_);
  resources.establish_workload = std::move(establish_workload_);
  publish_quarantine();
}

void Client::Impl::publish_quarantine() noexcept {
  auto *resources = quarantine_holder_.release();
  auto *head = quarantine_head_.load(std::memory_order_relaxed);
  do {
    resources->next = head;
  } while (!quarantine_head_.compare_exchange_weak(
      head, resources, std::memory_order_release, std::memory_order_relaxed));

  try {
    log_warning(tt::LogOp, "kvcache_manager Client: quarantined an "
                           "indeterminate request tail until reset");
  } catch (...) {
    // Publication already retained the resources; diagnostics are best effort.
  }
}

Client::Impl::Impl(MeshDevice *client_mesh,
                   std::optional<tt::tt_fabric::FabricNodeId> server_node,
                   const std::optional<MeshCoordinate> &client_coord,
                   bool static_intermesh_galaxy_to_t3k,
                   uint32_t server_worker_group, uint32_t static_intermesh_lane)
    : client_submesh_(client_mesh),
      static_intermesh_galaxy_to_t3k_(static_intermesh_galaxy_to_t3k),
      server_worker_group_(server_worker_group),
      static_intermesh_lane_(static_intermesh_lane),
      deployment_server_node_(std::move(server_node)) {
  detail::verify_client_operations_abi();
  TT_FATAL(client_submesh_ != nullptr, "client_mesh must be non-null");
  TT_FATAL(IsDefaultMetalContext(*client_submesh_),
           "kvcache_manager Client currently supports only submeshes from the "
           "default MetalContext");
  TT_FATAL(IsFastDispatchEnabled(*client_submesh_),
           "kvcache_manager Client requires fast dispatch because slow "
           "dispatch cannot bound ESTABLISH completion");
  if (client_coord.has_value()) {
    const auto &shape = client_submesh_->shape();
    TT_FATAL(shape.dims() == client_coord->dims() &&
                 MeshCoordinateRange(shape).contains(*client_coord),
             "client_coord {} is outside client_mesh shape {}", *client_coord,
             shape);
    client_coord_ = *client_coord;
  } else {
    TT_FATAL(client_submesh_->shape() == MeshShape(1, 1),
             "client_mesh must have shape (1,1) when no client_coord is given");
  }
  chip_range_ = MeshCoordinateRange(client_coord_);
  validate_kvcache_fabric_config("kvcache_manager Client");
  resolve_client_connection();
  initialize_session();
}

void Client::Impl::initialize_session() {
  client_submesh_owner_ = client_submesh_->shared_from_this();
  quarantine_holder_ = std::make_unique<QuarantineResources>();
  establish_timeout_ms_ = config::resolve_establish_timeout_ms();
  const int clock_rate_mhz = client_submesh_->get_clock_rate_mhz();
  TT_FATAL(clock_rate_mhz > 0,
           "client_submesh AI clock must be positive, got {} MHz",
           clock_rate_mhz);
  const uint32_t attempt_timeout_ms =
      std::min(establish_timeout_ms_, kEstablishAttemptTimeoutMs);
  establish_attempt_timeout_cycles_ =
      static_cast<uint64_t>(attempt_timeout_ms) *
      static_cast<uint64_t>(clock_rate_mhz) * 1'000;
  const uint64_t nonce = make_establish_nonce();
  establish_nonce_low_ = static_cast<uint32_t>(nonce);
  establish_nonce_high_ = static_cast<uint32_t>(nonce >> 32);

  {
    std::lock_guard<std::mutex> lock(client_registry_mutex);
    ClientRegistryKey key{client_submesh_, client_coord_};
    TT_FATAL(active_clients.find(key) == active_clients.end(),
             "A Client is already active on mesh {} at coordinate {}",
             static_cast<const void *>(client_submesh_), client_coord_);
    active_clients.insert(key);
  }

  try {
    kvcache_profile_ =
        config::read_bool_env("KVM_CLIENT_PROFILE").value_or(false);
    put_phase_profile_enabled_ =
        config::read_bool_env("KVM_CLIENT_PUT_PHASE_PROFILE").value_or(false);
    profile_detail_ = config::resolve_profile_detail();
    value_ring_depth_ = config::resolve_client_value_ring_depth();
    operations_ = backend::create_client_operations(make_operations_config());
    const auto addresses =
        backend::client_operations_device_addresses(*operations_);
    client_l1_base_ = addresses.client_l1_base;
    recv_buf_base_ = addresses.recv_l1_base;
    reserve_bootstrap_l1();
    send_establish_request();
    apply_establish_response();
  } catch (...) {
    const auto failure = std::current_exception();
    const bool outcome_unknown = establish_may_have_started_;

    if (outcome_unknown) {
      lifecycle_state_ = LifecycleState::Quarantined;
      quarantine_resources();
    } else {
      std::lock_guard<std::mutex> lock(client_registry_mutex);
      active_clients.erase(ClientRegistryKey{client_submesh_, client_coord_});
    }
    std::rethrow_exception(failure);
  }

  establish_workload_.reset();
  lifecycle_state_ = LifecycleState::Established;
}

Client::Impl::~Impl() noexcept {
  try {
    close();
  } catch (...) {
    // Explicit close preserves established state on a drain failure so callers
    // can retry. Destruction cannot retry later, so release best-effort unless
    // an outcome-unknown tail was quarantined.
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ != LifecycleState::Quarantined) {
      release_host_resources();
    }
  }
}

bool Client::Impl::is_established() const {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  return lifecycle_state_ == LifecycleState::Established;
}

void Client::Impl::close() {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  if (lifecycle_state_ == LifecycleState::Quarantined ||
      lifecycle_state_ == LifecycleState::Closed) {
    return;
  }
  TT_FATAL(lifecycle_state_ == LifecycleState::Established,
           "Client cannot close while it is still constructing");
  try {
    backend::client_operations_close(*operations_);
  } catch (...) {
    sync_operations_quarantine();
    throw;
  }
  sync_operations_quarantine();
  if (lifecycle_state_ == LifecycleState::Quarantined) {
    return;
  }
  release_host_resources();
}

void Client::Impl::release_host_resources() {
  operations_.reset();
  establish_workload_.reset();
  establish_tx_buf_.reset();
  establish_response_buf_.reset();
  establish_sem_buf_.reset();
  {
    std::lock_guard<std::mutex> lock(client_registry_mutex);
    active_clients.erase(ClientRegistryKey{client_submesh_, client_coord_});
  }
  client_submesh_owner_.reset();
  lifecycle_state_ = LifecycleState::Closed;
}

} // namespace kvcache_manager
