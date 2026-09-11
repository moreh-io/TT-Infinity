// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/device.hpp>
#include <tt_stl/assert.hpp>

#include <ttnn/operations/kvcache_manager/common/protocol/bootstrap.hpp>

namespace kvcache_manager::detail {

inline tt::tt_metal::CoreCoord
require_bootstrap_worker_noc(tt::tt_metal::IDevice *device,
                             tt::tt_metal::CoreCoord logical_core,
                             const char *role) {
  TT_FATAL(device != nullptr, "{} has no selected device", role);
  TT_FATAL(device->arch() == tt::ARCH::WORMHOLE_B0,
           "{} endpoint-free bootstrap supports only WORMHOLE_B0", role);

  const auto translated = device->worker_core_from_logical_core(logical_core);
  TT_FATAL(ttnn::operations::kvcache_manager::protocol::is_bootstrap_worker_noc(
               static_cast<uint32_t>(logical_core.x),
               static_cast<uint32_t>(logical_core.y),
               static_cast<uint32_t>(translated.x),
               static_cast<uint32_t>(translated.y)),
           "{} requires logical worker ({}, {}) to map to canonical translated "
           "NoC coordinate ({}, {}), got ({}, {})",
           role, logical_core.x, logical_core.y,
           ttnn::operations::kvcache_manager::protocol::bootstrap_worker_noc_x(
               logical_core.x),
           ttnn::operations::kvcache_manager::protocol::bootstrap_worker_noc_y(
               logical_core.y),
           translated.x, translated.y);
  return translated;
}

} // namespace kvcache_manager::detail
