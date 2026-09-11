// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>
#include <string>

#include "server_runtime/server_device_runtime.hpp"
#include <ttnn/operations/kvcache_manager/client/client_operations.hpp>

namespace kvcache_manager::detail {

namespace client_backend = ttnn::operations::kvcache_manager::backend;

inline constexpr uint32_t kRequiredClientOperationsAbiVersion = 4;
static_assert(client_backend::kClientOperationsAbiVersion ==
              kRequiredClientOperationsAbiVersion);

inline void verify_client_operations_abi() {
  const uint32_t actual = client_backend::client_operations_abi_version();
  if (actual != kRequiredClientOperationsAbiVersion) {
    throw std::runtime_error(
        "kvcache-manager requires TTNN KV cache Client operations ABI " +
        std::to_string(kRequiredClientOperationsAbiVersion) +
        ", but loaded ABI " + std::to_string(actual));
  }
}

} // namespace kvcache_manager::detail
