// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/operations/kvcache_manager/common/config.hpp>
#include <ttnn/operations/kvcache_manager/common/protocol/layout.hpp>
#include <ttnn/operations/kvcache_manager/common/server_descriptor.hpp>

namespace kvcache_manager::detail::server_runtime {

namespace ttnn_kvm = ::ttnn::operations::kvcache_manager;
namespace config = ttnn_kvm::config;
namespace protocol = ttnn_kvm::protocol;
using ServerDescriptor = ttnn_kvm::ServerDescriptor;

} // namespace kvcache_manager::detail::server_runtime
