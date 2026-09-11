// SPDX-FileCopyrightText: (c) 2026 Moreh
//
// SPDX-License-Identifier: Apache-2.0

#include "server_runtime/server_runtime_core.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <thread>
#include <tuple>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>

#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/operations/kvcache_manager/common/config.hpp"
#include "ttnn/operations/kvcache_manager/common/fabric_helpers.hpp"
#include "ttnn/operations/kvcache_manager/common/protocol/layout.hpp"
#include "ttnn/operations/kvcache_manager/common/protocol/packet.hpp"

namespace kvcache_manager::detail::server_runtime {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace {
// Bound mismatching re-reads after the initial candidate image.
constexpr uint32_t kValueReadMaxMismatches = 101;
} // namespace

const uint8_t *ServerRuntimeCore::host_value_bytes(const HostValue &hv) const {
  if (hv.pool_units > 0) {
    return pool_host_ptr_ +
           static_cast<size_t>(hv.pool_unit) * protocol::kMaxChunkValueBytes;
  }
  return hv.heap.data();
}

// First-fit over the free extents. Returns the unit offset of a run of `units`
// contiguous units, or nullopt when the pool cannot hold it (the caller then
// uses a heap vector). A value is never split: the device reads it as one span.
std::optional<uint32_t> ServerRuntimeCore::pool_alloc(uint32_t units) {
  if (pool_host_ptr_ == nullptr || units == 0) {
    return std::nullopt;
  }
  for (auto it = pool_free_.begin(); it != pool_free_.end(); ++it) {
    if (it->second < units) {
      continue;
    }
    const uint32_t unit = it->first;
    const uint32_t leftover = it->second - units;
    pool_free_.erase(it);
    if (leftover > 0) {
      pool_free_.emplace(unit + units, leftover);
    }
    return unit;
  }
  return std::nullopt;
}

// Return an extent, merging it with an adjacent free extent on either side so
// the pool cannot fragment into unusable slivers.
void ServerRuntimeCore::pool_release(uint32_t unit, uint32_t units) {
  if (units == 0) {
    return;
  }
  TT_FATAL(unit + units <= pool_units_total_ && unit + units >= unit,
           "pool_release: extent [{},{}) is outside the {}-unit pool", unit,
           unit + units, pool_units_total_);
  auto next = pool_free_.lower_bound(unit);
  TT_FATAL(next == pool_free_.end() || next->first >= unit + units,
           "pool_release: extent [{},{}) overlaps free extent [{},{})", unit,
           unit + units, next->first, next->first + next->second);
  if (next != pool_free_.end() && next->first == unit + units) {
    units += next->second;
    next = pool_free_.erase(next);
  }
  if (next != pool_free_.begin()) {
    auto prev = std::prev(next);
    TT_FATAL(prev->first + prev->second <= unit,
             "pool_release: extent [{},{}) overlaps free extent [{},{})", unit,
             unit + units, prev->first, prev->first + prev->second);
    if (prev->first + prev->second == unit) {
      prev->second += units;
      return;
    }
  }
  pool_free_.emplace(unit, units);
}

void ServerRuntimeCore::invalidate_pool_meta(uint32_t slot) {
  if (pool_meta_host_ptr_ == nullptr) {
    return;
  }
  TT_FATAL(slot < max_keys_, "invalidate_pool_meta: slot {} >= max_keys {}",
           slot, max_keys_);
  volatile uint32_t *meta = reinterpret_cast<volatile uint32_t *>(
      pool_meta_host_ptr_ +
      static_cast<size_t>(slot) * protocol::kHostPoolMetaBytes);
  meta[protocol::kHostPoolMetaGenerationBeginWord] = 0;
  __builtin_ia32_sfence();
  meta[protocol::kHostPoolMetaGenerationEndWord] = 0;
  meta[protocol::kHostPoolMetaReadOffWord] = 0;
  meta[protocol::kHostPoolMetaTotalLenWord] = 0;
  __builtin_ia32_sfence();
}

void ServerRuntimeCore::invalidate_pool_meta_if_generation(
    uint32_t slot, uint32_t generation) {
  if (pool_meta_host_ptr_ == nullptr || generation == 0) {
    return;
  }
  TT_FATAL(slot < max_keys_,
           "invalidate_pool_meta_if_generation: slot {} >= max_keys {}", slot,
           max_keys_);
  volatile uint32_t *meta = reinterpret_cast<volatile uint32_t *>(
      pool_meta_host_ptr_ +
      static_cast<size_t>(slot) * protocol::kHostPoolMetaBytes);
  if (meta[protocol::kHostPoolMetaGenerationBeginWord] != generation ||
      meta[protocol::kHostPoolMetaGenerationEndWord] != generation) {
    return;
  }
  invalidate_pool_meta(slot);
}

uint32_t ServerRuntimeCore::publish_pool_meta(uint32_t slot, uint32_t read_off,
                                              uint32_t total_len) {
  TT_FATAL(pool_meta_host_ptr_ != nullptr,
           "publish_pool_meta requires mapped host metadata");
  TT_FATAL(slot < max_keys_, "publish_pool_meta: slot {} >= max_keys {}", slot,
           max_keys_);
  uint32_t generation = ++pool_meta_generations_[slot];
  if (generation == 0) {
    generation = ++pool_meta_generations_[slot];
  }

  volatile uint32_t *meta = reinterpret_cast<volatile uint32_t *>(
      pool_meta_host_ptr_ +
      static_cast<size_t>(slot) * protocol::kHostPoolMetaBytes);
  meta[protocol::kHostPoolMetaGenerationBeginWord] = 0;
  __builtin_ia32_sfence();
  meta[protocol::kHostPoolMetaReadOffWord] = read_off;
  meta[protocol::kHostPoolMetaTotalLenWord] = total_len;
  meta[protocol::kHostPoolMetaGenerationEndWord] = generation;
  __builtin_ia32_sfence();
  meta[protocol::kHostPoolMetaGenerationBeginWord] = generation;
  __builtin_ia32_sfence();
  return generation;
}

// Erase a host backing value and reclaim its pool extent. Called with
// host_values_mutex_ held.
void ServerRuntimeCore::drop_host_value(uint32_t slot) {
  auto it = host_values_.find(slot);
  if (it == host_values_.end()) {
    return;
  }
  const uint32_t unit = it->second.pool_unit;
  const uint32_t units = it->second.pool_units;
  invalidate_pool_meta(slot);
  host_values_.erase(it);
  if (units != 0) {
    pool_release(unit, units);
  }
}

// Targeted PCIe read of just this value's heap units into the caller's storage,
// with a double-read stability guard. Large values are read once per occupied
// DRAM bank and scattered back to logical unit order; small values keep direct
// unit reads because bank grouping would not reduce calls.
void ServerRuntimeCore::read_value_units_into(std::span<uint8_t> dst,
                                              uint32_t value_byte_off,
                                              uint32_t total_len,
                                              ValueReadProfile *profile) {
  IDevice *device = server_submesh_->get_device(MeshCoordinate(0, 0));
  TT_FATAL(device != nullptr,
           "read_value_units_into: server_submesh has no chip at (0,0)");
  auto *value_dev = dram_value_buf_->get_device_buffer(MeshCoordinate(0, 0));
  TT_FATAL(
      value_dev != nullptr,
      "read_value_units_into: value MeshBuffer has no device buffer at (0,0)");
  const uint32_t U = protocol::kMaxChunkValueBytes;
  const uint32_t num_banks =
      device->allocator()->get_num_banks(BufferType::DRAM);
  const uint32_t aligned_page =
      static_cast<uint32_t>(value_dev->aligned_page_size());
  const uint32_t base = static_cast<uint32_t>(value_dev->address());
  const uint32_t base_unit = value_byte_off / U;
  const uint32_t units = protocol::chunk_count_for(total_len);
  TT_FATAL(
      dst.size() >= static_cast<size_t>(units) * U,
      "read_value_units_into: destination {} B holds fewer than {} whole units",
      dst.size(), units);
  TT_FATAL(aligned_page >= U,
           "read_value_units_into: aligned page {} B is smaller than value "
           "unit {} B",
           aligned_page, U);
  // `dst` remains the candidate image; only a mismatch copies the newer read
  // into it.
  std::vector<uint8_t> other(static_cast<size_t>(units) * U, 0);
  const bool bulk = units > num_banks;
  const uint32_t max_pages_per_bank = (units + num_banks - 1) / num_banks;
  std::vector<uint8_t> scratch(
      bulk ? static_cast<size_t>(max_pages_per_bank) * aligned_page : 0);
  auto read_all = [&](uint8_t *into) {
    if (profile != nullptr) {
      ++profile->passes;
    }
    if (!bulk) {
      for (uint32_t u = 0; u < units; ++u) {
        const uint32_t unit = base_unit + u;
        const int bank = static_cast<int>(unit % num_banks);
        const uint32_t page_addr = base + (unit / num_banks) * aligned_page;
        const auto io_start = profile != nullptr
                                  ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
        tt::tt_metal::detail::ReadFromDeviceDRAMChannel(
            device, bank, page_addr,
            std::span<uint8_t>(into + static_cast<size_t>(u) * U, U));
        if (profile != nullptr) {
          profile->io_ns +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - io_start)
                  .count();
          ++profile->calls;
          profile->bytes += U;
        }
      }
      return;
    }

    const uint32_t bank_rotation = base_unit % num_banks;
    for (uint32_t bank = 0; bank < num_banks; ++bank) {
      const uint32_t first_u = (bank + num_banks - bank_rotation) % num_banks;
      if (first_u >= units) {
        continue;
      }
      const uint32_t n_pages = 1 + (units - 1 - first_u) / num_banks;
      const uint32_t first_page_addr =
          base + ((base_unit + first_u) / num_banks) * aligned_page;
      const size_t read_bytes = static_cast<size_t>(n_pages) * aligned_page;
      const auto io_start = profile != nullptr
                                ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
      tt::tt_metal::detail::ReadFromDeviceDRAMChannel(
          device, static_cast<int>(bank), first_page_addr,
          std::span<uint8_t>(scratch.data(), read_bytes));
      if (profile != nullptr) {
        profile->io_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - io_start)
                              .count();
        ++profile->calls;
        profile->bytes += read_bytes;
      }

      const auto scatter_start = profile != nullptr
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
      for (uint32_t page = 0; page < n_pages; ++page) {
        const uint32_t u = first_u + page * num_banks;
        std::memcpy(into + static_cast<size_t>(u) * U,
                    scratch.data() + static_cast<size_t>(page) * aligned_page,
                    U);
      }
      if (profile != nullptr) {
        profile->scatter_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - scatter_start)
                .count();
      }
    }
  };
  read_all(dst.data());
  for (uint32_t attempt = 0;; ++attempt) {
    read_all(other.data());
    if (std::memcmp(dst.data(), other.data(), total_len) == 0) {
      break;
    }
    if (profile != nullptr) {
      ++profile->retries;
    }
    std::memcpy(dst.data(), other.data(), static_cast<size_t>(units) * U);
    if (attempt + 1 >= kValueReadMaxMismatches) {
      log_warning(tt::LogOp,
                  "read_value_units_into (value_byte_off {}): value unstable "
                  "over {} bytes after {} retries; using last "
                  "read",
                  value_byte_off, total_len, attempt + 1);
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

std::vector<uint8_t>
ServerRuntimeCore::read_value_units(uint32_t value_byte_off,
                                    uint32_t total_len) {
  const uint32_t U = protocol::kMaxChunkValueBytes;
  const uint32_t units = protocol::chunk_count_for(total_len);
  std::vector<uint8_t> bytes(static_cast<size_t>(units) * U, 0);
  read_value_units_into(std::span<uint8_t>(bytes), value_byte_off, total_len);
  bytes.resize(total_len);
  return bytes;
}

} // namespace kvcache_manager::detail::server_runtime
