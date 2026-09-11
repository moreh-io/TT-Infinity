// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace kvcache_manager {

struct MoveProfile {
  uint64_t to_host_count = 0;
  uint64_t to_host_worker_ns = 0;
  uint64_t to_host_complete_ns = 0;
  uint64_t to_host_read_calls = 0;
  uint64_t to_host_read_bytes = 0;
  uint64_t to_host_read_passes = 0;
  uint64_t to_host_read_retries = 0;
  uint64_t to_host_read_io_ns = 0;
  uint64_t to_host_scatter_ns = 0;
  uint64_t to_host_device_pushes = 0;
  uint64_t to_device_count = 0;
  uint64_t to_device_worker_ns = 0;
  uint64_t to_device_complete_ns = 0;
};

struct AllocatorProfile {
  uint64_t allocation_calls = 0;
  uint64_t allocation_successes = 0;
  uint64_t allocation_failures = 0;
  uint64_t victim_scans = 0;
  uint64_t victims_evicted = 0;
  uint64_t lease_calls = 0;
  uint64_t lease_successes = 0;
  uint64_t lease_failures = 0;
  uint64_t lease_victim_scans = 0;
  uint64_t lease_victims_evicted = 0;
  uint64_t scan_cycles = 0;
  uint64_t lease_scan_cycles = 0;
  uint64_t allocation_cycles = 0;
  uint32_t clock_mhz = 0;
};

struct PutProfile {
  uint64_t count = 0;
  uint64_t prepare_ns = 0;
  uint64_t program_ns = 0;
  uint64_t enqueue_ns = 0;
  uint64_t landed_wait_ns = 0;
  uint64_t commit_wait_ns = 0;
  uint64_t total_ns = 0;
};

} // namespace kvcache_manager
