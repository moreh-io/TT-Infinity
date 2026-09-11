// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace kvcache_manager::detail {

class SharedMemoryMapping final {
public:
  static SharedMemoryMapping create(std::size_t size);

  ~SharedMemoryMapping() noexcept;
  SharedMemoryMapping(SharedMemoryMapping &&other) noexcept;
  SharedMemoryMapping &operator=(SharedMemoryMapping &&other) noexcept;

  SharedMemoryMapping(const SharedMemoryMapping &) = delete;
  SharedMemoryMapping &operator=(const SharedMemoryMapping &) = delete;

  [[nodiscard]] void *data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
  SharedMemoryMapping(void *data, std::size_t size) noexcept;
  void reset() noexcept;

  void *data_ = nullptr;
  std::size_t size_ = 0;
};

} // namespace kvcache_manager::detail
