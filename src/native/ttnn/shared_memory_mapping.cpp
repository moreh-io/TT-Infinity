// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#include "shared_memory_mapping.hpp"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace kvcache_manager::detail {
namespace {

std::runtime_error system_error(const char *operation, int error) {
  return std::runtime_error(std::string(operation) + ": " +
                            std::strerror(error));
}

} // namespace

SharedMemoryMapping::SharedMemoryMapping(void *data, std::size_t size) noexcept
    : data_(data), size_(size) {}

SharedMemoryMapping SharedMemoryMapping::create(std::size_t size) {
  if (size == 0) {
    throw std::invalid_argument("shared-memory mapping size must be positive");
  }

  static std::atomic<uint64_t> sequence{0};
  int fd = -1;
  std::string name;
  for (unsigned attempt = 0; attempt < 32; ++attempt) {
    name = "/kvcache_manager_" + std::to_string(getpid()) + "_" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
    fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd >= 0) {
      break;
    }
    if (errno != EEXIST) {
      throw system_error("shm_open", errno);
    }
  }
  if (fd < 0) {
    throw std::runtime_error("failed to allocate a unique shared-memory name");
  }

  if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
    const int error = errno;
    close(fd);
    shm_unlink(name.c_str());
    throw system_error("ftruncate", error);
  }

  void *data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  const int map_error = errno;
  close(fd);
  if (data == MAP_FAILED) {
    shm_unlink(name.c_str());
    throw system_error("mmap", map_error);
  }

  // The mapping remains valid after unlink, and a crash cannot leave a stale
  // object in /dev/shm because no second process opens this staging region.
  if (shm_unlink(name.c_str()) != 0) {
    const int error = errno;
    munmap(data, size);
    throw system_error("shm_unlink", error);
  }

  std::memset(data, 0, size);
  return SharedMemoryMapping(data, size);
}

SharedMemoryMapping::~SharedMemoryMapping() noexcept { reset(); }

SharedMemoryMapping::SharedMemoryMapping(SharedMemoryMapping &&other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

SharedMemoryMapping &
SharedMemoryMapping::operator=(SharedMemoryMapping &&other) noexcept {
  if (this != &other) {
    reset();
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void SharedMemoryMapping::reset() noexcept {
  if (data_ != nullptr) {
    munmap(data_, size_);
    data_ = nullptr;
    size_ = 0;
  }
}

} // namespace kvcache_manager::detail
