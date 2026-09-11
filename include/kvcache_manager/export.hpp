// SPDX-FileCopyrightText: (c) 2026 Moreh
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(_WIN32)
#if defined(KVCACHE_MANAGER_BUILDING_LIBRARY)
#define KVCACHE_MANAGER_API __declspec(dllexport)
#else
#define KVCACHE_MANAGER_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define KVCACHE_MANAGER_API __attribute__((visibility("default")))
#else
#define KVCACHE_MANAGER_API
#endif
