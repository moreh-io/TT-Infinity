# SPDX-FileCopyrightText: (c) 2026 Moreh
#
# SPDX-License-Identifier: Apache-2.0
"""The server kernel's NCRISC image must fit its IRAM with room to spare.

"Does it fit" is already covered everywhere: an oversized image throws at Server construction, so
every test that builds a Server catches it. What nothing caught was the margin. The image has twice
been driven close to the ceiling -- once inherited, once by adding per-client state -- and on both
occasions the entire suite stayed green at a few hundred spare bytes, until a configuration that
happened to compile slightly larger fell off the edge and took the whole device down with a hard
throw at construction.

So this asserts headroom, not fitness. It is meant to fail while there is still room to react.

RUN EACH TEST HERE IN ITS OWN PYTEST PROCESS. Both drop the kernel's JIT output so the size they
measure comes from the current source, but a process that has already built that kernel serves it
from its in-memory program cache and writes nothing back -- leaving nothing to measure. They also
need different device counts (peer-rich vs exactly one chip), so they cannot share an invocation
with each other either.
"""

import os
import pathlib
import shutil
import struct

import pytest

# The hardware ceiling the kernel is linked against.
IRAM_LIMIT = 0x4000
# Fail once the image is within this much of the ceiling. Sized so the current worst variant passes
# with a real margin while still leaving a warning shot before construction starts throwing.
REQUIRED_HEADROOM = 512
BUDGET = IRAM_LIMIT - REQUIRED_HEADROOM

_CACHE_ROOT = pathlib.Path(
    os.environ.get("TT_METAL_CACHE", pathlib.Path.home() / ".cache" / "tt-metal-cache")
)


def _load_segment_bytes(elf_path: pathlib.Path) -> int:
    """Memory size of the first PT_LOAD segment of a 32-bit little-endian ELF.

    Parsed directly rather than shelling out to readelf: this runs wherever the tests do, and the
    number it reports is the same one the loader compares against the IRAM limit.
    """
    data = elf_path.read_bytes()
    assert data[:4] == b"\x7fELF", f"{elf_path} is not an ELF"
    assert data[4] == 1, "expected a 32-bit ELF"
    (e_phoff,) = struct.unpack_from("<I", data, 0x1C)
    (e_phentsize,) = struct.unpack_from("<H", data, 0x2A)
    (e_phnum,) = struct.unpack_from("<H", data, 0x2C)
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, _p_offset, _p_vaddr, _p_paddr, _p_filesz, p_memsz = struct.unpack_from(
            "<6I", data, off
        )
        if p_type == 1:  # PT_LOAD
            return p_memsz
    raise AssertionError(f"{elf_path} has no PT_LOAD segment")


def _server_kernel_images():
    return sorted(_CACHE_ROOT.glob("*/kernels/server_kernel/*/ncrisc/ncrisc.elf"))


def test_server_kernel_ncrisc_image_keeps_its_headroom(
    kvcache_manager_multiclient_setup,
):
    """Build the kernel fresh and check every variant it produces against the budget.

    Uses the peer-rich fixture (EIGHT visible devices) on purpose: the image grows with what the
    server must be able to reach, so the variant worth measuring is the one with peers, not the
    minimal two-chip build.
    """
    from kvcache_manager import Server

    _parent_mesh, _model_submesh, server_submesh = kvcache_manager_multiclient_setup

    # Drop only this kernel's JIT output so the sizes below come from the current source rather than
    # whatever an earlier build left behind. It is derived data and rebuilds on demand; the cost is
    # one compile. Do not run this suite concurrently with another that shares the cache.
    for cached in _CACHE_ROOT.glob("*/kernels/server_kernel"):
        shutil.rmtree(cached, ignore_errors=True)

    with Server(server_submesh):
        pass

    images = _server_kernel_images()
    assert images, (
        "no server_kernel NCRISC image was produced. Either the cache layout moved (looked under "
        f"{_CACHE_ROOT}), or this test shared a process with one that had already built the kernel -- "
        "the in-memory program cache then serves it and nothing is written back. Run this test alone."
    )

    sizes = {str(p): _load_segment_bytes(p) for p in images}
    worst_path, worst = max(sizes.items(), key=lambda kv: kv[1])
    assert worst <= BUDGET, (
        f"server kernel NCRISC image is {worst} B (0x{worst:x}), leaving only {IRAM_LIMIT - worst} B under the "
        f"{IRAM_LIMIT} B IRAM limit; at least {REQUIRED_HEADROOM} B must remain.\n"
        f"largest variant: {worst_path}\n"
        "Shrink the kernel before adding to it -- rarely-taken handlers can be marked KVM_COLD_HANDLER "
        "(noinline + Oz), and a build with no reachable peer already compiles only its idle loop via "
        "KVM_SERVER_IDLE_ONLY. See kvcache_manager README section 11."
    )


def test_a_peerless_server_compiles_only_its_idle_loop(
    kvcache_manager_mesh_and_submesh,
):
    """The no-peer build must stay tiny. Needs exactly ONE visible device.

    It serves nothing, so it compiles just the termination wait. That is not a nicety: carrying the
    request handlers into this build is what used to overrun the IRAM limit and make a 1x1 mesh
    unable to construct a Server at all.
    """
    from kvcache_manager import Server

    _parent_mesh, server_submesh = kvcache_manager_mesh_and_submesh

    for cached in _CACHE_ROOT.glob("*/kernels/server_kernel"):
        shutil.rmtree(cached, ignore_errors=True)

    with Server(server_submesh):
        pass

    images = _server_kernel_images()
    assert images, (
        "no server_kernel NCRISC image was produced; run this test in its own process (see the "
        "module docstring)"
    )
    smallest = min(_load_segment_bytes(p) for p in images)
    # Generous bound: the idle build measures a few hundred bytes, while a build carrying the
    # handlers is over ten times that. This separates the two without pinning an exact size.
    assert smallest < 0x800, (
        f"the peerless server kernel is {smallest} B (0x{smallest:x}); it should contain only the "
        "termination wait. KVM_SERVER_IDLE_ONLY may no longer be reaching this build."
    )
