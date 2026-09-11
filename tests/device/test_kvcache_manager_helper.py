# SPDX-FileCopyrightText: (c) 2026 Moreh
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helper module for moreh kvcache_manager tests (no test functions).

Provides `make_dram_tensor` — turns a host bytes buffer into a client-chip
DRAM-resident RowMajor UINT8 ttnn tensor (the wire input type for Client.put)
— and `make_empty_dram_tensor`, an equivalently-shaped uninitialised tensor
the caller passes as Client.get's out-param. The `test_` filename prefix is
intentional so the module sits alongside the test files inside the package
(the directory has an `__init__.py`, so relative imports from sibling test
modules resolve).

`make_typed_tensor` / `make_empty_typed_tensor` cover the arbitrary
dtype/layout passthrough path: any INTERLEAVED-DRAM ttnn tensor whose
page_size is a multiple of 32 (TILE bf16/fp32/uint8/bfloat8_b tiles satisfy
this — bfloat8_b is 1088 = 34 × 32; a ROW_MAJOR width × element_size that is
32-aligned does too). The caller keeps the original torch tensor so a wire
round-trip can be checked with `torch.equal`, since the wire is a byte
passthrough. For lossy block-float dtypes (bfloat8_b) the round-trip is
checked dequant-vs-dequant: the reference is ttnn.to_torch of the quantized
input tensor, not the pre-quantization float source.
"""


_CHUNK = 4288  # kMaxChunkValueBytes


def server_fabric_node(server_submesh):
    import ttnn

    return server_submesh.get_fabric_node_id(ttnn.MeshCoordinate(0, 0))


def _round_up_to_chunk(n):
    return ((n + _CHUNK - 1) // _CHUNK) * _CHUNK


def make_dram_tensor(client_submesh, data_bytes):
    """Allocate a RowMajor UINT8 tensor on the client chip's DRAM (via the
    caller-owned client_submesh), padded up to a multiple of
    kMaxChunkValueBytes (= 4288) and shaped as (num_pages,
    kMaxChunkValueBytes) so the kernel's InterleavedAddrGen page_size
    matches. Returns (ttnn.Tensor, len(data_bytes)).
    """
    import torch
    import ttnn

    assert isinstance(data_bytes, (bytes, bytearray)), "data must be bytes"
    assert len(data_bytes) > 0, "DRAM tensor must be non-empty"
    padded_len = _round_up_to_chunk(len(data_bytes))
    pad = bytearray(padded_len - len(data_bytes))
    flat = torch.frombuffer(bytearray(data_bytes) + pad, dtype=torch.uint8)
    num_pages = padded_len // _CHUNK
    shaped = flat.reshape(num_pages, _CHUNK)
    dram_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
    # 1x1 submesh: no mesh_mapper needed; the tensor lives on a single chip.
    t = ttnn.from_torch(
        shaped,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=client_submesh,
        memory_config=dram_mem_cfg,
    )
    return t, len(data_bytes)


def make_empty_dram_tensor(client_submesh, n_bytes):
    """Allocate an out-param tensor for Client.get with the same RowMajor UINT8
    (num_pages, kMaxChunkValueBytes) shape as make_dram_tensor would produce for
    `n_bytes` of payload. Contents are uninitialised — the kernel overwrites the
    first n_bytes on a hit. Returns the ttnn.Tensor.
    """
    import torch
    import ttnn

    assert n_bytes > 0, "out_tensor must be non-empty"
    padded_len = _round_up_to_chunk(n_bytes)
    num_pages = padded_len // _CHUNK
    flat = torch.zeros(padded_len, dtype=torch.uint8)
    shaped = flat.reshape(num_pages, _CHUNK)
    dram_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
    return ttnn.from_torch(
        shaped,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=client_submesh,
        memory_config=dram_mem_cfg,
    )


def make_typed_tensor(client_submesh, src_torch, dtype, layout):
    """Place an arbitrary dtype/layout torch tensor on the client chip's DRAM
    as an INTERLEAVED ttnn tensor (the wire input type for the tiled/typed
    Client.put path). `src_torch` must already be the torch dtype that matches
    `dtype` so the wire byte-passthrough round-trips exactly. Returns
    (ttnn.Tensor, n_bytes) where n_bytes is the device packed byte count of the
    tensor — the byte length the client streams over the wire.

    For TILE layout n_bytes is derived from the device packed page layout
    (padded_shape tile count * ttnn.tile_size(dtype)) rather than the torch
    source element_size. This matters for block-float dtypes (e.g. bfloat8_b),
    where torch has no native type so src_torch is float32 (4 B/elem) while the
    device packs each 32x32 tile into a smaller, dtype-specific page (1088 B for
    bfloat8_b). Using ttnn.tile_size(dtype) keeps n_bytes correct for every
    TILE dtype without special-casing. For ROW_MAJOR the device element size
    matches the torch source, so numel * element_size is exact.
    """
    import ttnn

    t = ttnn.from_torch(
        src_torch,
        dtype=dtype,
        layout=layout,
        device=client_submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if layout == ttnn.TILE_LAYOUT:
        padded = list(t.padded_shape)
        num_tiles = 1
        for d in padded[:-2]:
            num_tiles *= d
        num_tiles *= (padded[-2] // 32) * (padded[-1] // 32)
        n_bytes = num_tiles * ttnn.tile_size(dtype)
    else:
        n_bytes = src_torch.numel() * src_torch.element_size()
    return t, n_bytes


def make_empty_typed_tensor(client_submesh, shape, torch_dtype, dtype, layout):
    """Allocate a zeros-filled out-param tensor for Client.get with the same
    shape/dtype/layout as `make_typed_tensor` would produce. The kernel
    overwrites it on a hit; `torch_dtype` is the torch dtype matching `dtype`.
    Returns the ttnn.Tensor.
    """
    import torch
    import ttnn

    zeros = torch.zeros(shape, dtype=torch_dtype)
    return ttnn.from_torch(
        zeros,
        dtype=dtype,
        layout=layout,
        device=client_submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
