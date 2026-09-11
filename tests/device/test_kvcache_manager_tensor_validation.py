# SPDX-FileCopyrightText: (c) 2026 Moreh
#
# SPDX-License-Identifier: Apache-2.0

"""Client tensor storage and device-ownership validation."""

from .test_kvcache_manager_helper import (
    make_dram_tensor,
    make_empty_dram_tensor,
    server_fabric_node,
)


def test_put_get_reject_non_client_tensors_without_state_corruption(
    kvcache_manager_client_setup, expect_error
):
    import torch
    import ttnn
    from kvcache_manager import Client, Server

    _parent_mesh, client_submesh, server_submesh = kvcache_manager_client_setup
    value = b"tensor-ownership"

    with (
        Server(server_submesh) as server,
        Client(client_submesh, server_fabric_node(server_submesh)) as client,
    ):
        valid_value, value_len = make_dram_tensor(client_submesh, value)
        client.put(b"valid", valid_value, value_len, sync_commit=True)

        def allocate_validation_tensor(shape, memory_config, dtype=ttnn.uint8):
            return ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape),
                dtype,
                ttnn.ROW_MAJOR_LAYOUT,
                client_submesh,
                memory_config,
            )

        l1_tensor = allocate_validation_tensor((1, 32), ttnn.L1_MEMORY_CONFIG)
        unaligned_page_tensor = allocate_validation_tensor(
            (1, 2, 3, 4), ttnn.DRAM_MEMORY_CONFIG
        )
        # DRAM-sharded buffers enumerate banks on y=0; one bank keeps construction portable.
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}
        )
        dram_shard_spec = ttnn.ShardSpec(
            dram_grid,
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        dram_sharded_tensor = allocate_validation_tensor(
            (1, 1, 32, 32),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.DRAM,
                dram_shard_spec,
            ),
            dtype=ttnn.bfloat16,
        )
        rank6_tensor = allocate_validation_tensor(
            (1, 1, 1, 1, 1, 32), ttnn.DRAM_MEMORY_CONFIG
        )

        host_tensor = ttnn.from_torch(
            torch.zeros((1, 4288), dtype=torch.uint8),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        server_value, _ = make_dram_tensor(server_submesh, value)
        deallocated_value, _ = make_dram_tensor(client_submesh, value)
        ttnn.deallocate(deallocated_value)

        invalid_puts = (
            (host_tensor, "put: tensor must use DEVICE storage"),
            (deallocated_value, "put: tensor must be allocated on device"),
            (server_value, "put: tensor must belong to client_submesh"),
            (l1_tensor, "put: tensor must be in DRAM"),
            (dram_sharded_tensor, "put: tensor must be INTERLEAVED DRAM"),
            (rank6_tensor, "put: tensor rank 6 exceeds kTensorShapeMaxRank 5"),
            (unaligned_page_tensor, "put: tensor page_size 4 must be a multiple of 32"),
        )
        for tensor, message in invalid_puts:
            with expect_error(RuntimeError, message):
                client.put(b"invalid-put", tensor, value_len)

        assert not client.exists(b"invalid-put")
        assert client.ping(b"after-invalid-put") == b"after-invalid-put"

        host_output = ttnn.from_torch(
            torch.zeros((1, 4288), dtype=torch.uint8),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        server_output = make_empty_dram_tensor(server_submesh, value_len)
        deallocated_output = make_empty_dram_tensor(client_submesh, value_len)
        ttnn.deallocate(deallocated_output)

        invalid_gets = (
            (host_output, "get: tensor must use DEVICE storage"),
            (deallocated_output, "get: tensor must be allocated on device"),
            (server_output, "get: tensor must belong to client_submesh"),
            (l1_tensor, "get: tensor must be in DRAM"),
            (dram_sharded_tensor, "get: tensor must be INTERLEAVED DRAM"),
            (rank6_tensor, "get: tensor rank 6 exceeds kTensorShapeMaxRank 5"),
            (unaligned_page_tensor, "get: tensor page_size 4 must be a multiple of 32"),
        )
        for tensor, message in invalid_gets:
            with expect_error(RuntimeError, message):
                client.get(b"valid", tensor)

        assert client.ping(b"after-invalid-get") == b"after-invalid-get"
        valid_output = make_empty_dram_tensor(client_submesh, value_len)
        assert client.get(b"valid", valid_output)
        expected = torch.tensor(list(value), dtype=torch.uint8)
        assert torch.equal(ttnn.to_torch(valid_output).flatten()[:value_len], expected)
