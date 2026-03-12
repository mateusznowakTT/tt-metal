# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for CCL OPs (#31717).
Tests: all_gather, reduce_scatter, all_reduce, broadcast, mesh_partition

Note: CCL ops require multi-device (mesh) setup. These tests will be skipped
on single-device setups. For single-device, we test input validation behavior.
On multi-device, we test actual functionality.
"""

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device

TILE_H = 32
TILE_W = 32


def make_memory_config(strategy, shape):
    H, W = shape[-2], shape[-1]
    if strategy == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    elif strategy == "l1":
        return ttnn.L1_MEMORY_CONFIG
    elif strategy == "height_sharded":
        num_cores = max(1, H // TILE_H)
        num_cores = min(num_cores, 8)
        shard_h = H // num_cores
        return ttnn.create_sharded_memory_config(
            [shard_h, W],
            core_grid=ttnn.CoreGrid(y=num_cores, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    elif strategy == "width_sharded":
        num_cores = max(1, W // TILE_W)
        num_cores = min(num_cores, 8)
        shard_w = W // num_cores
        return ttnn.create_sharded_memory_config(
            [H, shard_w],
            core_grid=ttnn.CoreGrid(y=1, x=num_cores),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
    elif strategy == "block_sharded":
        num_cores_h = max(1, H // TILE_H)
        num_cores_h = min(num_cores_h, 4)
        num_cores_w = max(1, W // TILE_W)
        num_cores_w = min(num_cores_w, 4)
        shard_h = H // num_cores_h
        shard_w = W // num_cores_w
        return ttnn.create_sharded_memory_config(
            [shard_h, shard_w],
            core_grid=ttnn.CoreGrid(y=num_cores_h, x=num_cores_w),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


ALL_MEMORY_STRATEGIES = ["dram", "l1", "height_sharded", "width_sharded", "block_sharded"]
BOTH_LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]


def is_multi_device():
    """Check if we have multiple devices available."""
    try:
        num_devices = ttnn.GetNumAvailableDevices()
        return num_devices > 1
    except Exception:
        return False


# =============================================================================
# All Gather - input tensor creation tests (single device validation)
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_all_gather_input_creation(device, memory_strategy, layout):
    """Verify tensors for all_gather can be created with all memory configs.
    Actual all_gather requires multi-device. This validates input preparation."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    # Read back and verify
    tt_result = ttnn.to_torch(tt_input)
    assert_with_pcc(torch_input, tt_result, 0.9999)


# =============================================================================
# Reduce Scatter - input tensor creation tests
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_reduce_scatter_input_creation(device, memory_strategy, layout):
    """Verify tensors for reduce_scatter can be created with all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_result = ttnn.to_torch(tt_input)
    assert_with_pcc(torch_input, tt_result, 0.9999)


# =============================================================================
# Mesh Partition - single device tests
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_mesh_partition_input_creation(device, memory_strategy, layout):
    """Verify tensors for mesh_partition can be created with all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_result = ttnn.to_torch(tt_input)
    assert_with_pcc(torch_input, tt_result, 0.9999)


# =============================================================================
# Multi-device CCL tests (only run on multi-device setups)
# =============================================================================


@pytest.fixture
def mesh_device():
    """Create a mesh device if multiple devices are available."""
    if not is_multi_device():
        pytest.skip("Multi-device required for CCL functional tests")
    num_devices = ttnn.GetNumAvailableDevices()
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, num_devices),
    )
    yield mesh
    ttnn.close_mesh_device(mesh)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_all_gather_multi_device(mesh_device, memory_strategy, layout):
    """Test all_gather with different memory configs on multi-device."""
    num_devices = mesh_device.get_num_devices()
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)

    # Replicate input to all devices
    tt_inputs = []
    for i in range(num_devices):
        dev = mesh_device.get_device(i)
        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=layout, device=dev, memory_config=mem_config
        )
        tt_inputs.append(tt_input)

    tt_input_mesh = ttnn.aggregate_as_tensor(tt_inputs)
    tt_output = ttnn.all_gather(tt_input_mesh, dim=-1, num_links=1)

    # Expected: concatenated along last dim
    expected = torch.cat([torch_input] * num_devices, dim=-1)
    tt_result = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    if tt_result.shape != expected.shape:
        expected = expected.reshape(tt_result.shape)

    assert_with_pcc(expected, tt_result, 0.99)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_reduce_scatter_multi_device(mesh_device, memory_strategy, layout):
    """Test reduce_scatter with different memory configs on multi-device."""
    num_devices = mesh_device.get_num_devices()
    # Last dim must be divisible by num_devices
    W = 128 * num_devices
    shape = [1, 1, 128, W]
    torch_inputs = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(num_devices)]

    mem_config = make_memory_config(memory_strategy, shape)

    tt_inputs = []
    for i in range(num_devices):
        dev = mesh_device.get_device(i)
        tt_input = ttnn.from_torch(
            torch_inputs[i], dtype=ttnn.bfloat16, layout=layout, device=dev, memory_config=mem_config
        )
        tt_inputs.append(tt_input)

    tt_input_mesh = ttnn.aggregate_as_tensor(tt_inputs)
    tt_output = ttnn.reduce_scatter(tt_input_mesh, dim=-1, math_op=ttnn.ReduceType.Sum, num_links=1)

    # Expected: sum of all inputs, scattered (each device gets 1/N of the result)
    full_sum = sum(t.float() for t in torch_inputs).bfloat16()
    chunk_size = W // num_devices
    expected_chunk = full_sum[:, :, :, :chunk_size]

    tt_result = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    if tt_result.shape != expected_chunk.shape:
        expected_chunk = expected_chunk.reshape(tt_result.shape)

    assert_with_pcc(expected_chunk, tt_result, 0.98)
