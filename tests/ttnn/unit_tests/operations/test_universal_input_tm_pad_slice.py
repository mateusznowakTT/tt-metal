# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for TM OPs - Pad/Slice/Concat/Split (#31716).
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


# =============================================================================
# Pad
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_pad_last_dim(device, memory_strategy, layout):
    """Test padding on last dimension with all memory configs."""
    shape = [1, 1, 64, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    # Pad to 128 on last dim
    tt_output = ttnn.pad(tt_input, padding=((0, 0), (0, 0), (0, 0), (0, 64)), value=0.0)
    torch_output = torch.nn.functional.pad(torch_input, (0, 64, 0, 0), value=0.0)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_pad_height(device, memory_strategy, layout):
    """Test padding on height dimension with all memory configs."""
    shape = [1, 1, 64, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.pad(tt_input, padding=((0, 0), (0, 0), (0, 64), (0, 0)), value=0.0)
    torch_output = torch.nn.functional.pad(torch_input, (0, 0, 0, 64), value=0.0)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


# =============================================================================
# Slice
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_slice_last_dim(device, memory_strategy, layout):
    """Test slicing on last dimension with all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.slice(tt_input, [0, 0, 0, 0], [1, 1, 128, 64])
    torch_output = torch_input[:, :, :, :64]

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_slice_height(device, memory_strategy, layout):
    """Test slicing on height dimension with all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.slice(tt_input, [0, 0, 0, 0], [1, 1, 64, 128])
    torch_output = torch_input[:, :, :64, :]

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


# =============================================================================
# Concat
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
@pytest.mark.parametrize("dim", [-1, -2])
def test_concat(device, memory_strategy, layout, dim):
    """Test concat with all memory configs and both layouts."""
    shape = [1, 1, 64, 64]
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.concat([tt_a, tt_b], dim=dim)
    torch_output = torch.cat([torch_a, torch_b], dim=dim)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_concat_three_tensors(device, memory_strategy, layout):
    """Test concat with 3 tensors across all memory configs."""
    shape = [1, 1, 64, 64]
    torch_tensors = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(3)]

    mem_config = make_memory_config(memory_strategy, shape)
    tt_tensors = [
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)
        for t in torch_tensors
    ]

    tt_output = ttnn.concat(tt_tensors, dim=-1)
    torch_output = torch.cat(torch_tensors, dim=-1)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


# =============================================================================
# Split
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_split_last_dim(device, memory_strategy, layout):
    """Test split on last dimension with all memory configs."""
    shape = [1, 1, 64, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_outputs = ttnn.split(tt_input, 2, dim=-1)
    torch_outputs = torch.chunk(torch_input, 2, dim=-1)

    for tt_out, torch_out in zip(tt_outputs, torch_outputs):
        tt_result = ttnn.to_torch(tt_out)
        if tt_result.shape != torch_out.shape:
            torch_out = torch_out.reshape(tt_result.shape)
        assert_with_pcc(torch_out, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_split_height_dim(device, memory_strategy, layout):
    """Test split on height dimension with all memory configs."""
    shape = [1, 1, 128, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_outputs = ttnn.split(tt_input, 2, dim=-2)
    torch_outputs = torch.chunk(torch_input, 2, dim=-2)

    for tt_out, torch_out in zip(tt_outputs, torch_outputs):
        tt_result = ttnn.to_torch(tt_out)
        if tt_result.shape != torch_out.shape:
            torch_out = torch_out.reshape(tt_result.shape)
        assert_with_pcc(torch_out, tt_result, 0.9999)
