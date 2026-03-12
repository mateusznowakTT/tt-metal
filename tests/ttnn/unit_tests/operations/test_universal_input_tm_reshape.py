# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for TM OPs - Reshape/Permute/Transpose (#31716).
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
# Reshape
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_reshape_basic(device, memory_strategy, layout):
    """Test reshape with all memory configs and both layouts."""
    shape = [1, 1, 128, 128]
    new_shape = [1, 1, 64, 256]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.reshape(tt_input, new_shape)
    torch_output = torch_input.reshape(new_shape)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_reshape_batch(device, memory_strategy, layout):
    """Test reshape changing batch dims with all memory configs."""
    shape = [2, 2, 64, 64]
    new_shape = [4, 1, 64, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.reshape(tt_input, new_shape)
    torch_output = torch_input.reshape(new_shape)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


# =============================================================================
# Permute
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
@pytest.mark.parametrize(
    "perm",
    [
        (0, 1, 3, 2),  # Swap last two dims
        (0, 2, 1, 3),  # Swap middle dims
        (0, 3, 2, 1),  # Reverse inner dims
    ],
)
def test_permute(device, memory_strategy, layout, perm):
    """Test permute with all memory configs and both layouts."""
    shape = [1, 2, 64, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.permute(tt_input, perm)
    torch_output = torch_input.permute(perm).contiguous()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


# =============================================================================
# Transpose
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_transpose_wh(device, memory_strategy, layout):
    """Test transpose of last two dims (W-H) with all memory configs."""
    shape = [1, 1, 64, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.transpose(tt_input, -2, -1)
    torch_output = torch_input.transpose(-2, -1).contiguous()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_transpose_hc(device, memory_strategy, layout):
    """Test transpose H-C dims with all memory configs."""
    shape = [1, 4, 64, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.transpose(tt_input, 1, 2)
    torch_output = torch_input.transpose(1, 2).contiguous()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_transpose_cn(device, memory_strategy, layout):
    """Test transpose C-N dims with all memory configs."""
    shape = [2, 4, 64, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.transpose(tt_input, 0, 1)
    torch_output = torch_input.transpose(0, 1).contiguous()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)
