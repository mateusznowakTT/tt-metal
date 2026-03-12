# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for TM OPs - Repeat/RepeatInterleave/Expand/Roll/Stack (#31716).
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
# Repeat
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_repeat_height(device, memory_strategy, layout):
    """Test repeat along height dim with all memory configs."""
    shape = [1, 1, 32, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.repeat(tt_input, ttnn.Shape([1, 1, 2, 1]))
    torch_output = torch_input.repeat(1, 1, 2, 1)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_repeat_width(device, memory_strategy, layout):
    """Test repeat along width dim with all memory configs."""
    shape = [1, 1, 64, 32]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.repeat(tt_input, ttnn.Shape([1, 1, 1, 2]))
    torch_output = torch_input.repeat(1, 1, 1, 2)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


# =============================================================================
# Repeat Interleave
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_repeat_interleave(device, memory_strategy, layout):
    """Test repeat_interleave with all memory configs."""
    shape = [1, 1, 32, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.repeat_interleave(tt_input, repeats=2, dim=-2)
    torch_output = torch_input.repeat_interleave(2, dim=-2)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


# =============================================================================
# Expand
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_expand(device, memory_strategy, layout):
    """Test expand with all memory configs."""
    shape = [1, 1, 1, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.expand(tt_input, [1, 1, 32, 64])
    torch_output = torch_input.expand(1, 1, 32, 64)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


# =============================================================================
# Roll
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_roll(device, memory_strategy, layout):
    """Test roll with all memory configs."""
    shape = [1, 1, 64, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.roll(tt_input, shifts=16, dims=-1)
    torch_output = torch.roll(torch_input, shifts=16, dims=-1)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


# =============================================================================
# Stack
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_stack(device, memory_strategy, layout):
    """Test stack with all memory configs."""
    shape = [1, 1, 32, 64]
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.stack([tt_a, tt_b], dim=0)
    torch_output = torch.stack([torch_a, torch_b], dim=0)

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)
    assert_with_pcc(torch_output, tt_result, 0.9999)
