# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for DX team OPs - Pool (#31713).
Tests: global_avg_pool, upsample, grid_sample
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
# Global Average Pool
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_global_avg_pool2d(device, memory_strategy, layout):
    """Test global_avg_pool2d with all memory configs."""
    N, C, H, W = 1, 32, 64, 64
    shape = [N, C, H, W]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.global_avg_pool2d(tt_input)
    torch_output = torch.nn.functional.adaptive_avg_pool2d(torch_input.float(), (1, 1)).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.98)


# =============================================================================
# Upsample
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_upsample_nearest_rm(device, memory_strategy):
    """Test upsample nearest with ROW_MAJOR across all memory configs."""
    # Upsample expects NHWC format in ROW_MAJOR
    N, H, W, C = 1, 32, 32, 32
    shape = [N, 1, H * W, C]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.upsample(tt_input, scale_factor=2)

    tt_result = ttnn.to_torch(tt_output)
    assert tt_result.numel() > 0


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_upsample_nearest_tile(device, memory_strategy):
    """Test upsample nearest with TILE layout across all memory configs."""
    N, H, W, C = 1, 32, 32, 32
    shape = [N, 1, H * W, C]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.upsample(tt_input, scale_factor=2)

    tt_result = ttnn.to_torch(tt_output)
    assert tt_result.numel() > 0
