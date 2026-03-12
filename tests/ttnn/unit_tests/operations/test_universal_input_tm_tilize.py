# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for TM OPs - Tilize/Untilize (#31716).
These ops by design convert between RM and TILE layouts, so we test
all sharding types for their respective input layouts.
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


# =============================================================================
# Tilize (ROW_MAJOR -> TILE)
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_tilize(device, memory_strategy):
    """Test tilize (RM->TILE) with all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.tilize(tt_input)

    # Verify by reading back
    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_input, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_tilize_with_val_padding(device, memory_strategy):
    """Test tilize_with_val_padding with all memory configs."""
    # Non-tile-aligned input
    shape = [1, 1, 100, 100]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # For sharded configs, we need tile-aligned sizes
    # Use the padded shape for memory config
    padded_shape = [1, 1, 128, 128]
    mem_config = make_memory_config(memory_strategy, padded_shape)

    # First put into RM on DRAM, then reshard if needed
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if memory_strategy not in ("dram", "l1"):
        # For sharded, pad first then shard
        tt_input = ttnn.pad(tt_input, padding=((0, 0), (0, 0), (0, 28), (0, 28)), value=0.0)
        tt_input = ttnn.to_memory_config(tt_input, mem_config)
        tt_output = ttnn.tilize(tt_input)
    else:
        tt_input = ttnn.to_memory_config(tt_input, mem_config)
        tt_output = ttnn.tilize_with_val_padding(tt_input, padded_shape, 0.0)

    tt_result = ttnn.to_torch(tt_output)
    # Check the non-padded region matches
    assert_with_pcc(torch_input, tt_result[:, :, :100, :100], 0.9999)


# =============================================================================
# Untilize (TILE -> ROW_MAJOR)
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_untilize(device, memory_strategy):
    """Test untilize (TILE->RM) with all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.untilize(tt_input)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_input, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_untilize_with_unpadding(device, memory_strategy):
    """Test untilize_with_unpadding with all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    # Unpad to smaller size
    tt_output = ttnn.untilize_with_unpadding(tt_input, [0, 0, 95, 95])

    tt_result = ttnn.to_torch(tt_output)
    torch_output = torch_input[:, :, :96, :96]
    assert_with_pcc(torch_output, tt_result, 0.9999)
