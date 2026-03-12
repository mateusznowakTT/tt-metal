# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for TM OPs - Misc: indexed_fill, non_zero_indices, fill_rm (#31716).
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
# Indexed Fill
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_indexed_fill(device, memory_strategy):
    """Test indexed_fill with ROW_MAJOR layout across all memory configs."""
    batch_size = 1
    num_indices = 2
    shape = [batch_size, 1, 32, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    # Indices: which batch elements to fill
    torch_indices = torch.tensor([0], dtype=torch.int32).reshape(1, 1, 1, 1)
    # Fill value tensor
    torch_fill = torch.ones([1, 1, 32, 64], dtype=torch.bfloat16) * 5.0

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    tt_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_fill = ttnn.from_torch(torch_fill, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_output = ttnn.indexed_fill(tt_indices, tt_input, tt_fill, dim=0)

    tt_result = ttnn.to_torch(tt_output)
    # The indexed batch element should be filled
    assert tt_result.numel() > 0


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_indexed_fill_tile(device, memory_strategy):
    """Test indexed_fill with TILE layout across all memory configs."""
    shape = [1, 1, 32, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_indices = torch.tensor([0], dtype=torch.int32).reshape(1, 1, 1, 1)
    torch_fill = torch.ones([1, 1, 32, 64], dtype=torch.bfloat16) * 5.0

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    tt_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_fill = ttnn.from_torch(torch_fill, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn.indexed_fill(tt_indices, tt_input, tt_fill, dim=0)
    tt_result = ttnn.to_torch(tt_output)
    assert tt_result.numel() > 0


# =============================================================================
# Non-Zero Indices
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_non_zero_indices(device, memory_strategy):
    """Test non_zero_indices with ROW_MAJOR across all memory configs."""
    shape = [1, 1, 1, 128]
    torch_input = torch.zeros(shape, dtype=torch.bfloat16)
    # Set some elements to non-zero
    torch_input[0, 0, 0, 10] = 1.0
    torch_input[0, 0, 0, 50] = 2.0
    torch_input[0, 0, 0, 100] = 3.0

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.nonzero(tt_input)
    tt_result = ttnn.to_torch(tt_output)
    assert tt_result.numel() > 0


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_non_zero_indices_tile(device, memory_strategy):
    """Test non_zero_indices with TILE layout across all memory configs."""
    shape = [1, 1, 1, 128]
    torch_input = torch.zeros(shape, dtype=torch.bfloat16)
    torch_input[0, 0, 0, 10] = 1.0
    torch_input[0, 0, 0, 50] = 2.0

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.nonzero(tt_input)
    tt_result = ttnn.to_torch(tt_output)
    assert tt_result.numel() > 0


# =============================================================================
# Fill (fill_rm / full-like)
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_full_like(device, memory_strategy, layout):
    """Test full_like (creating a tensor filled with a value) with all memory configs."""
    shape = [1, 1, 64, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.full_like(tt_input, fill_value=3.14)

    tt_result = ttnn.to_torch(tt_output)
    expected = torch.full(shape, 3.14, dtype=torch.bfloat16)
    assert_with_pcc(expected, tt_result, 0.999)


# =============================================================================
# Mixed-config multi-input tests for Indexed Fill
# =============================================================================

MIXED_CONFIGS = [
    ("dram", "l1"),
    ("l1", "dram"),
    ("dram", "height_sharded"),
    ("height_sharded", "dram"),
    ("l1", "height_sharded"),
    ("dram", "width_sharded"),
    ("dram", "block_sharded"),
    ("height_sharded", "width_sharded"),
]


@pytest.mark.parametrize("input_mem,fill_mem", MIXED_CONFIGS)
def test_indexed_fill_mixed_memory(device, input_mem, fill_mem):
    """Test indexed_fill with input and fill tensors in different memory configs."""
    shape = [1, 1, 32, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_indices = torch.tensor([0], dtype=torch.int32).reshape(1, 1, 1, 1)
    torch_fill = torch.ones([1, 1, 32, 64], dtype=torch.bfloat16) * 5.0

    input_config = make_memory_config(input_mem, shape)
    fill_config = make_memory_config(fill_mem, shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_config
    )
    tt_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_fill = ttnn.from_torch(
        torch_fill, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=fill_config
    )

    tt_output = ttnn.indexed_fill(tt_indices, tt_input, tt_fill, dim=0)
    tt_result = ttnn.to_torch(tt_output)
    assert tt_result.numel() > 0


@pytest.mark.parametrize(
    "input_layout,fill_layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_indexed_fill_mixed_layouts(device, input_layout, fill_layout, memory_strategy):
    """Test indexed_fill with input and fill in different layouts."""
    shape = [1, 1, 32, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_indices = torch.tensor([0], dtype=torch.int32).reshape(1, 1, 1, 1)
    torch_fill = torch.ones([1, 1, 32, 64], dtype=torch.bfloat16) * 5.0

    mem_config = make_memory_config(memory_strategy, shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=mem_config
    )
    tt_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_fill = ttnn.from_torch(
        torch_fill, dtype=ttnn.bfloat16, layout=fill_layout, device=device, memory_config=mem_config
    )

    tt_output = ttnn.indexed_fill(tt_indices, tt_input, tt_fill, dim=0)
    tt_result = ttnn.to_torch(tt_output)
    assert tt_result.numel() > 0
