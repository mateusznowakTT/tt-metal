# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for TM OPs - Clone/Gather/Scatter/Bcast/Sort/Fold (#31716).
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
# Clone
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_clone(device, memory_strategy, layout):
    """Test clone with all memory configs and both layouts."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.clone(tt_input, memory_config=mem_config)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_input, tt_result, 0.9999)


# =============================================================================
# Gather
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_gather_tile(device, memory_strategy):
    """Test gather with TILE layout across all memory configs."""
    shape = [1, 1, 32, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    # Index tensor: same shape, indices along last dim
    torch_indices = torch.randint(0, 64, shape, dtype=torch.int32)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    tt_indices = ttnn.from_torch(
        torch_indices, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.gather(tt_input, tt_indices, dim=-1)
    torch_output = torch.gather(torch_input, -1, torch_indices.long())

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_gather_rm(device, memory_strategy):
    """Test gather with ROW_MAJOR layout across all memory configs."""
    shape = [1, 1, 32, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_indices = torch.randint(0, 64, shape, dtype=torch.int32)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    tt_indices = ttnn.from_torch(
        torch_indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.gather(tt_input, tt_indices, dim=-1)
    torch_output = torch.gather(torch_input, -1, torch_indices.long())

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


# =============================================================================
# Bcast (broadcast)
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_bcast_h_tile(device, memory_strategy):
    """Test broadcast along H with TILE layout across all memory configs."""
    shape_a = [1, 1, 128, 64]
    shape_b = [1, 1, 1, 64]
    torch_a = torch.randn(shape_a, dtype=torch.bfloat16)
    torch_b = torch.randn(shape_b, dtype=torch.bfloat16)

    mem_config_a = make_memory_config(memory_strategy, shape_a)
    tt_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config_a
    )
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn.add(tt_a, tt_b)
    torch_output = torch_a + torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_bcast_w_tile(device, memory_strategy):
    """Test broadcast along W with TILE layout across all memory configs."""
    shape_a = [1, 1, 64, 128]
    shape_b = [1, 1, 64, 1]
    torch_a = torch.randn(shape_a, dtype=torch.bfloat16)
    torch_b = torch.randn(shape_b, dtype=torch.bfloat16)

    mem_config_a = make_memory_config(memory_strategy, shape_a)
    tt_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config_a
    )
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn.add(tt_a, tt_b)
    torch_output = torch_a + torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_bcast_rm(device, memory_strategy):
    """Test broadcast with ROW_MAJOR layout across all memory configs."""
    shape_a = [1, 1, 128, 64]
    shape_b = [1, 1, 1, 64]
    torch_a = torch.randn(shape_a, dtype=torch.bfloat16)
    torch_b = torch.randn(shape_b, dtype=torch.bfloat16)

    mem_config_a = make_memory_config(memory_strategy, shape_a)
    tt_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config_a
    )
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_output = ttnn.add(tt_a, tt_b)
    torch_output = torch_a + torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)


# =============================================================================
# Sort
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_sort(device, memory_strategy, layout):
    """Test sort with all memory configs and both layouts."""
    shape = [1, 1, 32, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_values, tt_indices = ttnn.sort(tt_input, dim=-1)
    torch_values, torch_indices = torch.sort(torch_input, dim=-1)

    tt_values_result = ttnn.to_torch(tt_values)
    assert_with_pcc(torch_values, tt_values_result, 0.999)


# =============================================================================
# Fold
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
def test_fold(device, memory_strategy, layout):
    """Test fold with all memory configs."""
    # fold expects: stride_h, stride_w, and input [N, 1, H*W, C] typically
    shape = [1, 1, 128, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.fold(tt_input, stride_h=2, stride_w=2)

    tt_result = ttnn.to_torch(tt_output)
    # Just check it runs without error and produces output
    assert tt_result.numel() > 0


# =============================================================================
# Mixed-config multi-input tests for Bcast (binary ops with different configs)
# =============================================================================

MIXED_CONFIGS = [
    ("dram", "l1"),
    ("l1", "dram"),
    ("dram", "height_sharded"),
    ("height_sharded", "dram"),
    ("l1", "height_sharded"),
    ("dram", "width_sharded"),
    ("width_sharded", "dram"),
    ("dram", "block_sharded"),
    ("height_sharded", "width_sharded"),
    ("width_sharded", "block_sharded"),
]


@pytest.mark.parametrize("mem_a,mem_b", MIXED_CONFIGS)
def test_bcast_add_mixed_memory_tile(device, mem_a, mem_b):
    """Test broadcast add with two inputs in different memory configs (TILE)."""
    shape_a = [1, 1, 128, 128]
    shape_b = [1, 1, 1, 128]
    torch_a = torch.randn(shape_a, dtype=torch.bfloat16)
    torch_b = torch.randn(shape_b, dtype=torch.bfloat16)

    config_a = make_memory_config(mem_a, shape_a)
    config_b = make_memory_config(mem_b, shape_b)

    tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config_a)
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config_b)

    tt_output = ttnn.add(tt_a, tt_b)
    torch_output = torch_a + torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)


@pytest.mark.parametrize("mem_a,mem_b", MIXED_CONFIGS)
def test_bcast_mul_mixed_memory_tile(device, mem_a, mem_b):
    """Test broadcast multiply with two inputs in different memory configs (TILE)."""
    shape_a = [1, 1, 128, 128]
    shape_b = [1, 1, 128, 1]
    torch_a = torch.randn(shape_a, dtype=torch.bfloat16)
    torch_b = torch.randn(shape_b, dtype=torch.bfloat16)

    config_a = make_memory_config(mem_a, shape_a)
    config_b = make_memory_config(mem_b, shape_b)

    tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config_a)
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config_b)

    tt_output = ttnn.mul(tt_a, tt_b)
    torch_output = torch_a * torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)


@pytest.mark.parametrize("mem_a,mem_b", MIXED_CONFIGS)
def test_binary_add_same_shape_mixed_memory(device, mem_a, mem_b):
    """Test element-wise add with same-shape inputs in different memory configs."""
    shape = [1, 1, 128, 128]
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    config_a = make_memory_config(mem_a, shape)
    config_b = make_memory_config(mem_b, shape)

    tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config_a)
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config_b)

    tt_output = ttnn.add(tt_a, tt_b)
    torch_output = torch_a + torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)


@pytest.mark.parametrize(
    "layout_a,layout_b",
    [
        (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_binary_add_mixed_layouts(device, layout_a, layout_b, memory_strategy):
    """Test element-wise add with inputs in different layouts."""
    shape = [1, 1, 128, 128]
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)

    tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=layout_a, device=device, memory_config=mem_config)
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=layout_b, device=device, memory_config=mem_config)

    tt_output = ttnn.add(tt_a, tt_b)
    torch_output = torch_a + torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)


@pytest.mark.parametrize("mem_a,mem_b", MIXED_CONFIGS)
def test_binary_sub_mixed_memory_rm(device, mem_a, mem_b):
    """Test element-wise subtract with RM inputs in different memory configs."""
    shape = [1, 1, 128, 128]
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    config_a = make_memory_config(mem_a, shape)
    config_b = make_memory_config(mem_b, shape)

    tt_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=config_a
    )
    tt_b = ttnn.from_torch(
        torch_b, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=config_b
    )

    tt_output = ttnn.sub(tt_a, tt_b)
    torch_output = torch_a - torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)
