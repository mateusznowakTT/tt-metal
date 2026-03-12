# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for Reduce OPs (#31715, excluding matmul).
Tests: reduce (sum/mean/max/min), argmax, topk, prod, cumsum/cumprod
"""

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device

# --- Helpers ---

TILE_H = 32
TILE_W = 32


def make_memory_config(strategy, shape):
    """Create a sharded memory config for given strategy and tensor shape."""
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
INTERLEAVED_STRATEGIES = ["dram", "l1"]
BOTH_LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]


# =============================================================================
# Reduce sum/mean/max/min
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("reduce_op", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("dim", [-1, -2])
def test_reduce_generic_tile(device, memory_strategy, reduce_op, dim):
    """Test generic reduce ops (sum/mean/max/min) with TILE layout across all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    op_map = {"sum": ttnn.sum, "mean": ttnn.mean, "max": ttnn.max, "min": ttnn.min}
    torch_op_map = {
        "sum": torch.sum,
        "mean": torch.mean,
        "max": lambda x, d: torch.max(x, d).values,
        "min": lambda x, d: torch.min(x, d).values,
    }

    tt_output = op_map[reduce_op](tt_input, dim=dim)
    torch_output = torch_op_map[reduce_op](torch_input, dim)

    tt_result = ttnn.to_torch(tt_output)
    # Reshape to match - reduce ops may keep dims
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("reduce_op", ["sum", "mean", "max", "min"])
def test_reduce_generic_rm(device, memory_strategy, reduce_op):
    """Test generic reduce ops with ROW_MAJOR layout across all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    op_map = {"sum": ttnn.sum, "mean": ttnn.mean, "max": ttnn.max, "min": ttnn.min}
    torch_op_map = {
        "sum": torch.sum,
        "mean": torch.mean,
        "max": lambda x, d: torch.max(x, d).values,
        "min": lambda x, d: torch.min(x, d).values,
    }

    tt_output = op_map[reduce_op](tt_input, dim=-1)
    torch_output = torch_op_map[reduce_op](torch_input, -1)

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.99)


# =============================================================================
# Argmax
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", BOTH_LAYOUTS)
@pytest.mark.parametrize("dim", [-1, -2])
def test_argmax(device, memory_strategy, layout, dim):
    """Test argmax with all memory configs and both layouts."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config)

    tt_output = ttnn.argmax(tt_input, dim=dim)
    torch_output = torch.argmax(torch_input, dim=dim)

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    # argmax returns indices - check exact match where possible, or PCC
    assert_with_pcc(torch_output.float(), tt_result.float(), 0.99)


# =============================================================================
# Topk
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("k", [1, 32])
def test_topk(device, memory_strategy, k):
    """Test topk with all memory configs (TILE layout only - topk requires it)."""
    # topk requires last dim to be power of 2
    shape = [1, 1, 32, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_values, tt_indices = ttnn.topk(tt_input, k)
    torch_values, torch_indices = torch.topk(torch_input, k, dim=-1)

    tt_values_torch = ttnn.to_torch(tt_values)
    if tt_values_torch.shape != torch_values.shape:
        torch_values = torch_values.reshape(tt_values_torch.shape)

    assert_with_pcc(torch_values, tt_values_torch, 0.99)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_topk_rm(device, memory_strategy):
    """Test topk with ROW_MAJOR layout across all memory configs."""
    shape = [1, 1, 32, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_values, tt_indices = ttnn.topk(tt_input, 1)
    torch_values, _ = torch.topk(torch_input, 1, dim=-1)

    tt_values_torch = ttnn.to_torch(tt_values)
    if tt_values_torch.shape != torch_values.shape:
        torch_values = torch_values.reshape(tt_values_torch.shape)

    assert_with_pcc(torch_values, tt_values_torch, 0.99)


# =============================================================================
# Prod
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("all_dimensions", [True, False])
def test_prod(device, memory_strategy, all_dimensions):
    """Test prod with all memory configs."""
    shape = [1, 1, 64, 64]
    # Use values close to 1 to avoid overflow/underflow in product
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 0.01 + 1.0

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    if all_dimensions:
        tt_output = ttnn.prod(tt_input)
        torch_output = torch.prod(torch_input)
    else:
        tt_output = ttnn.prod(tt_input, dim=-1)
        torch_output = torch.prod(torch_input, dim=-1)

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.95)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_prod_rm(device, memory_strategy):
    """Test prod with ROW_MAJOR layout."""
    shape = [1, 1, 64, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 0.01 + 1.0

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.prod(tt_input, dim=-1)
    torch_output = torch.prod(torch_input, dim=-1)

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.95)


# =============================================================================
# Cumsum / Cumprod (accumulation)
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_cumsum_tile(device, memory_strategy, dim):
    """Test cumsum with TILE layout across all memory configs."""
    shape = [1, 1, 64, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.cumsum(tt_input, dim=dim)
    torch_output = torch.cumsum(torch_input, dim=dim)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_cumsum_rm(device, memory_strategy, dim):
    """Test cumsum with ROW_MAJOR layout across all memory configs."""
    shape = [1, 1, 64, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.cumsum(tt_input, dim=dim)
    torch_output = torch.cumsum(torch_input, dim=dim)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_cumprod_tile(device, memory_strategy, dim):
    """Test cumprod with TILE layout across all memory configs."""
    shape = [1, 1, 64, 64]
    # Use values close to 1 to avoid overflow
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 0.01 + 1.0

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.cumprod(tt_input, dim=dim)
    torch_output = torch.cumprod(torch_input, dim=dim)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.95)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_cumprod_rm(device, memory_strategy, dim):
    """Test cumprod with ROW_MAJOR layout across all memory configs."""
    shape = [1, 1, 64, 64]
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 0.01 + 1.0

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.cumprod(tt_input, dim=dim)
    torch_output = torch.cumprod(torch_input, dim=dim)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.95)
