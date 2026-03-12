# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Corner case tests for universal input/output support (#31523).
Tests:
1. Output memory config different from input (e.g., input sharded, output DRAM)
2. Non-tile-aligned shapes with various memory configs
3. Very small and large tensors
4. Odd batch sizes and rank combinations
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
# 1. Output memory config different from input
# =============================================================================

OUTPUT_CROSS_CONFIGS = [
    ("dram", "l1"),
    ("l1", "dram"),
    ("height_sharded", "dram"),
    ("height_sharded", "l1"),
    ("width_sharded", "dram"),
    ("block_sharded", "dram"),
    ("dram", "height_sharded"),
    ("dram", "width_sharded"),
    ("dram", "block_sharded"),
    ("height_sharded", "width_sharded"),
]


@pytest.mark.parametrize("input_mem,output_mem", OUTPUT_CROSS_CONFIGS)
def test_reduce_sum_cross_output_config(device, input_mem, output_mem):
    """Test reduce sum with input in one config, output requested in another."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    in_config = make_memory_config(input_mem, shape)
    out_shape = [1, 1, 1, 128]
    out_config = make_memory_config(output_mem, out_shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_config
    )

    tt_output = ttnn.sum(tt_input, dim=-2, memory_config=out_config)
    torch_output = torch.sum(torch_input, dim=-2)

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("input_mem,output_mem", OUTPUT_CROSS_CONFIGS)
def test_softmax_cross_output_config(device, input_mem, output_mem):
    """Test softmax with output memory config different from input."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    in_config = make_memory_config(input_mem, shape)
    out_config = make_memory_config(output_mem, shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_config
    )

    tt_output = ttnn.softmax(tt_input, dim=-1, memory_config=out_config)
    torch_output = torch.softmax(torch_input.float(), dim=-1).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("input_mem,output_mem", OUTPUT_CROSS_CONFIGS)
def test_layernorm_cross_output_config(device, input_mem, output_mem):
    """Test layernorm with output memory config different from input."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    in_config = make_memory_config(input_mem, shape)
    out_config = make_memory_config(output_mem, shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_config
    )

    tt_output = ttnn.layer_norm(tt_input, memory_config=out_config)
    torch_output = torch.nn.functional.layer_norm(torch_input.float(), [shape[-1]]).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("input_mem,output_mem", OUTPUT_CROSS_CONFIGS)
def test_clone_cross_output_config(device, input_mem, output_mem):
    """Test clone with output memory config different from input."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    in_config = make_memory_config(input_mem, shape)
    out_config = make_memory_config(output_mem, shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_config
    )

    tt_output = ttnn.clone(tt_input, memory_config=out_config)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_input, tt_result, 0.9999)


@pytest.mark.parametrize("input_mem,output_mem", OUTPUT_CROSS_CONFIGS)
def test_binary_add_cross_output_config(device, input_mem, output_mem):
    """Test binary add with output in different memory config than inputs."""
    shape = [1, 1, 128, 128]
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    in_config = make_memory_config(input_mem, shape)
    out_config = make_memory_config(output_mem, shape)

    tt_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_config
    )
    tt_b = ttnn.from_torch(
        torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_config
    )

    tt_output = ttnn.add(tt_a, tt_b, memory_config=out_config)
    torch_output = torch_a + torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)


@pytest.mark.parametrize("input_mem,output_mem", OUTPUT_CROSS_CONFIGS)
def test_permute_cross_output_config(device, input_mem, output_mem):
    """Test permute with output in different memory config than input."""
    shape = [1, 2, 64, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    in_config = make_memory_config(input_mem, shape)
    out_shape = [1, 2, 128, 64]
    out_config = make_memory_config(output_mem, out_shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_config
    )

    tt_output = ttnn.permute(tt_input, (0, 1, 3, 2), memory_config=out_config)
    torch_output = torch_input.permute(0, 1, 3, 2).contiguous()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


@pytest.mark.parametrize("input_mem,output_mem", OUTPUT_CROSS_CONFIGS)
def test_transpose_cross_output_config(device, input_mem, output_mem):
    """Test transpose with output in different memory config."""
    shape = [1, 1, 64, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    in_config = make_memory_config(input_mem, shape)
    out_shape = [1, 1, 128, 64]
    out_config = make_memory_config(output_mem, out_shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_config
    )

    tt_output = ttnn.transpose(tt_input, -2, -1, memory_config=out_config)
    torch_output = torch_input.transpose(-2, -1).contiguous()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


# =============================================================================
# 2. Non-tile-aligned shapes
# =============================================================================

NON_ALIGNED_SHAPES = [
    [1, 1, 33, 64],  # H not tile-aligned
    [1, 1, 64, 33],  # W not tile-aligned
    [1, 1, 33, 33],  # Both not aligned
    [1, 1, 1, 128],  # Very thin (1 row)
    [1, 1, 128, 1],  # Very narrow (1 col)
    [1, 1, 17, 65],  # Odd primes
]


@pytest.mark.parametrize("shape", NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_reduce_sum_non_aligned(device, shape, memory_strategy):
    """Test reduce sum with non-tile-aligned shapes."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.sum(tt_input, dim=-1)
    torch_output = torch.sum(torch_input, dim=-1)

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("shape", NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_softmax_non_aligned(device, shape, memory_strategy):
    """Test softmax with non-tile-aligned shapes."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.softmax(tt_input, dim=-1)
    torch_output = torch.softmax(torch_input.float(), dim=-1).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("shape", NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_clone_non_aligned(device, shape, memory_strategy):
    """Test clone with non-tile-aligned shapes."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.clone(tt_input, memory_config=mem_config)
    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_input, tt_result, 0.9999)


@pytest.mark.parametrize("shape", NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_layernorm_non_aligned(device, shape, memory_strategy):
    """Test layernorm with non-tile-aligned shapes."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.layer_norm(tt_input)
    torch_output = torch.nn.functional.layer_norm(torch_input.float(), [shape[-1]]).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("shape", NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_binary_add_non_aligned(device, shape, memory_strategy):
    """Test binary add with non-tile-aligned shapes."""
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    tt_b = ttnn.from_torch(
        torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.add(tt_a, tt_b)
    torch_output = torch_a + torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)


# Non-aligned shapes with ROW_MAJOR layout
@pytest.mark.parametrize("shape", NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_binary_add_non_aligned_rm(device, shape, memory_strategy):
    """Test binary add with non-tile-aligned shapes in RM layout."""
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    tt_b = ttnn.from_torch(
        torch_b, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.add(tt_a, tt_b)
    torch_output = torch_a + torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)


# =============================================================================
# 3. Very small and large tensors
# =============================================================================

EXTREME_SHAPES = [
    [1, 1, 32, 32],  # Exactly 1 tile
    [1, 1, 1, 32],  # Single row, 1 tile wide
    [1, 1, 32, 1],  # Single col, 1 tile tall
    [1, 1, 512, 512],  # Large
    [1, 1, 1024, 128],  # Tall
    [1, 1, 128, 1024],  # Wide
]


@pytest.mark.parametrize("shape", EXTREME_SHAPES)
@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_reduce_sum_extreme_shapes(device, shape, memory_strategy):
    """Test reduce sum with extreme shapes."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.sum(tt_input, dim=-1)
    torch_output = torch.sum(torch_input, dim=-1)

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("shape", EXTREME_SHAPES)
@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_clone_extreme_shapes(device, shape, memory_strategy):
    """Test clone with extreme shapes."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.clone(tt_input, memory_config=mem_config)
    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_input, tt_result, 0.9999)


# =============================================================================
# 4. Odd batch sizes / higher rank
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("batch_shape", [[3, 1, 64, 64], [2, 3, 64, 64], [1, 7, 64, 64]])
def test_reduce_sum_odd_batch(device, memory_strategy, batch_shape):
    """Test reduce sum with non-power-of-2 batch sizes."""
    torch_input = torch.randn(batch_shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, batch_shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.sum(tt_input, dim=-1)
    torch_output = torch.sum(torch_input, dim=-1)

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("batch_shape", [[3, 1, 64, 64], [2, 3, 64, 64], [1, 7, 64, 64]])
def test_softmax_odd_batch(device, memory_strategy, batch_shape):
    """Test softmax with non-power-of-2 batch sizes."""
    torch_input = torch.randn(batch_shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, batch_shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.softmax(tt_input, dim=-1)
    torch_output = torch.softmax(torch_input.float(), dim=-1).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("batch_shape", [[3, 1, 64, 64], [2, 3, 64, 64], [1, 7, 64, 64]])
def test_binary_add_odd_batch(device, memory_strategy, batch_shape):
    """Test binary add with non-power-of-2 batch sizes."""
    torch_a = torch.randn(batch_shape, dtype=torch.bfloat16)
    torch_b = torch.randn(batch_shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, batch_shape)
    tt_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    tt_b = ttnn.from_torch(
        torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.add(tt_a, tt_b)
    torch_output = torch_a + torch_b

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("batch_shape", [[3, 1, 64, 64], [2, 3, 64, 64]])
def test_concat_odd_batch(device, memory_strategy, batch_shape):
    """Test concat with non-power-of-2 batch sizes."""
    torch_a = torch.randn(batch_shape, dtype=torch.bfloat16)
    torch_b = torch.randn(batch_shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, batch_shape)
    tt_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    tt_b = ttnn.from_torch(
        torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.concat([tt_a, tt_b], dim=-1)
    torch_output = torch.cat([torch_a, torch_b], dim=-1)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("batch_shape", [[3, 1, 64, 64], [2, 3, 64, 64]])
def test_transpose_odd_batch(device, memory_strategy, batch_shape):
    """Test transpose WH with non-power-of-2 batch sizes."""
    torch_input = torch.randn(batch_shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, batch_shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.transpose(tt_input, -2, -1)
    torch_output = torch_input.transpose(-2, -1).contiguous()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.9999)
