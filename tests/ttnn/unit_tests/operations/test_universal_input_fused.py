# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for Fused OPs (#31715, excluding matmul).
Tests: softmax, layernorm, groupnorm, rmsnorm, batch_norm
"""

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device

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


# =============================================================================
# Softmax
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_softmax_tile(device, memory_strategy, dim):
    """Test softmax with TILE layout across all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.softmax(tt_input, dim=dim)
    torch_output = torch.softmax(torch_input.float(), dim=dim).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_softmax_rm(device, memory_strategy):
    """Test softmax with ROW_MAJOR layout across all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.softmax(tt_input, dim=-1)
    torch_output = torch.softmax(torch_input.float(), dim=-1).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


# =============================================================================
# Layer Norm
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_layernorm_tile(device, memory_strategy):
    """Test layernorm with TILE layout across all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.layer_norm(tt_input)
    torch_output = torch.nn.functional.layer_norm(torch_input.float(), [shape[-1]]).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_layernorm_rm(device, memory_strategy):
    """Test layernorm with ROW_MAJOR layout across all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.layer_norm(tt_input)
    torch_output = torch.nn.functional.layer_norm(torch_input.float(), [shape[-1]]).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_layernorm_with_weight_bias(device, memory_strategy):
    """Test layernorm with weight and bias across all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.randn([shape[-1]], dtype=torch.bfloat16)
    torch_bias = torch.randn([shape[-1]], dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    tt_weight = ttnn.from_torch(
        torch_weight.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_bias = ttnn.from_torch(
        torch_bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_output = ttnn.layer_norm(tt_input, weight=tt_weight, bias=tt_bias)
    torch_output = torch.nn.functional.layer_norm(
        torch_input.float(), [shape[-1]], torch_weight.float(), torch_bias.float()
    ).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.99)


# =============================================================================
# RMS Norm
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_rmsnorm_tile(device, memory_strategy):
    """Test rmsnorm with TILE layout across all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.randn([shape[-1]], dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    tt_weight = ttnn.from_torch(
        torch_weight.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_output = ttnn.rms_norm(tt_input, weight=tt_weight)

    # Torch reference
    variance = torch_input.float().pow(2).mean(-1, keepdim=True)
    torch_output = (
        torch_input.float() * torch.rsqrt(variance + 1e-6) * torch_weight.float().reshape(1, 1, 1, -1)
    ).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.98)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_rmsnorm_rm(device, memory_strategy):
    """Test rmsnorm with ROW_MAJOR layout across all memory configs."""
    shape = [1, 1, 128, 128]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.randn([shape[-1]], dtype=torch.bfloat16)

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    tt_weight = ttnn.from_torch(
        torch_weight.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_output = ttnn.rms_norm(tt_input, weight=tt_weight)

    variance = torch_input.float().pow(2).mean(-1, keepdim=True)
    torch_output = (
        torch_input.float() * torch.rsqrt(variance + 1e-6) * torch_weight.float().reshape(1, 1, 1, -1)
    ).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.98)


# =============================================================================
# Group Norm
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_groupnorm_tile(device, memory_strategy):
    """Test groupnorm with TILE layout across all memory configs."""
    N, C, H, W = 1, 64, 32, 32
    num_groups = 8
    shape = [N, C, H, W]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # Reshape for ttnn: [N, 1, N*H*W, C] is common for groupnorm
    # Actually groupnorm expects [N, 1, H*W, C] typically in NHWC
    torch_input_nhwc = torch_input.permute(0, 2, 3, 1).reshape(N, 1, H * W, C)

    mem_config = make_memory_config(memory_strategy, [N, 1, H * W, C])
    tt_input = ttnn.from_torch(
        torch_input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.group_norm(tt_input, num_groups=num_groups)
    torch_output = torch.nn.functional.group_norm(torch_input.float(), num_groups).bfloat16()
    torch_output_nhwc = torch_output.permute(0, 2, 3, 1).reshape(N, 1, H * W, C)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output_nhwc, tt_result, 0.98)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_groupnorm_rm(device, memory_strategy):
    """Test groupnorm with ROW_MAJOR layout across all memory configs."""
    N, C, H, W = 1, 64, 32, 32
    num_groups = 8
    shape = [N, C, H, W]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1).reshape(N, 1, H * W, C)

    mem_config = make_memory_config(memory_strategy, [N, 1, H * W, C])
    tt_input = ttnn.from_torch(
        torch_input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    tt_output = ttnn.group_norm(tt_input, num_groups=num_groups)
    torch_output = torch.nn.functional.group_norm(torch_input.float(), num_groups).bfloat16()
    torch_output_nhwc = torch_output.permute(0, 2, 3, 1).reshape(N, 1, H * W, C)

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output_nhwc, tt_result, 0.98)


# =============================================================================
# Batch Norm
# =============================================================================


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_batchnorm_tile(device, memory_strategy):
    """Test batch_norm with TILE layout across all memory configs."""
    N, C, H, W = 1, 32, 32, 32
    shape = [N, C, H, W]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_mean = torch.randn([C], dtype=torch.bfloat16)
    torch_var = torch.abs(torch.randn([C], dtype=torch.bfloat16)) + 0.1

    # For ttnn batch_norm, input is typically [N, C, H, W]
    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    tt_mean = ttnn.from_torch(
        torch_mean.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_var = ttnn.from_torch(
        torch_var.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_output = ttnn.batch_norm(tt_input, running_mean=tt_mean, running_var=tt_var)
    torch_output = torch.nn.functional.batch_norm(
        torch_input.float(), torch_mean.float(), torch_var.float(), training=False
    ).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.98)


@pytest.mark.parametrize("memory_strategy", ALL_MEMORY_STRATEGIES)
def test_batchnorm_rm(device, memory_strategy):
    """Test batch_norm with ROW_MAJOR layout across all memory configs."""
    N, C, H, W = 1, 32, 32, 32
    shape = [N, C, H, W]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_mean = torch.randn([C], dtype=torch.bfloat16)
    torch_var = torch.abs(torch.randn([C], dtype=torch.bfloat16)) + 0.1

    mem_config = make_memory_config(memory_strategy, shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    tt_mean = ttnn.from_torch(
        torch_mean.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_var = ttnn.from_torch(
        torch_var.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_output = ttnn.batch_norm(tt_input, running_mean=tt_mean, running_var=tt_var)
    torch_output = torch.nn.functional.batch_norm(
        torch_input.float(), torch_mean.float(), torch_var.float(), training=False
    ).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_result, 0.98)
