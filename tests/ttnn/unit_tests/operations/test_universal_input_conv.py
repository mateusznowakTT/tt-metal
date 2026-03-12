# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for Conv-family OPs (#31714).
Tests: conv2d, pool2d, upsample (conv context), halo

Note: Conv ops have unique requirements - activation must typically be sharded.
This tests whether interleaved inputs are also accepted (or auto-resharded).
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
# Conv2d - input activation memory config
# =============================================================================


@pytest.mark.parametrize("input_memory_strategy", ALL_MEMORY_STRATEGIES)
def test_conv2d_input_memory(device, input_memory_strategy):
    """Test conv2d with activation in different memory configs.
    Conv2d currently requires sharded activation - testing if interleaved is accepted."""
    batch_size = 1
    in_channels = 32
    out_channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 3

    torch_input = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16)

    # NHWC for ttnn conv
    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, in_channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    input_mem = make_memory_config(input_memory_strategy, flat_shape)
    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mem
    )
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=None,
        kernel_size=(kernel_size, kernel_size),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=input_h,
        input_width=input_w,
    )

    torch_output = torch.nn.functional.conv2d(torch_input.float(), torch_weight.float(), padding=1).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.97)


# =============================================================================
# Conv2d - weight memory config
# =============================================================================


@pytest.mark.parametrize("weight_memory_strategy", ALL_MEMORY_STRATEGIES)
def test_conv2d_weight_memory(device, weight_memory_strategy):
    """Test conv2d with weight in different memory configs."""
    batch_size = 1
    in_channels = 32
    out_channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 3

    torch_input = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, in_channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    tt_input = ttnn.from_torch(torch_input_flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    weight_shape = torch_weight.shape
    weight_mem = make_memory_config(
        weight_memory_strategy, [1, 1, out_channels, in_channels * kernel_size * kernel_size]
    )
    tt_weight = ttnn.from_torch(
        torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=weight_mem
    )

    tt_output = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=None,
        kernel_size=(kernel_size, kernel_size),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=input_h,
        input_width=input_w,
    )

    torch_output = torch.nn.functional.conv2d(torch_input.float(), torch_weight.float(), padding=1).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.97)


# =============================================================================
# Pool2d (max_pool2d) - input memory config
# =============================================================================


@pytest.mark.parametrize("input_memory_strategy", ALL_MEMORY_STRATEGIES)
def test_max_pool2d_input_memory(device, input_memory_strategy):
    """Test max_pool2d with input in different memory configs.
    Pool2d currently requires sharded ROW_MAJOR input."""
    batch_size = 1
    channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 2
    stride = 2

    torch_input = torch.randn(batch_size, channels, input_h, input_w, dtype=torch.bfloat16)

    # NHWC format for ttnn
    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    input_mem = make_memory_config(input_memory_strategy, flat_shape)
    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mem
    )

    tt_output = ttnn.max_pool2d(
        input_tensor=tt_input,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(0, 0),
    )

    torch_output = torch.nn.functional.max_pool2d(torch_input.float(), kernel_size, stride).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("input_memory_strategy", ALL_MEMORY_STRATEGIES)
def test_max_pool2d_tile_input(device, input_memory_strategy):
    """Test max_pool2d with TILE layout input (currently requires ROW_MAJOR)."""
    batch_size = 1
    channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 2
    stride = 2

    torch_input = torch.randn(batch_size, channels, input_h, input_w, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    input_mem = make_memory_config(input_memory_strategy, flat_shape)
    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mem
    )

    tt_output = ttnn.max_pool2d(
        input_tensor=tt_input,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(0, 0),
    )

    torch_output = torch.nn.functional.max_pool2d(torch_input.float(), kernel_size, stride).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.99)


# =============================================================================
# Halo - input memory config
# =============================================================================


@pytest.mark.parametrize("input_memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_halo_input_memory(device, input_memory_strategy, layout):
    """Test halo op with input in different memory configs.
    Halo currently requires sharded input."""
    batch_size = 1
    channels = 32
    input_h = 8
    input_w = 8

    torch_input = torch.randn(batch_size, channels, input_h, input_w, dtype=torch.bfloat16)

    # NHWC format
    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    input_mem = make_memory_config(input_memory_strategy, flat_shape)
    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=input_mem
    )

    # Halo needs sliding window config
    sliding_window_config = ttnn.SlidingWindowConfig(
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        window_h=3,
        window_w=3,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        num_cores_nhw=1,
    )

    tt_output = ttnn.halo(
        tt_input,
        sliding_window_config,
        pad_val=0,
    )

    tt_result = ttnn.to_torch(tt_output)
    assert tt_result.numel() > 0


# =============================================================================
# Conv2d - mixed memory configs between activation and weight
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


@pytest.mark.parametrize("activation_mem,weight_mem", MIXED_CONFIGS)
def test_conv2d_mixed_activation_weight_memory(device, activation_mem, weight_mem):
    """Test conv2d with activation and weight in different memory configs."""
    batch_size = 1
    in_channels = 32
    out_channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 3

    torch_input = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, in_channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    act_config = make_memory_config(activation_mem, flat_shape)
    weight_flat_shape = [1, 1, out_channels, in_channels * kernel_size * kernel_size]
    weight_config = make_memory_config(weight_mem, weight_flat_shape)

    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=act_config
    )
    tt_weight = ttnn.from_torch(
        torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=weight_config
    )

    tt_output = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=None,
        kernel_size=(kernel_size, kernel_size),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=input_h,
        input_width=input_w,
    )

    torch_output = torch.nn.functional.conv2d(torch_input.float(), torch_weight.float(), padding=1).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.97)


@pytest.mark.parametrize(
    "act_layout,weight_layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        (ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    ],
)
def test_conv2d_mixed_layouts(device, act_layout, weight_layout):
    """Test conv2d with activation and weight in different layouts."""
    batch_size = 1
    in_channels = 32
    out_channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 3

    torch_input = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, in_channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    tt_input = ttnn.from_torch(torch_input_flat, dtype=ttnn.bfloat16, layout=act_layout, device=device)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=weight_layout, device=device)

    tt_output = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=None,
        kernel_size=(kernel_size, kernel_size),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=input_h,
        input_width=input_w,
    )

    torch_output = torch.nn.functional.conv2d(torch_input.float(), torch_weight.float(), padding=1).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.97)


@pytest.mark.parametrize("activation_mem,bias_mem", MIXED_CONFIGS[:6])
def test_conv2d_mixed_activation_bias_memory(device, activation_mem, bias_mem):
    """Test conv2d with activation, weight, and bias all in different memory configs."""
    batch_size = 1
    in_channels = 32
    out_channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 3

    torch_input = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16)
    torch_bias = torch.randn(out_channels, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, in_channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    act_config = make_memory_config(activation_mem, flat_shape)
    bias_shape = [1, 1, 1, out_channels]
    bias_config = make_memory_config(bias_mem, bias_shape)

    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=act_config
    )
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_bias = ttnn.from_torch(
        torch_bias.reshape(bias_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=bias_config,
    )

    tt_output = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=tt_bias,
        kernel_size=(kernel_size, kernel_size),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=input_h,
        input_width=input_w,
    )

    torch_output = torch.nn.functional.conv2d(
        torch_input.float(), torch_weight.float(), torch_bias.float(), padding=1
    ).bfloat16()

    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.97)
