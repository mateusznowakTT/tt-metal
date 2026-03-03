# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Validates DISABLE_SFPLOADMACRO fallback implementations.

Each test covers one SFPU operation group modified in LLK-1241.
Run identically on real WH HW, WH ttsim, and BH ttsim — the device
fixture is transparent to simulation mode.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random

pytestmark = pytest.mark.use_module_device

_SHAPE_4D = (1, 1, 32, 32)
_SHAPE_3D = (1, 32, 32)
_N = 32 * 32


def _to_ttnn(t, dtype, device):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Typecast  (ckernel_sfpu_typecast.h)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "src_tt_dtype, dst_tt_dtype, torch_src_dtype, ref_fn",
    [
        # fp32 → uint16
        (ttnn.float32, ttnn.uint16, torch.float32, lambda x: x.to(torch.int32)),
        # uint16 → bfloat16
        (ttnn.uint16, ttnn.bfloat16, torch.int32, lambda x: x.to(torch.bfloat16)),
        # int32 → bfloat16
        (ttnn.int32, ttnn.bfloat16, torch.int32, lambda x: x.to(torch.bfloat16)),
        # fp32 → bfloat16
        (ttnn.float32, ttnn.bfloat16, torch.float32, lambda x: x.to(torch.bfloat16)),
        # uint16 → fp32
        (ttnn.uint16, ttnn.float32, torch.int32, lambda x: x.to(torch.float32)),
        # int32 → fp32
        (ttnn.int32, ttnn.float32, torch.int32, lambda x: x.to(torch.float32)),
        # uint32 → fp32  (use non-negative int32 values as uint32 proxy)
        (ttnn.uint32, ttnn.float32, torch.int32, lambda x: x.to(torch.float32)),
        # uint32 → bfloat16
        (ttnn.uint32, ttnn.bfloat16, torch.int32, lambda x: x.to(torch.bfloat16)),
        # uint16 → uint32  (zero-extend)
        (ttnn.uint16, ttnn.uint32, torch.int32, lambda x: x),
        # uint32 → uint16  (lower 16 bits)
        (ttnn.uint32, ttnn.uint16, torch.int32, lambda x: x & 0xFFFF),
        # int32 → uint16  (clamp to [0, 65535])
        (ttnn.int32, ttnn.uint16, torch.int32, lambda x: x.clamp(0, 65535)),
    ],
)
def test_typecast(device, src_tt_dtype, dst_tt_dtype, torch_src_dtype, ref_fn):
    torch.manual_seed(0)
    # Small non-negative integers — safe for every dtype combination
    raw = torch.arange(_N, dtype=torch.int32).reshape(_SHAPE_4D) % 100 + 1
    torch_input = raw.to(torch_src_dtype)
    expected = ref_fn(raw.to(torch_src_dtype))

    tt_out = ttnn.typecast(
        _to_ttnn(torch_input, src_tt_dtype, device), dst_tt_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    result = ttnn.to_torch(tt_out)

    assert_with_pcc(expected.to(result.dtype), result, pcc=0.999)


# ──────────────────────────────────────────────────────────────────────────────
# Exponential  (ckernel_sfpu_exp.h)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tt_dtype, torch_dtype",
    [
        (ttnn.bfloat16, torch.bfloat16),
        (ttnn.float32, torch.float32),
    ],
)
def test_exp(device, tt_dtype, torch_dtype):
    torch.manual_seed(0)
    torch_input = (torch.rand(_SHAPE_4D, dtype=torch.float32) * 4 - 2).to(torch_dtype)
    expected = torch.exp(torch_input)

    result = ttnn.to_torch(ttnn.exp(_to_ttnn(torch_input, tt_dtype, device)))

    assert_with_pcc(expected, result, pcc=0.9999)


# ──────────────────────────────────────────────────────────────────────────────
# Reduce max / min  (ckernel_sfpu_reduce.h — init_reduce_max_min / calculate_reduce_max_min)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize(
    "tt_dtype, torch_dtype",
    [
        (ttnn.bfloat16, torch.bfloat16),
        (ttnn.float32, torch.float32),
    ],
)
def test_reduce_max(device, dim, tt_dtype, torch_dtype):
    torch.manual_seed(0)
    torch_input = torch_random(_SHAPE_3D, -100, 100, dtype=torch.bfloat16).to(torch_dtype)
    expected, _ = torch.max(torch_input, dim=dim)

    tt_out = ttnn.max(_to_ttnn(torch_input, tt_dtype, device), dim=dim)
    tt_out = ttnn.to_layout(tt_out, ttnn.TILE_LAYOUT)
    tt_out = ttnn.from_device(tt_out)

    assert_with_pcc(expected, ttnn.to_torch(tt_out))


@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize(
    "tt_dtype, torch_dtype",
    [
        (ttnn.bfloat16, torch.bfloat16),
        (ttnn.float32, torch.float32),
    ],
)
def test_reduce_min(device, dim, tt_dtype, torch_dtype):
    torch.manual_seed(0)
    torch_input = torch_random(_SHAPE_3D, -100, 100, dtype=torch.bfloat16).to(torch_dtype)
    expected, _ = torch.min(torch_input, dim=dim)

    tt_out = ttnn.min(_to_ttnn(torch_input, tt_dtype, device), dim=dim)
    tt_out = ttnn.to_layout(tt_out, ttnn.TILE_LAYOUT)
    tt_out = ttnn.from_device(tt_out)

    assert_with_pcc(expected, ttnn.to_torch(tt_out))


# ──────────────────────────────────────────────────────────────────────────────
# Integer multiply  (ckernel_sfpu_mul_int.h)
# ──────────────────────────────────────────────────────────────────────────────


def test_mul_int(device):
    # Uses ttnn.uint16 to exercise ckernel_sfpu_mul_int.h (tt_llk path).
    # int32 mul routes to ckernel_sfpu_mul_int32.h (tt_metal path, separate work).
    torch.manual_seed(0)
    # Small values so product fits in uint16 [0, 65535]
    a = (torch.arange(_N, dtype=torch.int32).reshape(_SHAPE_4D) % 100) + 1
    b = (torch.arange(_N, dtype=torch.int32).reshape(_SHAPE_4D) % 10) + 1
    expected = (a * b) & 0xFFFF

    result = ttnn.to_torch(ttnn.mul(_to_ttnn(a, ttnn.uint16, device), _to_ttnn(b, ttnn.uint16, device)))

    assert torch.equal(result.to(torch.int32), expected)


# ──────────────────────────────────────────────────────────────────────────────
# Reciprocal  (ckernel_sfpu_recip.h — BH: 7b/8b/24b variants)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tt_dtype, torch_dtype",
    [
        (ttnn.bfloat16, torch.bfloat16),
        (ttnn.float32, torch.float32),
    ],
)
def test_reciprocal(device, tt_dtype, torch_dtype):
    torch.manual_seed(0)
    # Avoid values near zero to prevent inf/large error
    torch_input = (torch.rand(_SHAPE_4D, dtype=torch.float32) + 0.5).to(torch_dtype)
    expected = torch.reciprocal(torch_input)

    result = ttnn.to_torch(ttnn.reciprocal(_to_ttnn(torch_input, tt_dtype, device)))

    assert_with_pcc(expected, result, pcc=0.999)


# ──────────────────────────────────────────────────────────────────────────────
# Transpose  (llk_math_transpose_dest.h — transpose_dest_configure_mop)
# ──────────────────────────────────────────────────────────────────────────────


def test_transpose(device):
    torch.manual_seed(0)
    torch_input = torch.rand(_SHAPE_4D, dtype=torch.bfloat16)
    expected = torch_input.transpose(2, 3)

    result = ttnn.to_torch(ttnn.transpose(_to_ttnn(torch_input, ttnn.bfloat16, device), 2, 3))

    assert torch.equal(result, expected)


# ──────────────────────────────────────────────────────────────────────────────
# Unary max/min float  (ckernel_sfpu_unary_max_min.h — calculate_unary_max_min)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tt_dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
@pytest.mark.parametrize("scalar", [-5.0, 0.0, 3.14])
def test_unary_max_float(device, tt_dtype, torch_dtype, scalar):
    torch.manual_seed(0)
    torch_input = (torch.rand(_SHAPE_4D, dtype=torch.float32) * 20 - 10).to(torch_dtype)
    expected = torch.maximum(torch_input, torch.full(_SHAPE_4D, scalar, dtype=torch_dtype))

    result = ttnn.to_torch(ttnn.maximum(_to_ttnn(torch_input, tt_dtype, device), scalar))

    assert_with_pcc(expected, result, pcc=0.9999)


@pytest.mark.parametrize("tt_dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
@pytest.mark.parametrize("scalar", [-5.0, 0.0, 3.14])
def test_unary_min_float(device, tt_dtype, torch_dtype, scalar):
    torch.manual_seed(0)
    torch_input = (torch.rand(_SHAPE_4D, dtype=torch.float32) * 20 - 10).to(torch_dtype)
    expected = torch.minimum(torch_input, torch.full(_SHAPE_4D, scalar, dtype=torch_dtype))

    result = ttnn.to_torch(ttnn.minimum(_to_ttnn(torch_input, tt_dtype, device), scalar))

    assert_with_pcc(expected, result, pcc=0.9999)


# ──────────────────────────────────────────────────────────────────────────────
# Unary max/min int32  (ckernel_sfpu_unary_max_min.h — calculate_unary_max_min_int32)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tt_dtype, signed", [(ttnn.int32, True), (ttnn.uint32, False)])
@pytest.mark.parametrize("scalar", [-100, 0, 50, 2147483647, -2147483648])
def test_unary_max_int32(device, tt_dtype, signed, scalar):
    torch.manual_seed(0)
    n = _N
    torch_input = torch.linspace(-1000, 1000, n, dtype=torch.int32).reshape(_SHAPE_4D)
    if not signed:
        torch_input = torch_input.abs()
        scalar = abs(scalar)
        if scalar > 2147483647:
            pytest.skip("scalar overflows int32 for uint32 dtype")
    expected = torch.maximum(torch_input, torch.full(_SHAPE_4D, scalar, dtype=torch.int32))

    result = ttnn.to_torch(ttnn.maximum(_to_ttnn(torch_input, tt_dtype, device), scalar)).to(torch.int32)

    assert torch.equal(result, expected)


@pytest.mark.parametrize("tt_dtype, signed", [(ttnn.int32, True), (ttnn.uint32, False)])
@pytest.mark.parametrize("scalar", [-100, 0, 50, 2147483647, -2147483648])
def test_unary_min_int32(device, tt_dtype, signed, scalar):
    torch.manual_seed(0)
    n = _N
    torch_input = torch.linspace(-1000, 1000, n, dtype=torch.int32).reshape(_SHAPE_4D)
    if not signed:
        torch_input = torch_input.abs()
        scalar = abs(scalar)
        if scalar > 2147483647:
            pytest.skip("scalar overflows int32 for uint32 dtype")
    expected = torch.minimum(torch_input, torch.full(_SHAPE_4D, scalar, dtype=torch.int32))

    result = ttnn.to_torch(ttnn.minimum(_to_ttnn(torch_input, tt_dtype, device), scalar)).to(torch.int32)

    assert torch.equal(result, expected)


# ──────────────────────────────────────────────────────────────────────────────
# Binary max/min float  (ckernel_sfpu_binary_max_min.h — calculate_binary_max_min)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tt_dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
def test_binary_max_float(device, tt_dtype, torch_dtype):
    torch.manual_seed(0)
    a = (torch.rand(_SHAPE_4D, dtype=torch.float32) * 20 - 10).to(torch_dtype)
    b = (torch.rand(_SHAPE_4D, dtype=torch.float32) * 20 - 10).to(torch_dtype)
    expected = torch.maximum(a, b)

    result = ttnn.to_torch(ttnn.maximum(_to_ttnn(a, tt_dtype, device), _to_ttnn(b, tt_dtype, device)))

    assert_with_pcc(expected, result, pcc=0.9999)


@pytest.mark.parametrize("tt_dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
def test_binary_min_float(device, tt_dtype, torch_dtype):
    torch.manual_seed(0)
    a = (torch.rand(_SHAPE_4D, dtype=torch.float32) * 20 - 10).to(torch_dtype)
    b = (torch.rand(_SHAPE_4D, dtype=torch.float32) * 20 - 10).to(torch_dtype)
    expected = torch.minimum(a, b)

    result = ttnn.to_torch(ttnn.minimum(_to_ttnn(a, tt_dtype, device), _to_ttnn(b, tt_dtype, device)))

    assert_with_pcc(expected, result, pcc=0.9999)


# ──────────────────────────────────────────────────────────────────────────────
# Binary max/min int32  (ckernel_sfpu_binary_max_min.h — calculate_binary_max_min_int32)
# Tests cover same-sign negative, same-sign positive, and mixed-sign cases.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tt_dtype, low_a, high_a, low_b, high_b",
    [
        (ttnn.int32, -1000, 1000, -1000, 1000),  # mixed signs
        (ttnn.int32, -2000, -1, -2000, -1),  # both negative
        (ttnn.int32, 1, 2000, 1, 2000),  # both positive
        (ttnn.int32, -2147483647, 2147483647, -2147483647, 2147483647),  # full range
        (ttnn.uint32, 0, 2000, 0, 2000),  # unsigned
    ],
)
def test_binary_max_int32(device, tt_dtype, low_a, high_a, low_b, high_b):
    n = _N
    a = torch.linspace(low_a, high_a, n, dtype=torch.int32).reshape(_SHAPE_4D)
    b = torch.linspace(high_b, low_b, n, dtype=torch.int32).reshape(_SHAPE_4D)
    expected = torch.maximum(a, b)

    result = ttnn.to_torch(ttnn.maximum(_to_ttnn(a, tt_dtype, device), _to_ttnn(b, tt_dtype, device))).to(torch.int32)

    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "tt_dtype, low_a, high_a, low_b, high_b",
    [
        (ttnn.int32, -1000, 1000, -1000, 1000),
        (ttnn.int32, -2000, -1, -2000, -1),
        (ttnn.int32, 1, 2000, 1, 2000),
        (ttnn.int32, -2147483647, 2147483647, -2147483647, 2147483647),
        (ttnn.uint32, 0, 2000, 0, 2000),
    ],
)
def test_binary_min_int32(device, tt_dtype, low_a, high_a, low_b, high_b):
    n = _N
    a = torch.linspace(low_a, high_a, n, dtype=torch.int32).reshape(_SHAPE_4D)
    b = torch.linspace(high_b, low_b, n, dtype=torch.int32).reshape(_SHAPE_4D)
    expected = torch.minimum(a, b)

    result = ttnn.to_torch(ttnn.minimum(_to_ttnn(a, tt_dtype, device), _to_ttnn(b, tt_dtype, device))).to(torch.int32)

    assert torch.equal(result, expected)


# ──────────────────────────────────────────────────────────────────────────────
# Int32 multiply  (ckernel_sfpu_mul_int32.h — mul_int32, BH only)
# ──────────────────────────────────────────────────────────────────────────────


def test_mul_int32(device):
    torch.manual_seed(0)
    # Values small enough that the lower 32 bits of the product are deterministic
    a = torch.linspace(-100, 100, _N, dtype=torch.int32).reshape(_SHAPE_4D)
    b = torch.linspace(-50, 50, _N, dtype=torch.int32).reshape(_SHAPE_4D)
    # Reference: lower 32 bits of int64 product
    expected = (a.to(torch.int64) * b.to(torch.int64)).to(torch.int32)

    result = ttnn.to_torch(ttnn.mul(_to_ttnn(a, ttnn.int32, device), _to_ttnn(b, ttnn.int32, device))).to(torch.int32)

    assert torch.equal(result, expected)
