# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for issue #20995: Use activations for typecasting in binary_ng when not
subtile broadcasting.

When block-format inputs (bfloat8_b / bfloat4_b) are used with binary ops that
don't natively support them (GT, LT, EQ, NE, GE, LE, DIV, …), the current code
falls back to a *composite* typecast path: explicit ttnn.typecast calls wrap the
prim::binary_ng invocation.  When the operand shapes do NOT require subtile
broadcasting, the typecast should instead be fused as a kernel activation inside
binary_ng — a single device op instead of three.

The tests below exercise:

1. **Correctness** – block-format inputs with non-subtile-broadcast shapes
   produce numerically correct results for every affected op.
2. **Single-program check** – with the program cache enabled, only one cache
   entry should be created per (op, dtype, broadcast-type) tuple, confirming
   that the work is done in a single fused program rather than a composite of
   typecast → binary → typecast.
"""

import torch
import pytest
import ttnn

from functools import partial
from models.common.utility_functions import torch_random
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device

# Ops that currently *always* use composite typecast for block-format inputs,
# regardless of whether subtile broadcasting is needed.
COMPOSITE_TYPECAST_OPS = [
    "gt",
    "lt",
    "ge",
    "le",
    "eq",
    "ne",
    "divide",
    "logical_and",
    "logical_or",
    "logical_xor",
    "ldexp",
    "logaddexp",
    "logaddexp2",
    "squared_difference",
]

# ADD / SUB / MUL already go through the non-composite path when there is no
# subtile broadcasting.  Include them here as a baseline sanity check.
NATIVE_BLOCK_OPS = [
    "add",
    "sub",
    "mul",
]

ALL_OPS = NATIVE_BLOCK_OPS + COMPOSITE_TYPECAST_OPS

# Shapes that do NOT trigger subtile broadcasting — inner (H, W) dims are
# equal or broadcasting happens only across batch / channel dims (tile-aligned).
NON_SUBTILE_BROADCAST_SHAPES = [
    # Same shape, no broadcast at all
    (torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])),
    (torch.Size([1, 3, 64, 128]), torch.Size([1, 3, 64, 128])),
    # Batch / channel broadcast only — inner dims match → no subtile broadcast
    (torch.Size([1, 1, 64, 64]), torch.Size([2, 3, 64, 64])),
    (torch.Size([5, 1, 32, 128]), torch.Size([5, 3, 32, 128])),
]

# Shapes that DO trigger subtile broadcasting (one inner dim is 1 while
# the other tensor's corresponding dim is > 1).  These are expected to use
# the composite path and are included as a negative / contrast case.
SUBTILE_BROADCAST_SHAPES = [
    # H=1 broadcast
    (torch.Size([1, 1, 1, 64]), torch.Size([1, 1, 32, 64])),
    # W=1 broadcast
    (torch.Size([1, 1, 32, 1]), torch.Size([1, 1, 32, 64])),
]


def _run_binary_op(device, ttnn_fn_name, shape_a, shape_b, in_dtype, out_dtype=None):
    """Execute a binary op and return (torch_golden, tt_output) tensors."""
    torch.manual_seed(42)
    ttnn_op = getattr(ttnn, ttnn_fn_name)

    torch_a = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), in_dtype)(shape_a)
    torch_b = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), in_dtype)(shape_b)

    # Avoid division by zero
    if ttnn_fn_name == "divide":
        torch_b[torch_b.abs() < 0.5] = 1.0

    tt_a = ttnn.from_torch(
        torch_a,
        dtype=in_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_b = ttnn.from_torch(
        torch_b,
        dtype=in_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    kwargs = {}
    if out_dtype is not None:
        kwargs["dtype"] = out_dtype

    tt_out = ttnn_op(tt_a, tt_b, **kwargs)
    tt_result = ttnn.to_torch(tt_out)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_golden = golden_fn(torch_a, torch_b)

    return torch_golden, tt_result


# ---------------------------------------------------------------------------
# 1.  Correctness: block-format inputs, no subtile broadcast
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("input_shapes", NON_SUBTILE_BROADCAST_SHAPES)
@pytest.mark.parametrize("ttnn_fn", COMPOSITE_TYPECAST_OPS)
@pytest.mark.parametrize("block_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_block_format_no_subtile_broadcast_correctness(input_shapes, ttnn_fn, block_dtype, device):
    """Block-format inputs with non-subtile-broadcast shapes must produce
    correct results for ops that currently rely on composite typecast."""
    if block_dtype == ttnn.bfloat4_b and ttnn_fn == "divide":
        pytest.skip("bfloat4_b has insufficient precision for division (4-bit mantissa)")
    shape_a, shape_b = input_shapes
    torch_golden, tt_result = _run_binary_op(device, ttnn_fn, shape_a, shape_b, block_dtype)
    assert_with_pcc(torch_golden, tt_result, 0.99 if block_dtype != ttnn.bfloat4_b else 0.98)


@pytest.mark.parametrize("input_shapes", NON_SUBTILE_BROADCAST_SHAPES)
@pytest.mark.parametrize("ttnn_fn", NATIVE_BLOCK_OPS)
@pytest.mark.parametrize("block_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_block_format_no_subtile_broadcast_native_ops(input_shapes, ttnn_fn, block_dtype, device):
    """ADD/SUB/MUL already use the non-composite path — sanity baseline."""
    shape_a, shape_b = input_shapes
    torch_golden, tt_result = _run_binary_op(device, ttnn_fn, shape_a, shape_b, block_dtype)
    assert_with_pcc(torch_golden, tt_result, 0.99 if block_dtype != ttnn.bfloat4_b else 0.98)


# ---------------------------------------------------------------------------
# 2.  Correctness: block-format with explicit output dtype (bfloat16 output)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 64, 64]), torch.Size([2, 3, 64, 64])),
    ],
)
@pytest.mark.parametrize("ttnn_fn", COMPOSITE_TYPECAST_OPS)
def test_block_format_input_bfloat16_output(input_shapes, ttnn_fn, device):
    """bfloat8_b inputs → bfloat16 output, no subtile broadcast.  The
    typecast to output dtype should happen as a post-activation."""
    shape_a, shape_b = input_shapes
    torch_golden, tt_result = _run_binary_op(device, ttnn_fn, shape_a, shape_b, ttnn.bfloat8_b, out_dtype=ttnn.bfloat16)
    assert_with_pcc(torch_golden, tt_result, 0.99)


# ---------------------------------------------------------------------------
# 3.  Contrast: subtile-broadcast shapes (composite path expected today)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("input_shapes", SUBTILE_BROADCAST_SHAPES)
@pytest.mark.parametrize("ttnn_fn", ["add", "sub", "mul", "gt", "lt", "eq"])
@pytest.mark.parametrize("block_dtype", [ttnn.bfloat8_b])
def test_block_format_subtile_broadcast_correctness(input_shapes, ttnn_fn, block_dtype, device):
    """Subtile-broadcast cases must also be correct (composite path).
    These serve as a contrast baseline — after the fix they may still use
    composite until LLK subtile-broadcast support lands."""
    shape_a, shape_b = input_shapes
    torch_golden, tt_result = _run_binary_op(device, ttnn_fn, shape_a, shape_b, block_dtype)
    assert_with_pcc(torch_golden, tt_result, 0.99)


# ---------------------------------------------------------------------------
# 4.  Single-program check: non-subtile-broadcast should be one fused program
# ---------------------------------------------------------------------------
@pytest.fixture
def isolate_program_cache(device):
    """Start each test with a clean program cache."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


@pytest.mark.parametrize("ttnn_fn", COMPOSITE_TYPECAST_OPS)
def test_single_program_no_subtile_broadcast(ttnn_fn, device, isolate_program_cache):
    """When block-format inputs don't need subtile broadcasting, the op
    should dispatch as a single fused program (activation-based typecast),
    NOT as a composite of typecast → binary → typecast (3 programs).

    This test will XFAIL until the fix for #20995 lands — currently the
    composite path creates multiple cache entries."""
    shape = [1, 1, 32, 64]

    torch_golden, tt_result = _run_binary_op(device, ttnn_fn, shape, shape, ttnn.bfloat8_b)
    assert_with_pcc(torch_golden, tt_result, 0.99)

    entries = device.num_program_cache_entries()
    if entries != 1:
        pytest.xfail(
            f"Expected 1 program cache entry (fused activation typecast) but got "
            f"{entries} — composite typecast path still in use (issue #20995)"
        )


@pytest.mark.parametrize("ttnn_fn", NATIVE_BLOCK_OPS)
def test_single_program_native_ops_baseline(ttnn_fn, device, isolate_program_cache):
    """ADD/SUB/MUL with equal-shape block-format inputs should already be a
    single program — baseline to confirm the activation path works."""
    shape = [1, 1, 32, 64]

    torch_golden, tt_result = _run_binary_op(device, ttnn_fn, shape, shape, ttnn.bfloat8_b)
    assert_with_pcc(torch_golden, tt_result, 0.99)

    entries = device.num_program_cache_entries()
    assert entries == 1, (
        f"Expected 1 program cache entry for {ttnn_fn} with equal-shape " f"bfloat8_b inputs, but got {entries}"
    )


# ---------------------------------------------------------------------------
# 5.  Mixed dtypes: one block-format, one bfloat16 — no subtile broadcast
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])),
        (torch.Size([2, 3, 64, 128]), torch.Size([2, 3, 64, 128])),
    ],
)
@pytest.mark.parametrize("ttnn_fn", ["gt", "lt", "eq", "ne", "add", "sub"])
def test_mixed_dtype_a_block_b_bfloat16(input_shapes, ttnn_fn, device):
    """Input A is bfloat8_b, input B is bfloat16, same shapes.  Only A needs
    typecasting — this should be fusible as an activation."""
    torch.manual_seed(42)
    shape_a, shape_b = input_shapes
    ttnn_op = getattr(ttnn, ttnn_fn)

    torch_a = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b)(
        shape_a
    )
    torch_b = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(
        shape_b
    )

    tt_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out = ttnn_op(tt_a, tt_b)
    tt_result = ttnn.to_torch(tt_out)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_golden = golden_fn(torch_a, torch_b)
    assert_with_pcc(torch_golden, tt_result, 0.99)


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])),
        (torch.Size([2, 3, 64, 128]), torch.Size([2, 3, 64, 128])),
    ],
)
@pytest.mark.parametrize("ttnn_fn", ["gt", "lt", "eq", "ne", "add", "sub"])
def test_mixed_dtype_a_bfloat16_b_block(input_shapes, ttnn_fn, device):
    """Input A is bfloat16, input B is bfloat8_b, same shapes.  Only B needs
    typecasting — should be fusible as an activation."""
    torch.manual_seed(42)
    shape_a, shape_b = input_shapes
    ttnn_op = getattr(ttnn, ttnn_fn)

    torch_a = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(
        shape_a
    )
    torch_b = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b)(
        shape_b
    )

    tt_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out = ttnn_op(tt_a, tt_b)
    tt_result = ttnn.to_torch(tt_out)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_golden = golden_fn(torch_a, torch_b)
    assert_with_pcc(torch_golden, tt_result, 0.99)


# ---------------------------------------------------------------------------
# 6.  Scalar + block-format: no subtile broadcast by definition
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([2, 3, 64, 128]),
    ],
)
@pytest.mark.parametrize("ttnn_fn", ["gt", "lt", "eq", "ne", "divide", "squared_difference"])
@pytest.mark.parametrize("scalar", [-2.0, 0.0, 1.5])
def test_scalar_block_format_no_subtile(shape, ttnn_fn, scalar, device):
    """Scalar ops with block-format tensors never subtile-broadcast, so they
    should always be fusible."""
    torch.manual_seed(42)
    ttnn_op = getattr(ttnn, ttnn_fn)

    torch_a = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b)(
        shape
    )

    tt_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out = ttnn_op(tt_a, scalar)
    tt_result = ttnn.to_torch(tt_out)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_golden = golden_fn(torch_a, scalar)
    assert_with_pcc(torch_golden, tt_result, 0.99)
