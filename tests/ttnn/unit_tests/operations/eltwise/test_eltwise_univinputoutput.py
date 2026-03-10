# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for universal input/output support in eltwise operations (Issue #31712).

For each combination of memory layouts (interleaved DRAM, interleaved L1, sharded L1)
as inputs/outputs, checks whether eltwise ops (unary, binary, ternary) work or fail.

Based on code analysis:
- Unary: requires input and output memory_layout to match (TT_FATAL in unary_device_operation.cpp:92-95)
- Binary_ng: permissive — any combo of sharded/interleaved for each operand
- Ternary TTS/TST: falls back to TensorAccessor when sharded, should work
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

HEIGHT_SHARD_SHAPE = (64, 256)
HEIGHT_CORE_GRID = ttnn.CoreGrid(y=4, x=1)

BLOCK_SHARD_SHAPE = (512, 256)
BLOCK_CORE_GRID = ttnn.CoreGrid(y=2, x=4)


def make_l1_height_sharded_config():
    return ttnn.create_sharded_memory_config(
        shape=HEIGHT_SHARD_SHAPE,
        core_grid=HEIGHT_CORE_GRID,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def make_l1_block_sharded_config():
    return ttnn.create_sharded_memory_config(
        shape=BLOCK_SHARD_SHAPE,
        core_grid=BLOCK_CORE_GRID,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# ===========================================================================
# Unary tests — known-unsupported cross-memory-layout cases
# ===========================================================================


def test_unary_l1_sharded_input_dram_output(device):
    """
    Unary with L1-sharded input and DRAM-interleaved output.

    Currently UNSUPPORTED: unary_device_operation.cpp requires input and output
    memory_layout to match (TT_FATAL at line 92-95).
    """
    shape = (1, 1, 256, 256)
    l1_sharded = make_l1_height_sharded_config()

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_input = ttnn.to_memory_config(ttnn_input, l1_sharded)

    with pytest.raises(Exception):
        _ = ttnn.relu(ttnn_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_unary_dram_input_l1_sharded_output(device):
    """
    Unary with DRAM-interleaved input and L1-sharded output.

    Currently UNSUPPORTED: same TT_FATAL for mismatched memory_layout.
    """
    shape = (1, 1, 256, 256)
    l1_sharded = make_l1_height_sharded_config()

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    with pytest.raises(Exception):
        _ = ttnn.relu(ttnn_input, memory_config=l1_sharded)


# ===========================================================================
# Binary tests — cross-layout combos that should work via TensorAccessor
# ===========================================================================


def test_binary_sharded_inputs_interleaved_output(device):
    """
    Binary with both inputs L1-sharded and DRAM-interleaved output.

    Binary_ng has permissive validation — each operand can independently be
    sharded or interleaved.  DRAM output disables native L1 sharding and forces
    TensorAccessor fallback, which should work correctly.
    """
    shape = (1, 1, 256, 256)
    l1_sharded = make_l1_height_sharded_config()

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch_a + torch_b

    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_sharded)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_sharded)

    result = ttnn.add(ttnn_a, ttnn_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    assert result.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert result.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)


def test_binary_interleaved_inputs_sharded_output(device):
    """
    Binary with both inputs DRAM-interleaved and L1-sharded output.

    When output is sharded but inputs are not, binary_ng uses TensorAccessor to
    read from interleaved inputs and write to sharded output.
    """
    shape = (1, 1, 256, 256)
    l1_sharded = make_l1_height_sharded_config()

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch_a + torch_b

    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    result = ttnn.add(ttnn_a, ttnn_b, memory_config=l1_sharded)

    assert result.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)


def test_binary_mixed_a_sharded_b_interleaved_sharded_output(device):
    """
    Binary with A L1-sharded, B DRAM-interleaved, output L1-sharded.

    Since A and B have different memory configs, is_native_L1_sharding returns
    false → TensorAccessor fallback handles the mixed case.

    Corner case: checks that the interleaved operand is read correctly even when
    the other operand has a native shard layout on a subset of cores.
    """
    shape = (1, 1, 256, 256)
    l1_sharded = make_l1_height_sharded_config()

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch_a + torch_b

    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_sharded)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    result = ttnn.add(ttnn_a, ttnn_b, memory_config=l1_sharded)

    assert result.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)


def test_binary_mixed_a_sharded_b_interleaved_interleaved_output(device):
    """
    Binary with A L1-sharded, B DRAM-interleaved, output DRAM-interleaved.

    Corner case: output layout differs from both inputs.
    """
    shape = (1, 1, 256, 256)
    l1_sharded = make_l1_height_sharded_config()

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch_a + torch_b

    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_sharded)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    result = ttnn.add(ttnn_a, ttnn_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    assert result.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)


# ===========================================================================
# Binary cross-layout corner cases
# ===========================================================================


@pytest.mark.parametrize(
    "shard_strategy, shard_shape, core_grid",
    [
        # WIDTH sharding — only HEIGHT was tested in the basic cases
        (ttnn.ShardStrategy.WIDTH, (256, 64), ttnn.CoreGrid(y=1, x=4)),
        # BLOCK sharding — tests the 2D core grid layout
        (ttnn.ShardStrategy.BLOCK, (128, 64), ttnn.CoreGrid(y=2, x=4)),
    ],
)
def test_binary_sharded_inputs_interleaved_output_shard_strategies(device, shard_strategy, shard_shape, core_grid):
    """
    Binary with L1-sharded inputs → DRAM-interleaved output for WIDTH and BLOCK strategies.

    Corner case: the original test only used HEIGHT sharding.  WIDTH and BLOCK use
    different core grid layouts that exercise different access patterns in TensorAccessor.
    """
    shape = (1, 1, 256, 256)
    l1_sharded = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=core_grid,
        strategy=shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch_a + torch_b

    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_sharded)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_sharded)

    result = ttnn.add(ttnn_a, ttnn_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    assert result.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)


@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_binary_sharded_to_interleaved_col_major(device, shard_orientation):
    """
    Binary with L1-HEIGHT-sharded inputs (COL_MAJOR orientation) → DRAM output.

    Corner case: COL_MAJOR orientation flips the interpretation of shard height/width
    and can expose bugs in stride/offset calculations in TensorAccessor.
    """
    shape = (1, 1, 256, 256)
    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (64, 256)
    else:
        shard_shape = (256, 64)  # swapped for COL_MAJOR

    l1_sharded = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=HEIGHT_CORE_GRID,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch_a + torch_b

    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_sharded)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_sharded)

    result = ttnn.add(ttnn_a, ttnn_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)


def test_binary_interleaved_to_sharded_uneven_height(device):
    """
    Binary with DRAM-interleaved inputs → L1-HEIGHT-sharded output where tensor height
    is not evenly divisible by the number of cores.

    Corner case: uneven sharding means the last shard has fewer tiles than the others.
    TensorAccessor must handle the boundary core correctly without over-reading.
    Shape (1,1,192,256) across 4 cores → 3 tiles per core except last core gets 0 tiles
    unless padding is applied.  Uses a shape where height/cores leaves a remainder.
    """
    # 192 rows, 4 cores → 48 rows/core = 3 tiles/core (evenly divides here)
    # Use 160 rows → 40 rows/core = 2.5 tiles → needs ceiling: 3 tiles on first cores, 2 on last
    shape = (1, 1, 160, 256)
    # shard_shape height must be multiple of TILE_HEIGHT (32): ceil(160/4/32)*32 = 64
    shard_shape = (64, 256)
    l1_sharded = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=HEIGHT_CORE_GRID,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch_a + torch_b

    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    result = ttnn.add(ttnn_a, ttnn_b, memory_config=l1_sharded)
    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)


# ===========================================================================
# Where (ternary TTS/TST) — BLOCK sharding not yet covered by existing tests
# ===========================================================================


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_where_tts_block_sharding(device, predicate_sharded, true_sharded, out_sharded):
    """
    `ttnn.where(pred, true_tensor, scalar_false)` — TTS variant — with BLOCK sharding.

    ternary_program_factory.get_shard_specs() returns std::nullopt for TTS
    (only TTT is natively sharded). The TensorAccessor fallback is used instead,
    which should handle all memory combos correctly.

    Existing tests only cover HEIGHT/WIDTH sharding for TTS; this adds BLOCK.
    """
    shape = (1, 1, 1024, 1024)
    block_sharded = make_l1_block_sharded_config()
    scalar_false = -1.0

    torch_pred = torch.randint(0, 2, shape, dtype=torch.bfloat16)
    torch_true = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch.where(torch_pred.bool(), torch_true, torch.tensor(scalar_false, dtype=torch.bfloat16))

    pred = ttnn.from_torch(torch_pred, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    true_t = ttnn.from_torch(torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if predicate_sharded:
        pred = ttnn.to_memory_config(pred, block_sharded)
    if true_sharded:
        true_t = ttnn.to_memory_config(true_t, block_sharded)

    out_mem = block_sharded if out_sharded else ttnn.DRAM_MEMORY_CONFIG
    result = ttnn.where(pred, true_t, scalar_false, memory_config=out_mem)

    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_where_tst_block_sharding(device, predicate_sharded, false_sharded, out_sharded):
    """
    `ttnn.where(pred, scalar_true, false_tensor)` — TST variant — with BLOCK sharding.

    Same TensorAccessor fallback path as TTS.  Not covered by existing tests.

    Corner case: verifies the false_tensor (C operand) is read correctly when it is
    the only real tensor input alongside a scalar true value.
    """
    shape = (1, 1, 1024, 1024)
    block_sharded = make_l1_block_sharded_config()
    scalar_true = 2.5

    torch_pred = torch.randint(0, 2, shape, dtype=torch.bfloat16)
    torch_false = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch.where(torch_pred.bool(), torch.tensor(scalar_true, dtype=torch.bfloat16), torch_false)

    pred = ttnn.from_torch(torch_pred, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    false_t = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        pred = ttnn.to_memory_config(pred, block_sharded)
    if false_sharded:
        false_t = ttnn.to_memory_config(false_t, block_sharded)

    out_mem = block_sharded if out_sharded else ttnn.DRAM_MEMORY_CONFIG
    result = ttnn.where(pred, scalar_true, false_t, memory_config=out_mem)

    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)


# ===========================================================================
# Where TTS/TST corner cases
# ===========================================================================


@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_tts_block_sharding_col_major(device, shard_orientation):
    """
    `where(pred, tensor, scalar)` with BLOCK sharding in COL_MAJOR orientation.

    Corner case: COL_MAJOR orientation was not tested in the main parametrized test.
    For BLOCK sharding COL_MAJOR the shard shape interpretation is transposed vs
    ROW_MAJOR, which can expose TensorAccessor stride bugs.
    """
    shape = (1, 1, 1024, 1024)
    # For BLOCK COL_MAJOR: same shard shape but grid traversal order differs
    block_sharded = ttnn.create_sharded_memory_config(
        shape=BLOCK_SHARD_SHAPE,
        core_grid=BLOCK_CORE_GRID,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )
    scalar_false = -1.0

    torch_pred = torch.randint(0, 2, shape, dtype=torch.bfloat16)
    torch_true = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch.where(torch_pred.bool(), torch_true, torch.tensor(scalar_false, dtype=torch.bfloat16))

    pred = ttnn.from_torch(torch_pred, layout=ttnn.TILE_LAYOUT, device=device, memory_config=block_sharded)
    true_t = ttnn.from_torch(torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=block_sharded)

    result = ttnn.where(pred, true_t, scalar_false, memory_config=block_sharded)
    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)


def test_where_tts_sharded_predicate_interleaved_true_sharded_out(device):
    """
    `where(pred_sharded, true_interleaved, scalar)` — sharded predicate, interleaved true tensor.

    Corner case: when predicate is sharded but the true tensor is interleaved, the
    kernel must read the true tensor via TensorAccessor while predicate is already in
    the local CB.  This mixed-locality access is different from the all-sharded path.
    """
    shape = (1, 1, 1024, 1024)
    block_sharded = make_l1_block_sharded_config()
    scalar_false = 0.5

    torch_pred = torch.randint(0, 2, shape, dtype=torch.bfloat16)
    torch_true = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch.where(torch_pred.bool(), torch_true, torch.tensor(scalar_false, dtype=torch.bfloat16))

    pred = ttnn.from_torch(torch_pred, layout=ttnn.TILE_LAYOUT, device=device, memory_config=block_sharded)
    # true tensor deliberately left interleaved
    true_t = ttnn.from_torch(torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    result = ttnn.where(pred, true_t, scalar_false, memory_config=block_sharded)
    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)


def test_where_tst_sharded_predicate_interleaved_false_sharded_out(device):
    """
    `where(pred_sharded, scalar, false_interleaved)` — TST with sharded predicate,
    interleaved false tensor.

    Corner case: symmetric to the TTS case above but for the false path.  Verifies
    that the false (C) operand is read from DRAM while predicate is local.
    """
    shape = (1, 1, 1024, 1024)
    block_sharded = make_l1_block_sharded_config()
    scalar_true = 1.0

    torch_pred = torch.randint(0, 2, shape, dtype=torch.bfloat16)
    torch_false = torch.randn(shape, dtype=torch.bfloat16)
    torch_ref = torch.where(torch_pred.bool(), torch.tensor(scalar_true, dtype=torch.bfloat16), torch_false)

    pred = ttnn.from_torch(torch_pred, layout=ttnn.TILE_LAYOUT, device=device, memory_config=block_sharded)
    # false tensor deliberately left interleaved
    false_t = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    result = ttnn.where(pred, scalar_true, false_t, memory_config=block_sharded)
    assert_with_pcc(torch_ref, ttnn.to_torch(result), pcc=0.9999)
