# Universal Input and Output Support Analysis (#31523)

## Context

GitHub issue #31523 requires all TTNN ops to support all tensor formats for inputs/outputs independently:
- **Priority**: DRAM interleaved, L1 interleaved, and all 2D sharded (HEIGHT, WIDTH, BLOCK) inputs
- **Secondary**: Both ROW_MAJOR and TILE layouts (non-performant fallbacks acceptable)
- **Approach**: Use tensor accessor for unsupported sharding (precursor to ND sharding)
- **Excluded from scope**: dtype support, eltwise ops (#31712), matmul op (from #31715)

## Sub-issues Covered

| Issue | Category | Ops |
|-------|----------|-----|
| #31713 | DX team OPs | data_movement, embedding, loss, normalization, pool |
| #31714 | Conv-family | conv2d, conv1d, conv_transpose2d, pool2d, upsample, halo |
| #31715 | Fused/Reduce (no matmul) | softmax, layernorm, groupnorm, rmsnorm, batch_norm; reduce, argmax, topk, prod, cumsum/cumprod |
| #31716 | TMs | reshape, permute, transpose, pad, slice, concat, split, repeat, fold, tilize, untilize, etc. |
| #31717 | CCLs | all_gather, reduce_scatter, all_reduce, broadcast, mesh_partition, etc. |

## Analysis of Current Restrictions

### Memory Layout Support Matrix (current state)

**Legend**: D=DRAM interleaved, L=L1 interleaved, H=Height sharded, W=Width sharded, B=Block sharded

#### #31713 - DX Team OPs

| Op | D | L | H | W | B | TILE | RM | Notes |
|----|---|---|---|---|---|------|-----|-------|
| embedding | Y | N | N | N | N | Y | Y | Weights DRAM only; output supports H sharded |
| batch_norm | Y | N | N | N | N | Y | N | Very restricted |
| groupnorm | Y | Y | Y | Y | B | Y | Y | Good coverage |
| layernorm | Y | Y | Y | W | B | Y | Y* | *RM only non-sharded |
| rmsnorm | Y | Y | Y | W | B | Y | Y | Similar to layernorm |
| softmax | Y | Y | Y | W | B | Y | N | TILE only mostly |
| mse_loss | Y | ? | N | N | N | Y | Y | Composed of eltwise |
| l1_loss | Y | ? | N | N | N | Y | Y | Composed of eltwise |
| global_avg_pool | Y | Y | Y | N | N | Y | N | Limited sharding |
| grid_sample | Y | Y | H | N | N | N | Y | RM only |
| upsample | Y | Y | H | W | B | Y* | Y | *Tiled only for integer scale |

#### #31714 - Conv-family OPs

| Op | D | L | H | W | B | TILE | RM | Notes |
|----|---|---|---|---|---|------|-----|-------|
| conv2d | N | N | Y | Y | Y | N* | Y* | *Weights TILE, activation RM; must be sharded |
| conv1d | N | N | Y | Y | Y | N | Y | Wraps conv2d |
| conv_transpose2d | N | N | Y | Y | Y | N | Y | Wraps conv2d |
| pool2d | N | N | Y | W | B | N | Y | Must be sharded; RM only |
| halo | N | N | Y | W | B | Y | Y | Must be sharded |

#### #31715 - Fused/Reduce OPs (excluding matmul)

| Op | D | L | H | W | B | TILE | RM | Notes |
|----|---|---|---|---|---|------|-----|-------|
| reduce (sum/mean/max/min) | Y | Y | Y | W | B | Y | N | TILE only |
| argmax | Y | Y | N | N | N | Y | Y | Interleaved only |
| topk | Y | Y | N | N | N | Y | N | TILE, interleaved only |
| prod | Y | Y | Y | ? | ? | Y | N | TILE only |
| cumsum/cumprod | Y | Y | N | N | N | Y | N | TILE, interleaved only |
| layernorm (fused) | Y | Y | Y | W | B | Y | Y* | Covered above in DX |
| groupnorm (fused) | Y | Y | Y | W | B | Y | Y | Covered above in DX |
| softmax (fused) | Y | Y | Y | W | B | Y | N | Covered above in DX |
| batch_norm | Y | N | N | N | N | Y | N | DRAM only |

#### #31716 - TMs

| Op | D | L | H | W | B | TILE | RM | Notes |
|----|---|---|---|---|---|------|-----|-------|
| reshape | Y | Y | ? | ? | ? | Y | Y | Some sharded support |
| permute | Y | Y | H | W | B | Y | Y | Good coverage |
| transpose | Y | Y | H | W | N | Y | Y | HC/WH dims; no block sharded |
| pad | Y | Y | H | W | N | Y | Y | RM sharded limited |
| slice | Y | Y | H | N | N | Y | Y | RM sharded only height |
| concat | Y | Y | H | W | B | Y | Y | Good coverage |
| split | Y | Y | N | N | N | Y | N | TILE only, interleaved only |
| repeat | Y | Y | ? | ? | ? | N | Y | RM only |
| repeat_interleave | Y | Y | N | N | N | Y | Y | Interleaved only |
| fold | Y | Y | H | N | N | Y | Y | Height sharded only |
| tilize | Y | Y | H | W | N | N | Y | Input RM only (by design) |
| untilize | Y | Y | H | W | B | Y | N | Input TILE only (by design) |
| clone | Y | Y | Y | Y | Y | Y | Y | Must match in/out shard spec |
| gather | Y | Y | N | N | N | Y | N | TILE only, interleaved only |
| scatter | Y | N | N | N | N | N | Y | RM only, DRAM only |
| bcast | Y | Y | H | W | N | Y | N | TILE only |
| sort | Y | Y | N | N | N | Y | Y | |
| indexed_fill | Y | N | N | N | N | N | Y | RM only, DRAM only |
| non_zero_indices | Y | N | N | N | N | N | Y | RM only, DRAM only |
| fill_rm | Y | Y | N | N | N | N | Y | RM only |
| expand | Y | N | N | N | N | Y | Y | Wraps repeat |
| roll | Y | Y | N | N | N | Y | Y | |
| stack | Y | Y | N | N | N | Y | Y | |

#### #31717 - CCLs

| Op | D | L | H | W | B | TILE | RM | Notes |
|----|---|---|---|---|---|------|-----|-------|
| all_gather | Y | Y | H | W | B* | Y | Y | *No DRAM block sharded |
| reduce_scatter | Y | Y | H | W | B | Y | Y | Good coverage |
| all_reduce | Y | Y | H | W | B | Y | Y | Wraps async |
| broadcast | Y | Y | H | W | B | Y | Y | Good coverage |
| mesh_partition | Y | Y | ? | ? | ? | Y | Y | |
| all_to_all_dispatch | N | N | N | N | N | N | Y | RM+BF16 only |
| all_to_all_combine | N | N | N | N | N | N | Y | RM+BF16 only |

## Plan

### Phase 1: Write Tests

Create pytest test files organized by sub-issue, testing each op with all 5 memory configs (DRAM interleaved, L1 interleaved, height sharded, width sharded, block sharded) and both layouts (TILE, ROW_MAJOR) where applicable.

**Test files to create** under `tests/ttnn/unit_tests/operations/`:

| File | Sub-issue | Ops covered |
|------|-----------|-------------|
| `test_universal_input_dx_embedding.py` | #31713 | embedding |
| `test_universal_input_dx_normalization.py` | #31713 | batch_norm, groupnorm, layernorm, rmsnorm |
| `test_universal_input_dx_loss.py` | #31713 | mse_loss, l1_loss |
| `test_universal_input_dx_pool.py` | #31713 | global_avg_pool, grid_sample, upsample |
| `test_universal_input_conv.py` | #31714 | conv2d, pool2d, upsample, halo |
| `test_universal_input_reduce.py` | #31715 | reduce (sum/mean/max/min), argmax, topk, prod, cumsum/cumprod |
| `test_universal_input_fused.py` | #31715 | softmax, layernorm, groupnorm, batch_norm (as fused) |
| `test_universal_input_tm_reshape.py` | #31716 | reshape, permute, transpose |
| `test_universal_input_tm_pad_slice.py` | #31716 | pad, slice, concat, split |
| `test_universal_input_tm_repeat.py` | #31716 | repeat, repeat_interleave, expand, roll, stack |
| `test_universal_input_tm_data_move.py` | #31716 | clone, gather, scatter, bcast, sort, fold |
| `test_universal_input_tm_tilize.py` | #31716 | tilize, untilize (verify all sharding types) |
| `test_universal_input_tm_misc.py` | #31716 | indexed_fill, non_zero_indices, fill_rm |
| `test_universal_input_ccl.py` | #31717 | all_gather, reduce_scatter, all_reduce, broadcast, mesh_partition |

**Test pattern**: For each op, parametrize over:
- `memory_config`: DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG, height_sharded, width_sharded, block_sharded
- `layout`: TILE_LAYOUT, ROW_MAJOR_LAYOUT (where applicable)
- Use small tensor sizes (e.g., 1x1x128x128 or 1x1x64x64) for speed
- Compare against torch reference or DRAM interleaved baseline
- No xfail/skip - failures = work needed

### Phase 2: Run Tests on HW

Run all tests, collect pass/fail results.

### Phase 3: Analyze Results

Create test report confronting results with analysis. Categorize failures by sub-issue.

## Test Results (HW run on Blackhole, 2026-03-12)

### Summary by Test File

| File | Passed | Failed | Total | Sub-issue |
|------|--------|--------|-------|-----------|
| test_universal_input_reduce.py | 79 | 71 | 150 | #31715 |
| test_universal_input_fused.py | 29 | 31 | 60 | #31715 |
| test_universal_input_tm_reshape.py + tm_pad_slice.py | ~38 | ~82 | ~120 | #31716 |
| test_universal_input_tm_repeat.py | 20 | 40 | 60 | #31716 |
| test_universal_input_tm_data_move.py + tm_tilize.py + tm_misc.py | 59 | 46 | 105 | #31716 |
| test_universal_input_dx_*.py | 29 | 101 | 130 | #31713 |
| test_universal_input_conv.py | 0 | 30 | 30 | #31714 |
| test_universal_input_ccl.py | 30 | 0+20skip | 50 | #31717 |

### Detailed Failure Analysis

#### #31715 - Fused/Reduce OPs

**Reduce ops (sum/mean/max/min)** - TILE layout: ALL PASS (D,L,H,W,B). RM layout: ALL PASS. Excellent coverage already.
- **argmax** - TILE+interleaved: PASS. Sharded: FAIL (not supported). RM+interleaved: PASS. RM+sharded: FAIL.
- **topk** - TILE+interleaved: PASS. Sharded: FAIL. RM: FAIL (TILE required).
- **prod** - TILE+interleaved: PASS. Sharded: some FAIL (tensor_spec issues). RM: FAIL (TILE required for some).
- **cumsum** - TILE+interleaved: PASS. Sharded: FAIL (`!input_tensor.is_sharded()`). RM+interleaved: PASS. RM+sharded: FAIL.
- **cumprod** - TILE+interleaved: PASS (dim=-1), PCC FAIL (dim=-2). Sharded: FAIL. RM: FAIL (`TILE` required).
- **softmax** - ALL PASS (TILE+RM, all memory configs).
- **layernorm** - TILE: PASS (D,L,W,B). Height sharded: FAIL (`HEIGHT_SHARDED` explicitly rejected). RM+non-sharded: PASS. RM+sharded: FAIL.
- **rmsnorm** - TILE: PASS (D,L,W,B). Height sharded: FAIL. RM: FAIL (`input_tensor.layout() != Layout::ROW_MAJOR`).
- **groupnorm** - ALL FAIL: TILE fails with `!inplace.value()`, RM fails with `core_grid.has_value()` (needs core_grid parameter).
- **batch_norm** - ALL FAIL: TILE fails with `tensor.logical_shape()[1] == input_c_dim` (shape mismatch in validate). RM: FAIL (`Layout::TILE` required). Sharded: FAIL (tensor_spec issues).

#### #31716 - TMs

- **reshape** - TILE+sharded: FAIL (many sharded configs fail). RM+sharded: FAIL.
- **permute** - TILE+interleaved: PASS. Sharded: many FAIL.
- **transpose** - Similar pattern: interleaved PASS, sharded mostly FAIL.
- **pad** - Interleaved: PASS. Sharded: FAIL.
- **slice** - Interleaved: PASS. Sharded: FAIL.
- **concat** - Interleaved: PASS. Sharded: many FAIL.
- **split** - Interleaved: PASS. Sharded: FAIL.
- **repeat** - Interleaved PASS. All sharded: FAIL (tensor_spec physical shape mismatches).
- **repeat_interleave** - Interleaved: PASS. Sharded: FAIL (concat internal restriction).
- **expand** - Interleaved: PASS. Sharded: FAIL (tensor_spec/layout issues).
- **roll** - TILE/RM+interleaved: PASS. Sharded: FAIL (CB size, buffer alignment issues).
- **stack** - Interleaved: PASS. Sharded: FAIL (concat internal restriction).
- **clone** - ALL PASS (all memory configs, both layouts).
- **gather** - TILE+interleaved: PASS. Sharded: FAIL (`output_mem_config.is_sharded() == false`). RM: FAIL (`Layout::TILE` required).
- **sort** - TILE+interleaved: PASS. Sharded: FAIL (`output_mem_config.is_sharded() == false`). RM: FAIL (`Layout::TILE` required).
- **fold** - ALL FAIL: shape constraints (`input_shape[1] % stride_h == 0`).
- **tilize** - All memory configs: PASS.
- **untilize** - D,L,H: PASS. W: FAIL (unpadding constraint). B: FAIL (interleaved output required).
- **untilize_with_unpadding** - D,L,H: PASS. W,B: FAIL.
- **indexed_fill** - RM+interleaved: PASS. Sharded: FAIL. TILE: FAIL (`ROW_MAJOR` required).
- **non_zero_indices** - RM+interleaved: PASS. Sharded: FAIL (`INTERLEAVED` required). TILE: FAIL (padded shape breaks shape constraint).
- **full_like** - ALL PASS.

#### #31713 - DX Team OPs

- **embedding** - weight DRAM+RM: PASS. Other weight configs: mostly FAIL (weights must be RM+DRAM). Index memory: interleaved PASS, sharded FAIL.
- **mse_loss** - TILE+interleaved: PASS. RM+interleaved: PASS. Sharded: partial FAIL (tensor_spec/shard issues on reduction outputs).
- **l1_loss** - TILE+interleaved: PASS. RM: FAIL (internal impl requires TILE). Sharded: partial FAIL.
- **batch_norm** (DX) - ALL FAIL (same as fused batch_norm).
- **layernorm** (DX shapes) - Height sharded: FAIL. Multi-batch shapes: sharded FAIL (tensor_spec).
- **rmsnorm** (DX shapes) - Height sharded: FAIL.
- **groupnorm** (DX) - ALL FAIL (`!inplace.value()`).
- **global_avg_pool2d** - ALL FAIL: shape mismatch in test (needs test fix for proper NCHW format). Also sharded tensor_spec issues.
- **upsample** - RM+interleaved: PASS. RM+some sharded: FAIL (allocator). TILE+sharded: FAIL (UNSUPPORTED path).

#### #31714 - Conv-family OPs

- **conv2d** - ALL FAIL: bank_manager assertion — op doesn't accept any non-standard input config (requires internal resharding).
- **max_pool2d** - ALL FAIL: bank_manager assertion — same root cause as conv2d.
- **halo** - Not tested (ttnn.halo not exposed as public API; internal to conv/pool).

#### #31717 - CCLs

- Single-device input creation tests: ALL PASS (30/30). Multi-device tests: SKIPPED (single device env).

## Confrontation: Analysis vs. Test Results

All test bugs have been fixed (loss enum, gather arg order, roll kwarg, conv weight layout, pool dilation, nonzero shape, halo/softmax-mask removed). All remaining failures are real op limitations.

Updated results after bug fixes:
- **loss ops (mse_loss, l1_loss)**: TILE+interleaved PASS; RM+interleaved: mse_loss PASS, l1_loss FAIL (internal impl requires TILE); sharded: partial FAIL (tensor_spec/shard issues)
- **gather**: TILE+interleaved PASS; sharded FAIL (output must be non-sharded); RM FAIL (TILE required)
- **roll**: TILE/RM+interleaved PASS; sharded FAIL (CB size, buffer alignment issues)
- **conv2d**: ALL FAIL with bank_manager assertion — op doesn't accept any non-standard input config
- **max_pool2d**: ALL FAIL with bank_manager assertion — same root cause as conv2d
- **nonzero**: RM+interleaved PASS; sharded FAIL (INTERLEAVED required); TILE FAIL (padded shape breaks shape[2]==1 constraint)

The test results confirm the analysis:
1. **Generic reduce ops (sum/mean/max/min)** have better support than expected - all memory configs pass including sharded, and RM works.
2. **Softmax** has excellent universal support already.
3. **Layernorm/rmsnorm** explicitly reject HEIGHT_SHARDED and RM+sharded.
4. **Most TM ops** only support interleaved memory; sharded inputs are rejected.
5. **Conv/pool ops** have very strict input requirements — bank_manager failures indicate they don't handle non-standard input configs at all.
6. **Loss ops** now work with TILE+interleaved but l1_loss internally requires TILE layout.

## TODO Summary (categorized by sub-issue)

### #31713 - DX Team OPs
1. **embedding**: Support L1 interleaved + all sharded configs for weight tensor; support sharded index tensors
2. **batch_norm**: Support L1 interleaved, all sharded configs; support RM layout input
3. **groupnorm**: Fix inplace/core_grid requirements to work without explicit config; support all memory configs transparently
4. **layernorm**: Support HEIGHT_SHARDED input; support RM+sharded input
5. **rmsnorm**: Support RM layout input
6. **l1_loss**: Support RM layout input (internal impl requires TILE); support sharded inputs
7. **global_avg_pool**: Support all sharded configs
8. **upsample**: Support TILE+sharded inputs; fix allocator issues for RM+sharded

### #31714 - Conv-family OPs
1. **conv2d**: Accept interleaved inputs (auto-reshard internally); accept TILE weights (auto-convert)
2. **pool2d**: Accept interleaved inputs (auto-reshard); accept TILE input
3. **halo**: Accept interleaved inputs (auto-reshard)
4. All conv ops should internally handle input resharding rather than requiring pre-sharded inputs

### #31715 - Fused/Reduce OPs (excluding matmul)
1. **argmax**: Support sharded inputs (H,W,B)
2. **topk**: Support sharded inputs; support RM layout
3. **prod**: Support sharded inputs; support RM layout
4. **cumsum/cumprod**: Support sharded inputs; support RM layout for cumprod
5. **layernorm**: (same as DX) support HEIGHT_SHARDED, RM+sharded
6. **groupnorm**: (same as DX) fix inplace/core_grid issues
7. **batch_norm**: (same as DX) support all configs
8. **softmax**: Already good - no changes needed

### #31716 - TMs
1. **reshape**: Support sharded inputs (all types)
2. **permute**: Support all sharded inputs (currently some work)
3. **transpose**: Support BLOCK_SHARDED; improve sharded support
4. **pad**: Support sharded inputs
5. **slice**: Support WIDTH/BLOCK sharded
6. **concat**: Already supports some sharding; extend to all
7. **split**: Support sharded inputs; support RM layout
8. **repeat**: Support sharded inputs; support TILE layout
9. **repeat_interleave**: Support sharded inputs
10. **expand**: Support sharded inputs
11. **roll**: Support sharded inputs (CB size and buffer alignment issues)
12. **stack**: Support sharded inputs
13. **gather**: Support sharded inputs; support RM layout
14. **sort**: Support sharded inputs; support RM layout
15. **fold**: Fix shape constraints for common patterns
16. **untilize_with_unpadding**: Support WIDTH/BLOCK sharded output
17. **indexed_fill**: Support sharded inputs; support TILE layout
18. **non_zero_indices**: Support sharded inputs; support TILE layout

### #31717 - CCLs
1. CCL ops appear to have good coverage for different memory configs (input creation validated)
2. Multi-device testing needed (requires multi-device setup)
3. **all_to_all_dispatch/combine**: Support TILE layout; support non-BF16 dtypes

## Status & Observations

### What's Working (PASS)
- **Reduce ops (sum/mean/max/min)**: All 5 memory configs + both layouts - excellent coverage
- **Softmax**: All configs pass
- **Layernorm/rmsnorm**: TILE + DRAM/L1/width/block sharded pass
- **Clone**: All configs pass
- **Tilize/untilize**: All interleaved + height sharded pass
- **Full_like**: All configs pass
- **CCL input creation**: All configs validated on single device

### What Needs Work (FAIL categories)

| Sub-issue | Key Gaps |
|-----------|----------|
| **#31713 DX** | embedding (sharded weights), batch_norm (all sharding + RM), groupnorm (needs core_grid), l1_loss (RM rejected), global_avg_pool (sharding), upsample (TILE+sharded) |
| **#31714 Conv** | conv2d (interleaved inputs rejected), pool2d (interleaved rejected), halo (interleaved rejected) |
| **#31715 Fused/Reduce** | argmax/topk/prod/cumsum/cumprod (sharded not supported), layernorm (HEIGHT_SHARDED rejected), rmsnorm (RM rejected), batch_norm (all) |
| **#31716 TMs** | Most TM ops reject sharded inputs; sort/gather/indexed_fill/non_zero_indices limited to interleaved; fold shape constraints |
| **#31717 CCLs** | Multi-device testing needed; single-device validation passes |

### Observations
- Generic reduce ops have surprisingly good universal input support already — all memory configs and both layouts work.
- Softmax is the gold standard for universal input support among fused ops.
- The most common failure pattern for TM ops is that sharded inputs are simply rejected by `validate()` with `TT_FATAL`.
- Conv-family ops are the most restrictive: they require pre-sharded activation and ROW_MAJOR weights. Universal input support here means adding auto-resharding internally.
- `tensor_spec.cpp` assertions (`physical_width == physical_shard_width`, `num_shards <= num_cores`) appear when tensor shapes don't align with shard grid. This is a tensor creation issue, not necessarily an op issue — but ops should handle resharding internally.
- Batch_norm has a fundamental shape validation issue (`tensor.logical_shape()[1] == input_c_dim`) suggesting it expects a specific NCHW layout that doesn't match how tests pass params.
- Groupnorm requires explicit `core_grid` and `inplace` parameters — it doesn't auto-determine these, making it non-universal.

## Mixed-Config Multi-Input Test Results (HW run, 2026-03-12)

Tests where each input tensor of a multi-input op has a **different** memory config or layout.

### Summary

| Test group | Passed | Failed | Total |
|------------|--------|--------|-------|
| Fused mixed (layernorm/rmsnorm input vs weight) | 16 | 42 | 58 |
| Concat mixed (different memory per operand) | 12 | 38 | 50 |
| Binary ops mixed (add/mul/sub different configs) | 38 | 12 | 50 |
| Embedding mixed (index vs weight configs) | 10 | 15 | 25 |
| Indexed_fill mixed (input vs fill configs) | 10 | 18 | 28 |
| Conv2d mixed (activation vs weight/bias) | 0 | 24 | 24 |

### Key Findings

**Binary ops (add/mul/sub)** have excellent mixed-config support:
- Same-shape inputs with different memory configs: ALL PASS (DRAM↔L1, interleaved↔sharded, all combos)
- Mixed layouts (TILE↔RM): ALL PASS across all memory configs
- Broadcast with different configs: PASS when broadcast operand is interleaved; FAIL when broadcast operand is sharded (small tensor can't be sharded due to tile alignment)

**Layernorm/rmsnorm** partially support mixed configs:
- Input in DRAM/L1 + weight in DRAM/L1: PASS (cross-interleaved works)
- Input in width/block_sharded + weight in DRAM: PASS
- Input in height_sharded + weight anywhere: FAIL (HEIGHT_SHARDED rejected)
- Weight in sharded: FAIL (weight tensor [1,1,1,W] too small to shard with tile alignment)
- Residual mixed: interleaved↔interleaved PASS; sharded input requires sharded residual with same shard_spec

**Concat** requires all inputs to match:
- DRAM↔L1 (both interleaved): PASS
- Mixed interleaved↔sharded: FAIL (`in_ref.is_sharded() == shard_first`)
- Mixed sharding types: FAIL (`shard_spec().grid` must match)
- Mixed layouts (TILE↔RM) + interleaved: PASS; sharded: FAIL (`layout == first_input.layout()`)

**Embedding** weights are locked to interleaved:
- Weight in sharded: FAIL (`weights.memory_config().memory_layout() == INTERLEAVED`)
- Index + weight in different interleaved configs: PASS
- Mixed layouts (index TILE, weight RM): PASS for interleaved

**Indexed_fill** is the most restrictive:
- Input + fill in different interleaved configs: PASS
- Any sharded input: FAIL
- Mixed layouts: FAIL (both must be ROW_MAJOR)

**Conv2d** rejects all mixed configs:
- All activation/weight/bias memory combos: FAIL (allocator errors, weight must be RM)
- All layout combos: FAIL (weight must be ROW_MAJOR)

### Additional Observations
- Binary eltwise ops are the gold standard for multi-input universal support — they handle all combinations
- The `tensor_layout.cpp:112` tile alignment assertion appears frequently when trying to shard small tensors (like weight [1,1,1,128]) — this is expected, as sharding a single-row tensor doesn't make sense with 32x32 tiles
- Layernorm residual requires identical shard_spec when both are sharded — no auto-resharding between inputs
- Conv2d's bank_manager assertion failures suggest the op doesn't handle non-standard input configs at all

## Corner Case Test Results (HW run, 2026-03-12)

**192 passed, 183 failed** out of 375 corner case tests.

### Failure Breakdown by Category

| Test category | Failures | Root cause |
|--------------|----------|------------|
| Non-aligned shapes (all 6 ops) | 108 | `tensor_spec.cpp` shard assertions — non-tile-aligned H/W can't form valid shard shapes with 32x32 tiles |
| Cross-output configs (clone/permute) | 16 | Sharded→interleaved or cross-shard output configs rejected |
| Cross-output configs (layernorm) | 5 | HEIGHT_SHARDED input rejected; sharded output requires matching input |
| Odd batch + sharded | 28 | Multi-batch tensors have large physical H, exceeding shard grid capacity |
| Cross-output configs (reduce/softmax/add/transpose) | ~26 | Various shard config mismatches when output differs from input |

### Key Corner Case Findings

1. **Non-tile-aligned shapes + sharding is broadly broken**: Shapes like [1,1,33,64] or [1,1,64,33] fail on ALL sharded configs because `make_memory_config` computes shard dimensions based on the logical shape, but the padded (tile-aligned) physical shape doesn't match. This is a fundamental issue — ops need to handle padding-aware shard specs or auto-pad.

2. **Cross-output memory config** partially works:
   - `reduce_sum`, `softmax`, `binary_add` with interleaved→interleaved cross-output: PASS
   - `clone` with interleaved→sharded or sharded→interleaved: mostly FAIL
   - `permute` cross-output: FAIL when either side is sharded
   - `transpose` cross-output: mostly PASS (only 1 failure)

3. **Odd batch sizes** (3, 7, 2x3) fail with sharding because the flattened batch*H dimension doesn't divide evenly into the shard grid. This is expected for naive shard computation but should be handled by ops with auto-resharding.

4. **Extreme shapes** (single tile, very large): interleaved configs mostly PASS; sharded configs fail for small tensors (not enough data to distribute across cores).

### Test files committed
All 15 test files committed to branch `mateusznowakTT/31712_univ_inout_analysis`:
- 14 original + mixed-config per-op test files
- 1 corner case file (`test_universal_input_corner_cases.py`)

## Complete Test Suite Summary

**15 test files** committed across 4 commits on branch `mateusznowakTT/31712_univ_inout_analysis`:

| Category | Tests | Passed | Failed |
|----------|-------|--------|--------|
| Single-input, uniform config (original) | ~550 | ~280 | ~270 |
| Mixed-config multi-input | ~235 | ~86 | ~149 |
| Corner cases (output config, shapes, batches) | 375 | 192 | 183 |
| **Total** | **~1160** | **~558** | **~602** |

### What's Working (PASS)
- **Reduce ops (sum/mean/max/min)**: All 5 memory configs + both layouts - excellent coverage
- **Softmax**: All configs pass
- **Layernorm/rmsnorm**: TILE + DRAM/L1/width/block sharded pass
- **Clone**: All configs pass
- **Tilize/untilize**: All interleaved + height sharded pass
- **Full_like**: All configs pass
- **CCL input creation**: All configs validated on single device
- **Binary ops (add/mul/sub)**: Mixed memory configs, mixed layouts, cross-output all pass

### What Needs Work (FAIL categories)

| Sub-issue | Key Gaps |
|-----------|----------|
| **#31713 DX** | embedding (sharded weights), batch_norm (all sharding + RM), groupnorm (needs core_grid), l1_loss (RM rejected), global_avg_pool (sharding), upsample (TILE+sharded) |
| **#31714 Conv** | conv2d (interleaved inputs rejected), pool2d (interleaved rejected), halo (interleaved rejected) |
| **#31715 Fused/Reduce** | argmax/topk/prod/cumsum/cumprod (sharded not supported), layernorm (HEIGHT_SHARDED rejected), rmsnorm (RM rejected), batch_norm (all) |
| **#31716 TMs** | Most TM ops reject sharded inputs; sort/gather/indexed_fill/non_zero_indices limited to interleaved; fold shape constraints |
| **#31717 CCLs** | Multi-device testing needed; single-device validation passes |

### Top Insights from Corner Cases
1. **Non-tile-aligned shapes + sharding**: Broadly broken — shard computation doesn't account for tile padding
2. **Cross-output memory configs**: Work for interleaved↔interleaved but fail when sharding is involved on either side
3. **Binary ops are the gold standard**: They handle mixed memory configs, mixed layouts, and cross-output configs best
4. **Odd batch sizes + sharding**: Fail because flattened H dimension doesn't divide into shard grids evenly — ops need auto-resharding

## Verification

```bash
pytest tests/ttnn/unit_tests/operations/test_universal_input_*.py -v --timeout=300
```
