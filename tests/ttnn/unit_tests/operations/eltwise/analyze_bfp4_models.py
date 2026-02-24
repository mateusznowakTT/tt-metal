"""Verify BFP4 packer model using Bfp8_b -> Bfp4_b path (no SFPU kernel, 7-bit mantissa).

The Bfp8_b path ensures:
1. No SFPU kernel interference (path is no-op)
2. 7-bit mantissa in DST (same as Float16_b, unlike Float32 which may use 23-bit)
"""
import struct
import torch
import ttnn


def infer_bfp4_mantissa(val_f32, shared_exp_raw):
    if val_f32 == 0.0:
        return 0
    for man_n in range(1, 8):
        lz_try = 0
        for b in range(1, 3):
            if 1 <= man_n < (1 << (3 - b)):
                lz_try = b
        man_out = ((man_n << lz_try) << 1) & 7
        exp_out = shared_exp_raw - lz_try
        sign = 1 if val_f32 < 0 else 0
        test_bits = (sign << 31) | (exp_out << 23) | (man_out << 20)
        test_val = struct.unpack("f", struct.pack("I", test_bits))[0]
        if abs(test_val - val_f32) < 1e-10:
            return man_n
    return -1


def align_carry(man_full, ed):
    if ed > 0:
        return (man_full + 1) >> ed
    return man_full


def stage_1_rhu(man_8bit):
    return min((man_8bit >> 1) + (man_8bit & 1), 127)


def stage_2_trunc(man_7bit):
    return min(man_7bit >> 4, 7)


def stage_2_rhu(man_7bit):
    rem = man_7bit & 0xF
    man_3bit = man_7bit >> 4
    if rem >= 8:
        man_3bit += 1
    return min(man_3bit, 7)


def direct_trunc(man_8bit):
    return min(man_8bit >> 5, 7)


def direct_rhu(man_8bit):
    rem = man_8bit & 0x1F
    man_3bit = man_8bit >> 5
    if rem >= 16:
        man_3bit += 1
    return min(man_3bit, 7)


MODELS = {
    "2stg_carryAlign_rhu_trunc": lambda mf, ed: stage_2_trunc(stage_1_rhu(align_carry(mf, ed))),
    "2stg_carryAlign_rhu_rhu": lambda mf, ed: stage_2_rhu(stage_1_rhu(align_carry(mf, ed))),
    "1stg_carryAlign_trunc": lambda mf, ed: direct_trunc(align_carry(mf, ed)),
    "1stg_carryAlign_rhu": lambda mf, ed: direct_rhu(align_carry(mf, ed)),
}

device = ttnn.open_device(device_id=0)

for seed in [0, 10, 50, 200, -100]:
    torch.manual_seed(seed)
    torch_input = torch.randn((64, 64), dtype=torch.bfloat16)

    # Path: bf16 -> bfp8_b -> bfp4_b (both typecast ops use no-op SFPU)
    input_tensor = ttnn.from_torch(torch_input, memory_config=ttnn.L1_MEMORY_CONFIG)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
    # First convert to bfp8_b (the host-side packing would differ, so use device typecast)
    input_tensor = ttnn.typecast(input_tensor, dtype=ttnn.bfloat8_b)
    # Read back bfp8_b to get the intermediate values
    bfp8_vals = ttnn.to_torch(input_tensor)
    # Then convert bfp8_b -> bfp4_b
    input_tensor = ttnn.typecast(input_tensor, dtype=ttnn.bfloat4_b)
    actual = ttnn.to_torch(input_tensor)

    # Use bfp8_b intermediate values as the starting point for our model
    # (since the packer converts FROM these bfp8_b values TO bfp4_b)
    x_f32 = bfp8_vals.to(torch.float32).contiguous().reshape(-1, 16)
    x_bits = x_f32.view(torch.int32)
    exp = (x_bits >> 23) & 0xFF
    man_7bit = (x_bits >> 16) & 0x7F
    shared_exp_raw = exp.max(dim=-1, keepdim=True).values
    exp_diff_t = (shared_exp_raw - exp).clamp(0, 31)
    actual_f32_vals = actual.to(torch.float32).reshape(-1, 16)

    matches = {name: 0 for name in MODELS}
    total = 0

    for group in range(x_f32.shape[0]):
        se = shared_exp_raw[group].item()
        for i in range(16):
            e = exp[group, i].item()
            if e == 0:
                continue

            m7 = man_7bit[group, i].item()
            ed = exp_diff_t[group, i].item()
            man_full = (1 << 7) | m7
            dev_val = actual_f32_vals[group, i].item()
            hw_man = infer_bfp4_mantissa(dev_val, se)

            total += 1
            for name, model_fn in MODELS.items():
                if model_fn(man_full, ed) == hw_man:
                    matches[name] += 1

    print(f"\n{'='*80}")
    print(f"Seed={seed}  Total: {total}")
    print(f"{'='*80}")
    print(f"{'Model':<35s} {'Match':>7s} {'Rate':>9s}")
    print(f"{'-'*35} {'-'*7} {'-'*9}")
    for name in MODELS:
        m = matches[name]
        pct = 100 * m / total
        marker = " ***" if m == total else ""
        print(f"{name:<35s} {m:>5d}/{total:<5d} {pct:>7.3f}%{marker}")

ttnn.close_device(device)
