"""Test hypothesis: HW does Float16_b -> BFP8_b (round-half-up) -> BFP4_b (truncate)."""
import struct
import torch
import ttnn


def infer_bfp4_mantissa(val_f32, shared_exp_raw):
    """Given an unpacked float and shared exponent, infer the 3-bit BFP4 mantissa."""
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


def two_step_bfp4(man_full_8, exp_diff):
    """Simulate two-step: Float16_b -> BFP8 (round-half-up) -> BFP4 (truncate)."""
    # Step 0: align
    man_aligned = man_full_8 >> exp_diff

    # Step 1: Float16_b -> BFP8 (8-bit mantissa -> 7-bit mantissa, round-half-up)
    man_bfp8 = man_aligned >> 1
    rem_bfp8 = man_aligned & 1
    man_bfp8 = man_bfp8 + rem_bfp8  # round-half-up (tie=1, rem>=tie -> round)
    man_bfp8 = min(man_bfp8, 127)  # clamp to 7 bits

    # Step 2: BFP8 -> BFP4 (7-bit mantissa -> 3-bit mantissa, round-half-up)
    rem_bfp4 = man_bfp8 & 0xF
    man_bfp4 = man_bfp8 >> 4
    if rem_bfp4 >= 8:  # tie = 1 << (4-1) = 8
        man_bfp4 += 1
    man_bfp4 = min(man_bfp4, 7)

    return man_bfp4


torch.manual_seed(0)
torch_input = torch.randn((64, 64), dtype=torch.bfloat16)

# Device typecast
device = ttnn.open_device(device_id=0)
input_tensor = ttnn.from_torch(torch_input, memory_config=ttnn.L1_MEMORY_CONFIG)
input_tensor = ttnn.to_device(input_tensor, device)
input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
input_tensor = ttnn.typecast(input_tensor, dtype=ttnn.bfloat4_b)
actual = ttnn.to_torch(input_tensor)
ttnn.close_device(device)

# Extract bf16 mantissa bits
x_f32 = torch_input.to(torch.float32).contiguous().reshape(-1, 16)
x_bits = x_f32.view(torch.int32)
exp = (x_bits >> 23) & 0xFF
man_7bit = (x_bits >> 16) & 0x7F
is_zero = exp == 0

shared_exp_raw = exp.max(dim=-1, keepdim=True).values
exp_diff_t = (shared_exp_raw - exp).clamp(0, 31)

actual_f32 = actual.to(torch.float32).reshape(-1, 16)

# Count matches
two_step_match = 0
direct_rhu_match = 0
direct_trunc_match = 0
total = 0

for group in range(x_f32.shape[0]):
    se = shared_exp_raw[group].item()
    for i in range(16):
        dev_val = actual_f32[group, i].item()
        hw_man = infer_bfp4_mantissa(dev_val, se)

        e = exp[group, i].item()
        m7 = man_7bit[group, i].item()
        ed = exp_diff_t[group, i].item()

        if e == 0:
            continue

        man_full = (1 << 7) | m7
        ts = two_step_bfp4(man_full, ed)

        # Direct: 8-bit mantissa -> 3-bit with various rounding
        man_aligned = man_full >> ed
        man_before = man_aligned >> 5
        rem = man_aligned & 31
        rhu = min(man_before + (1 if rem >= 16 else 0), 7)
        trunc = min(man_before, 7)

        total += 1
        if hw_man == ts:
            two_step_match += 1
        if hw_man == rhu:
            direct_rhu_match += 1
        if hw_man == trunc:
            direct_trunc_match += 1

        if hw_man != ts:
            print(
                f"  MISMATCH G{group:03d} i{i:02d} | hw={hw_man} two_step={ts} rhu={rhu} trunc={trunc} | "
                f"man_aligned={man_aligned:3d} rem={rem:2d}"
            )

print(f"\n{'='*80}")
print(f"Total elements analyzed: {total}")
print(f"HW matches two-step (BFP8 rhu -> BFP4 trunc): {two_step_match}/{total} ({100*two_step_match/total:.2f}%)")
print(f"HW matches direct round-half-up              : {direct_rhu_match}/{total} ({100*direct_rhu_match/total:.2f}%)")
print(
    f"HW matches direct truncation                 : {direct_trunc_match}/{total} ({100*direct_trunc_match/total:.2f}%)"
)

if two_step_match == total:
    print("\n*** PERFECT MATCH: HW uses two-step conversion ***")
    print("  Step 1: Float16_b -> BFP8_b (round-half-up at 1-bit boundary)")
    print("  Step 2: BFP8_b -> BFP4_b (truncate 7-bit mantissa to 3-bit)")
