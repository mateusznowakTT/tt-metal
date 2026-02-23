"""Reverse-engineer HW mantissa values from device output."""
import struct
import torch
import ttnn


def infer_bfp4_mantissa(val_f32, shared_exp_raw):
    """Given an unpacked float and the shared exponent, infer the 3-bit BFP4 mantissa."""
    if val_f32 == 0.0:
        return 0
    bits = struct.unpack("I", struct.pack("f", abs(val_f32)))[0]
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    # From the host unpack: exp_out = shared_exp - shift_cnt, man_out = (man_n << lz << 1) & 7
    # The unpack finds leading zeros in man_n (3-bit field), normalizes, adjusts exponent
    # We need to reverse this:
    # exp_out = shared_exp_raw - lz
    # So lz = shared_exp_raw - exp_out = shared_exp_raw - exp
    lz = shared_exp_raw - exp
    if lz < 0 or lz > 2:
        return -1  # error
    # man_out = ((man_n << lz) << 1) & 7
    # From man_out, reverse: man_n = (man_out | (1 << lz+?) ) ... complex
    # Simpler: just try all 8 mantissa values and see which one unpacks to the given float
    for man_n in range(8):
        if man_n == 0:
            if val_f32 == 0.0:
                return 0
            continue
        # Find leading zeros in 3-bit field
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

# Python simulation (round-half-up, 24-bit mantissa)
x_f32 = torch_input.to(torch.float32).contiguous().reshape(-1, 16)
x_bits = x_f32.view(torch.int32)
sign = (x_bits >> 31) & 1
exp = (x_bits >> 23) & 0xFF
man = x_bits & 0x7FFFFF
is_zero = exp == 0
man_full = torch.where(is_zero, torch.zeros_like(man), man | (1 << 23))
shared_exp_raw = exp.max(dim=-1, keepdim=True).values
exp_diff = (shared_exp_raw - exp).clamp(0, 31)
man_aligned_24 = man_full >> exp_diff

# 24-bit → 3-bit
shift24 = 21
man_n_24 = man_aligned_24 >> shift24
rem_24 = man_aligned_24 & ((1 << shift24) - 1)
tie_24 = 1 << (shift24 - 1)

# Also compute with 8-bit mantissa (bf16 precision)
man_7bit = (x_bits >> 16) & 0x7F  # top 7 bits of bf16 mantissa
man_full_8 = torch.where(is_zero, torch.zeros_like(man_7bit), man_7bit | (1 << 7))
man_aligned_8 = man_full_8 >> exp_diff

shift8 = 5  # 8-3=5
man_n_8 = man_aligned_8 >> shift8
rem_8 = man_aligned_8 & ((1 << shift8) - 1)
tie_8 = 1 << (shift8 - 1)

# Rounding variants
man_rhu_24 = (man_n_24 + (rem_24 >= tie_24).to(torch.int32)).clamp(max=7)
man_trunc_24 = man_n_24.clamp(max=7)
man_rhu_8 = (man_n_8 + (rem_8 >= tie_8).to(torch.int32)).clamp(max=7)
man_trunc_8 = man_n_8.clamp(max=7)
man_rtne_8 = man_n_8.clone()
guard_8 = man_n_8 & 1
man_rtne_8 = man_rtne_8 + ((rem_8 > tie_8) | ((rem_8 == tie_8) & (guard_8 == 1))).to(torch.int32)
man_rtne_8 = man_rtne_8.clamp(max=7)

actual_f32 = actual.to(torch.float32).reshape(-1, 16)

# Infer HW mantissa for each element
print("=" * 140)
print("First 4 groups: comparing mantissa values")
print("=" * 140)

rhu24_match = 0
trunc24_match = 0
rhu8_match = 0
trunc8_match = 0
rtne8_match = 0
total_nonzero = 0

for group in range(x_f32.shape[0]):
    se = shared_exp_raw[group].item()
    for i in range(16):
        dev_val = actual_f32[group, i].item()
        hw_man = infer_bfp4_mantissa(dev_val, se)

        r24 = man_rhu_24[group, i].item()
        t24 = man_trunc_24[group, i].item()
        r8 = man_rhu_8[group, i].item()
        t8 = man_trunc_8[group, i].item()
        re8 = man_rtne_8[group, i].item()

        if hw_man > 0 or r24 > 0:  # skip both-zero
            total_nonzero += 1
            if hw_man == r24:
                rhu24_match += 1
            if hw_man == t24:
                trunc24_match += 1
            if hw_man == r8:
                rhu8_match += 1
            if hw_man == t8:
                trunc8_match += 1
            if hw_man == re8:
                rtne8_match += 1

        if group < 4:
            inp = x_f32[group, i].item()
            raw_e = exp[group, i].item()
            ed = exp_diff[group, i].item()
            rem8v = rem_8[group, i].item()
            tie8v = tie_8
            ok = "OK" if hw_man == r24 else "DIFF"
            print(
                f"  G{group:03d} i{i:02d} | inp={inp:10.5f} exp={raw_e:3d} ed={ed} | "
                f"hw={hw_man} rhu24={r24} trunc24={t24} rhu8={r8} trunc8={t8} rtne8={re8} | "
                f"rem8={rem8v:2d} tie8={tie8v:2d} | {ok}"
            )

print(f"\n{'='*80}")
print(f"Total non-trivial elements: {total_nonzero}")
print(f"HW matches round-half-up (24-bit) : {rhu24_match}/{total_nonzero} ({100*rhu24_match/total_nonzero:.1f}%)")
print(f"HW matches truncation   (24-bit)  : {trunc24_match}/{total_nonzero} ({100*trunc24_match/total_nonzero:.1f}%)")
print(f"HW matches round-half-up (8-bit)  : {rhu8_match}/{total_nonzero} ({100*rhu8_match/total_nonzero:.1f}%)")
print(f"HW matches truncation   (8-bit)   : {trunc8_match}/{total_nonzero} ({100*trunc8_match/total_nonzero:.1f}%)")
print(f"HW matches round-to-even (8-bit)  : {rtne8_match}/{total_nonzero} ({100*rtne8_match/total_nonzero:.1f}%)")
