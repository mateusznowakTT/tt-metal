"""Test model: alignment uses (man_full + 1) >> ed when ed > 0, then two-step conversion."""
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


def model_carry_align(man_full, ed):
    """Alignment: (man_full + 1) >> ed for ed > 0, else man_full."""
    if ed > 0:
        man_aligned = (man_full + 1) >> ed
    else:
        man_aligned = man_full

    # Step 1: 8-bit -> 7-bit round-half-up
    man_bfp8 = man_aligned >> 1
    rem_bfp8 = man_aligned & 1
    man_bfp8 = man_bfp8 + rem_bfp8
    man_bfp8 = min(man_bfp8, 127)

    # Step 2: 7-bit -> 3-bit truncation
    man_bfp4 = man_bfp8 >> 4
    man_bfp4 = min(man_bfp4, 7)
    return man_bfp4


def model_v3_original(man_full, ed):
    """Original v3 two-step: align(truncate), step1(rhu), step2(truncate)."""
    man_aligned = man_full >> ed

    man_bfp8 = man_aligned >> 1
    rem_bfp8 = man_aligned & 1
    man_bfp8 = man_bfp8 + rem_bfp8
    man_bfp8 = min(man_bfp8, 127)

    man_bfp4 = man_bfp8 >> 4
    man_bfp4 = min(man_bfp4, 7)
    return man_bfp4


torch.manual_seed(0)
torch_input = torch.randn((64, 64), dtype=torch.bfloat16)

device = ttnn.open_device(device_id=0)
input_tensor = ttnn.from_torch(torch_input, memory_config=ttnn.L1_MEMORY_CONFIG)
input_tensor = ttnn.to_device(input_tensor, device)
input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
input_tensor = ttnn.typecast(input_tensor, dtype=ttnn.bfloat4_b)
actual = ttnn.to_torch(input_tensor)
ttnn.close_device(device)

x_f32 = torch_input.to(torch.float32).contiguous().reshape(-1, 16)
x_bits = x_f32.view(torch.int32)
exp = (x_bits >> 23) & 0xFF
man_7bit = (x_bits >> 16) & 0x7F
shared_exp_raw = exp.max(dim=-1, keepdim=True).values
exp_diff_t = (shared_exp_raw - exp).clamp(0, 31)
actual_f32 = actual.to(torch.float32).reshape(-1, 16)

total = 0
v3_match = 0
carry_match = 0
mismatches_carry = []

for group in range(x_f32.shape[0]):
    se = shared_exp_raw[group].item()
    for i in range(16):
        e = exp[group, i].item()
        m7 = man_7bit[group, i].item()
        ed = exp_diff_t[group, i].item()

        if e == 0:
            continue

        man_full = (1 << 7) | m7
        dev_val = actual_f32[group, i].item()
        hw_man = infer_bfp4_mantissa(dev_val, se)

        v3 = model_v3_original(man_full, ed)
        carry = model_carry_align(man_full, ed)

        total += 1
        if hw_man == v3:
            v3_match += 1
        if hw_man == carry:
            carry_match += 1
        else:
            man_aligned = man_full >> ed
            shifted_out = man_full & ((1 << ed) - 1) if ed > 0 else 0
            mismatches_carry.append(
                f"  G{group:03d} i{i:02d} | hw={hw_man} carry={carry} v3={v3} | "
                f"m7={m7} ed={ed} man_full={man_full} man_aligned={man_aligned} "
                f"shifted_out={shifted_out}"
            )

print(f"Total: {total}")
print(f"V3 original (align-trunc, step1-rhu, step2-trunc):  {v3_match}/{total} ({100*v3_match/total:.4f}%)")
print(
    f"Carry-align model ((man_full+1)>>ed, step1-rhu, step2-trunc): {carry_match}/{total} ({100*carry_match/total:.4f}%)"
)

if mismatches_carry:
    print(f"\nRemaining mismatches for carry-align model ({len(mismatches_carry)}):")
    for d in mismatches_carry[:30]:
        print(d)
