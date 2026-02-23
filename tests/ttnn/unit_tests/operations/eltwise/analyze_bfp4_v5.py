"""Deep dive into rem=30 cases: what distinguishes truncate vs round-up?"""
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
is_zero = exp == 0
shared_exp_raw = exp.max(dim=-1, keepdim=True).values
exp_diff_t = (shared_exp_raw - exp).clamp(0, 31)
actual_f32 = actual.to(torch.float32).reshape(-1, 16)

# Analyze rem=30 and rem=31 in detail
print("=" * 100)
print("Detailed analysis of rem=30 and rem=31 cases")
print("=" * 100)
print()

# Test hypothesis: step 2 uses round-half-up, and apparent truncation is due to clamping
print("rem=30 cases:")
print(
    f"  {'Group':>5} {'Idx':>3} | {'man_aligned':>11} {'man_before':>10} {'rem':>3} | "
    f"{'hw_man':>6} {'trunc':>5} {'rhu':>3} | {'classification':>14} | {'two_step_trunc':>14} {'two_step_rhu':>12}"
)
print("-" * 120)

rem30_by_manbefore = {}
rem31_by_manbefore = {}

for group in range(x_f32.shape[0]):
    se = shared_exp_raw[group].item()
    for i in range(16):
        e = exp[group, i].item()
        m7 = man_7bit[group, i].item()
        ed = exp_diff_t[group, i].item()

        if e == 0:
            continue

        man_full = (1 << 7) | m7
        man_aligned = man_full >> ed
        man_before = man_aligned >> 5
        rem = man_aligned & 31

        if rem not in (30, 31):
            continue

        dev_val = actual_f32[group, i].item()
        hw_man = infer_bfp4_mantissa(dev_val, se)
        trunc_man = min(man_before, 7)

        # Two-step model with truncation in step 2
        man_bfp8 = man_aligned >> 1
        rem_bfp8 = man_aligned & 1
        man_bfp8 = man_bfp8 + rem_bfp8  # step 1: rhu
        man_bfp8 = min(man_bfp8, 127)
        rem_bfp4_trunc = man_bfp8 & 0xF
        ts_trunc = min(man_bfp8 >> 4, 7)

        # Two-step model with rhu in step 2
        ts_rhu = min((man_bfp8 >> 4) + (1 if (man_bfp8 & 0xF) >= 8 else 0), 7)

        if hw_man == trunc_man:
            cls = "TRUNCATE"
        elif hw_man == trunc_man + 1:
            cls = "ROUND_UP"
        else:
            cls = "OTHER"

        target = rem30_by_manbefore if rem == 30 else rem31_by_manbefore
        if man_before not in target:
            target[man_before] = {"trunc": 0, "rhu": 0, "other": 0, "both_zero": 0}
        if hw_man == 0 and man_before == 0:
            target[man_before]["both_zero"] += 1
        elif cls == "TRUNCATE":
            target[man_before]["trunc"] += 1
        elif cls == "ROUND_UP":
            target[man_before]["rhu"] += 1
        else:
            target[man_before]["other"] += 1

        if rem == 30 and man_before > 0:
            print(
                f"  {group:5d} {i:3d} | {man_aligned:11d} {man_before:10d} {rem:3d} | "
                f"{hw_man:6d} {trunc_man:5d} {ts_rhu:3d} | {cls:>14} | "
                f"ts_trunc={ts_trunc:2d} ts_rhu={ts_rhu:2d} rem_bfp4={rem_bfp4_trunc:2d}"
            )

print()
print("Summary by man_before for rem=30:")
print(f"  {'man_before':>10} | {'trunc':>5} {'rhu':>5} {'both_zero':>9} {'other':>5}")
for mb in sorted(rem30_by_manbefore.keys()):
    s = rem30_by_manbefore[mb]
    print(f"  {mb:10d} | {s['trunc']:5d} {s['rhu']:5d} {s['both_zero']:9d} {s['other']:5d}")

print()
print("Summary by man_before for rem=31:")
print(f"  {'man_before':>10} | {'trunc':>5} {'rhu':>5} {'both_zero':>9} {'other':>5}")
for mb in sorted(rem31_by_manbefore.keys()):
    s = rem31_by_manbefore[mb]
    print(f"  {mb:10d} | {s['trunc']:5d} {s['rhu']:5d} {s['both_zero']:9d} {s['other']:5d}")

# Final hypothesis test: does two-step with rhu in BOTH steps match if we account for clamping?
print()
print("=" * 100)
print("Test: two-step (step1=rhu, step2=rhu) with clamping - does it explain ALL data?")
print("=" * 100)

total = 0
match_ts_rhu2 = 0
mismatch_details = []

for group in range(x_f32.shape[0]):
    se = shared_exp_raw[group].item()
    for i in range(16):
        e = exp[group, i].item()
        m7 = man_7bit[group, i].item()
        ed = exp_diff_t[group, i].item()

        if e == 0:
            continue

        man_full = (1 << 7) | m7
        man_aligned = man_full >> ed
        man_before = man_aligned >> 5
        rem = man_aligned & 31

        dev_val = actual_f32[group, i].item()
        hw_man = infer_bfp4_mantissa(dev_val, se)

        # Two-step: step1=rhu(1bit), step2=rhu(4bit)
        man_bfp8 = man_aligned >> 1
        rem_bfp8 = man_aligned & 1
        man_bfp8 = man_bfp8 + rem_bfp8
        man_bfp8 = min(man_bfp8, 127)
        rem_bfp4 = man_bfp8 & 0xF
        man_bfp4 = man_bfp8 >> 4
        if rem_bfp4 >= 8:
            man_bfp4 += 1
        man_bfp4 = min(man_bfp4, 7)

        total += 1
        if hw_man == man_bfp4:
            match_ts_rhu2 += 1
        else:
            mismatch_details.append(
                f"  G{group:03d} i{i:02d} | hw={hw_man} pred={man_bfp4} | "
                f"man_aligned={man_aligned:3d} rem={rem:2d} man_before={man_before} "
                f"man_bfp8={man_bfp8} rem_bfp4={rem_bfp4}"
            )

print(f"Match: {match_ts_rhu2}/{total} ({100*match_ts_rhu2/total:.2f}%)")
if mismatch_details:
    print(f"Mismatches ({len(mismatch_details)}):")
    for d in mismatch_details[:50]:
        print(d)
