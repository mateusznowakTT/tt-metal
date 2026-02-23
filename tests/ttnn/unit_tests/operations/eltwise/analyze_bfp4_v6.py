"""Check if shifted-out bits from exp_diff alignment explain rem=30 discrepancy."""
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

print("rem=30 cases with full detail (shifted-out bits)")
print("=" * 140)
print(
    f"  {'G':>3} {'i':>2} | {'m7':>3} {'ed':>2} {'man_full':>8} {'man_aligned':>11} "
    f"{'shifted_out':>11} | {'man_before':>10} {'rem':>3} | "
    f"{'hw_man':>6} {'trunc':>5} | {'result':>8}"
)
print("-" * 140)

# Collect data for statistical analysis
trunc_shifted = []
rhu_shifted = []
both_zero_shifted = []

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

        if rem != 30:
            continue

        # Bits shifted out during exp_diff alignment
        shifted_out = man_full & ((1 << ed) - 1) if ed > 0 else 0
        shifted_out_max = (1 << ed) - 1 if ed > 0 else 0

        dev_val = actual_f32[group, i].item()
        hw_man = infer_bfp4_mantissa(dev_val, se)
        trunc_man = min(man_before, 7)

        if hw_man == 0 and man_before == 0:
            result = "BOTH0"
            both_zero_shifted.append((shifted_out, ed, man_full))
        elif hw_man == trunc_man:
            result = "TRUNC"
            trunc_shifted.append((shifted_out, ed, man_full))
        elif hw_man == trunc_man + 1:
            result = "RHU"
            rhu_shifted.append((shifted_out, ed, man_full))
        else:
            result = "OTHER"

        print(
            f"  {group:3d} {i:2d} | {m7:3d} {ed:2d} {man_full:8d} {man_aligned:11d} "
            f"{shifted_out:11d} | {man_before:10d} {rem:3d} | "
            f"{hw_man:6d} {trunc_man:5d} | {result:>8}"
        )

print()
print("=" * 80)
print("Shifted-out bits statistics for rem=30:")
print()
print(f"  TRUNCATE cases ({len(trunc_shifted)}):")
for so, ed, mf in sorted(trunc_shifted):
    so_max = (1 << ed) - 1 if ed > 0 else 0
    print(f"    shifted_out={so:3d} / max={so_max:3d}  ed={ed}  man_full={mf}")

print()
print(f"  ROUND_UP cases ({len(rhu_shifted)}):")
for so, ed, mf in sorted(rhu_shifted):
    so_max = (1 << ed) - 1 if ed > 0 else 0
    print(f"    shifted_out={so:3d} / max={so_max:3d}  ed={ed}  man_full={mf}")

print()
print(f"  BOTH_ZERO cases ({len(both_zero_shifted)}):")
for so, ed, mf in sorted(both_zero_shifted):
    so_max = (1 << ed) - 1 if ed > 0 else 0
    print(f"    shifted_out={so:3d} / max={so_max:3d}  ed={ed}  man_full={mf}")

# Check if shifted_out >= half of max explains the pattern
print()
print("=" * 80)
print("Hypothesis: HW rounds up when shifted_out >= (1 << (ed-1))?")
for label, data in [("TRUNCATE", trunc_shifted), ("ROUND_UP", rhu_shifted), ("BOTH_ZERO", both_zero_shifted)]:
    above_half = sum(1 for so, ed, _ in data if ed > 0 and so >= (1 << (ed - 1)))
    below_half = sum(1 for so, ed, _ in data if ed > 0 and so < (1 << (ed - 1)))
    ed_zero = sum(1 for _, ed, _ in data if ed == 0)
    print(f"  {label:10s}: above_half={above_half}, below_half={below_half}, ed=0={ed_zero}")
