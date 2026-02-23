"""Find exact rounding rule by checking every remainder value."""
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

# Per-remainder statistics
# For each 5-bit remainder value (0-31): count how many times HW rounds up vs truncates
from collections import defaultdict

rem_stats = defaultdict(lambda: {"round_up": 0, "truncate": 0, "both_zero": 0, "error": 0})

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

        trunc_man = min(man_before, 7)

        if hw_man < 0:
            rem_stats[rem]["error"] += 1
        elif hw_man == trunc_man and (hw_man == 0 and man_before == 0):
            rem_stats[rem]["both_zero"] += 1
        elif hw_man == trunc_man:
            rem_stats[rem]["truncate"] += 1
        elif hw_man == trunc_man + 1:
            rem_stats[rem]["round_up"] += 1
        else:
            rem_stats[rem]["error"] += 1

print("Remainder | Truncate | Round Up | Both Zero | Error | HW behavior")
print("-" * 75)
for rem in range(32):
    s = rem_stats[rem]
    total = s["truncate"] + s["round_up"] + s["both_zero"]
    if total == 0:
        continue
    if s["round_up"] == 0:
        behavior = "TRUNCATE"
    elif s["truncate"] == 0:
        behavior = "ROUND UP"
    else:
        behavior = f"MIXED (trunc={s['truncate']}, rhu={s['round_up']})"
    print(
        f"  rem={rem:2d}   | {s['truncate']:8d} | {s['round_up']:8d} | {s['both_zero']:9d} | {s['error']:5d} | {behavior}"
    )
