"""Analyze per-element discrepancy between Python BFP4 simulation and HW packer."""
import torch
import ttnn


def simulate_bfp_quantization(x, man_bits):
    """Python simulation from test file."""
    orig_shape = x.shape
    x_f32 = x.to(torch.float32).contiguous().reshape(-1, 16)
    x_bits = x_f32.view(torch.int32)

    sign = (x_bits >> 31) & 1
    exp = (x_bits >> 23) & 0xFF
    man = x_bits & 0x7FFFFF

    is_zero = exp == 0
    exp_bfp = torch.clamp(exp - 112, 0, 31)
    exp_bfp = torch.where(is_zero, torch.zeros_like(exp_bfp), exp_bfp)
    man_full = torch.where(is_zero, torch.zeros_like(man), man | (1 << 23))

    shared_exp = exp_bfp.max(dim=-1, keepdim=True).values
    exp_diff = (shared_exp - exp_bfp).clamp(0, 31)
    man_aligned = man_full >> exp_diff

    shift = 24 - man_bits
    remainder = man_aligned & ((1 << shift) - 1)
    tie = 1 << (shift - 1)
    man_n = man_aligned >> shift
    man_n = man_n + (remainder >= tie).to(torch.int32)
    man_n = man_n.clamp(max=(1 << man_bits) - 1)
    sign = torch.where(man_n == 0, torch.zeros_like(sign), sign)

    return sign, exp_bfp, shared_exp, exp_diff, man_n


def simulate_bfp_quantization_raw_exp(x, man_bits):
    """Same simulation but using raw 8-bit exponents (for _b format)."""
    orig_shape = x.shape
    x_f32 = x.to(torch.float32).contiguous().reshape(-1, 16)
    x_bits = x_f32.view(torch.int32)

    sign = (x_bits >> 31) & 1
    exp = (x_bits >> 23) & 0xFF
    man = x_bits & 0x7FFFFF

    is_zero = exp == 0
    man_full = torch.where(is_zero, torch.zeros_like(man), man | (1 << 23))

    # Use raw 8-bit exponents directly (no rebasing for _b format)
    exp_raw = torch.where(is_zero, torch.zeros_like(exp), exp)
    shared_exp_raw = exp_raw.max(dim=-1, keepdim=True).values
    exp_diff = (shared_exp_raw - exp_raw).clamp(0, 31)
    man_aligned = man_full >> exp_diff

    shift = 24 - man_bits
    remainder = man_aligned & ((1 << shift) - 1)
    tie = 1 << (shift - 1)
    man_n = man_aligned >> shift
    man_n = man_n + (remainder >= tie).to(torch.int32)
    man_n = man_n.clamp(max=(1 << man_bits) - 1)
    sign = torch.where(man_n == 0, torch.zeros_like(sign), sign)

    return sign, exp_raw, shared_exp_raw, exp_diff, man_n


torch.manual_seed(0)
torch_input = torch.randn((64, 64), dtype=torch.bfloat16)

# Python simulation (rebased exponents - what the test uses)
sign_py, exp_py, shared_py, diff_py, man_py = simulate_bfp_quantization(torch_input, 3)

# Python simulation with raw exponents (matching _b format HW behavior)
sign_raw, exp_raw, shared_raw, diff_raw, man_raw = simulate_bfp_quantization_raw_exp(torch_input, 3)

# Device typecast
device = ttnn.open_device(device_id=0)
input_tensor = ttnn.from_torch(torch_input, memory_config=ttnn.L1_MEMORY_CONFIG)
input_tensor = ttnn.to_device(input_tensor, device)
input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
input_tensor = ttnn.typecast(input_tensor, dtype=ttnn.bfloat4_b)
actual = ttnn.to_torch(input_tensor)
ttnn.close_device(device)

# Also get host-side BFP4 pack result (via from_torch with bfp4_b)
host_bfp4 = ttnn.from_torch(
    torch_input,
    dtype=ttnn.bfloat4_b,
    device=None,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
host_result = ttnn.to_torch(host_bfp4)

# Compare first 2 groups (32 elements = rows 0-1 of tile face 0)
x_f32 = torch_input.to(torch.float32).reshape(-1, 16)
actual_f32 = actual.to(torch.float32).reshape(-1, 16)
host_f32 = host_result.to(torch.float32).reshape(-1, 16)

print("=" * 120)
print("Per-element analysis for first 2 groups of 16 (rows 0-1 of first face)")
print("=" * 120)

for group in range(2):
    print(f"\n--- Group {group} (elements {group*16}-{group*16+15}) ---")
    print(f"  Shared exp (rebased): {shared_py[group].item()}, Shared exp (raw): {shared_raw[group].item()}")
    print(
        f"  {'Idx':>3} | {'Input':>10} | {'Raw Exp':>7} | {'Reb Exp':>7} | {'ExpDiff':>7} | {'ManN(reb)':>9} | {'ManN(raw)':>9} | {'Py Sim':>10} | {'Host':>10} | {'Device':>10} | {'Match?':>6}"
    )
    print(
        f"  {'-'*3}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}"
    )

    x_bits = x_f32[group].view(torch.int32)
    for i in range(16):
        inp = x_f32[group, i].item()
        raw_e = (x_bits[i].item() >> 23) & 0xFF
        reb_e = exp_py[group, i].item()
        ed = diff_py[group, i].item()
        mn_reb = man_py[group, i].item()
        mn_raw = man_raw[group, i].item()

        # Reconstruct Python sim output (rebased path)
        import struct

        py_out = 0.0
        if mn_reb > 0:
            lz = 0
            tmp = mn_reb
            for b in range(1, 3):
                if 1 <= tmp < (1 << (3 - b)):
                    lz = b
            man_out = ((mn_reb << lz) << 1) & 7
            exp_out = shared_py[group].item() - lz + 112
            s = sign_py[group, i].item()
            bits = (s << 31) | (exp_out << 23) | (man_out << 20)
            py_out = struct.unpack("f", struct.pack("I", bits))[0]

        dev_val = actual_f32[group, i].item()
        host_val = host_f32[group, i].item()
        match_dev = "OK" if abs(py_out - dev_val) < 1e-10 else "DIFF"
        match_host = "h" + ("OK" if abs(py_out - host_val) < 1e-10 else "DIFF")

        print(
            f"  {i:3d} | {inp:10.6f} | {raw_e:7d} | {reb_e:7d} | {ed:7d} | {mn_reb:9d} | {mn_raw:9d} | {py_out:10.6f} | {host_val:10.6f} | {dev_val:10.6f} | {match_dev:>6}"
        )

# Overall statistics
py_sim_full = torch.zeros_like(torch_input, dtype=torch.float32)
# Quick full simulation for comparison
xf = torch_input.to(torch.float32).reshape(-1, 16)
xb = xf.view(torch.int32)
s = (xb >> 31) & 1
e = (xb >> 23) & 0xFF
m = xb & 0x7FFFFF
iz = e == 0
eb = torch.clamp(e - 112, 0, 31)
eb = torch.where(iz, torch.zeros_like(eb), eb)
mf = torch.where(iz, torch.zeros_like(m), m | (1 << 23))
se = eb.max(dim=-1, keepdim=True).values
ed2 = (se - eb).clamp(0, 31)
ma = mf >> ed2
sh = 21
rem = ma & ((1 << sh) - 1)
ti = 1 << (sh - 1)
mn = ma >> sh
mn = mn + (rem >= ti).to(torch.int32)
mn = mn.clamp(max=7)

total = mn.numel()
match_raw = (man_raw == mn).sum().item()
differ_raw = total - match_raw

print(f"\n{'='*80}")
print(f"Rebased vs raw exponent comparison:")
print(f"  Total elements: {total}")
print(f"  ManN(rebased) == ManN(raw): {match_raw} ({100*match_raw/total:.1f}%)")
print(f"  ManN(rebased) != ManN(raw): {differ_raw} ({100*differ_raw/total:.1f}%)")

# Compare device vs host
dev_flat = actual.to(torch.float32).reshape(-1)
host_flat = host_result.to(torch.float32).reshape(-1)
dev_host_match = torch.equal(actual, host_result)
print(f"\n  Device == Host pack: {dev_host_match}")
if not dev_host_match:
    diffs = (dev_flat != host_flat).sum().item()
    print(f"  Differing elements (device vs host): {diffs}/{dev_flat.numel()}")
