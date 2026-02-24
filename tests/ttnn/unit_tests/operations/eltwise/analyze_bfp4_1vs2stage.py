"""Targeted test: construct bf16 values where 1-stage and 2-stage models diverge.

For ed=0, the models differ when the 8->7 RHU round-up causes a carry past bit 4
of the 7-bit result:

  man_full = (1 << 7) | man_7bit
  2-stage: (man_full >> 1) + (man_full & 1) = man_bfp8, then man_bfp8 >> 4
  1-stage: man_full >> 5

They differ when man_full is odd AND the 7-bit intermediate crosses a multiple of 16.
Example: man_full=159 (man_7bit=31): 2stg gives 5, 1stg gives 4.

We also need ed=0, meaning all 16 values in a group must have the SAME exponent.
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


def make_bf16_with_man7(man_7bit, exp_biased=127):
    """Construct a bf16 value with specific mantissa and exponent."""
    bits = (exp_biased << 23) | (man_7bit << 16)
    return struct.unpack("f", struct.pack("I", bits))[0]


# Find all man_7bit values where 1-stage and 2-stage differ for ed=0
diverging = []
for m7 in range(128):
    man_full = (1 << 7) | m7
    # 2-stage
    man_bfp8 = (man_full >> 1) + (man_full & 1)
    man_bfp8 = min(man_bfp8, 127)
    two_stg = min(man_bfp8 >> 4, 7)
    # 1-stage
    one_stg = min(man_full >> 5, 7)
    if two_stg != one_stg:
        diverging.append((m7, two_stg, one_stg))

print(f"Man_7bit values where models diverge (ed=0): {len(diverging)}")
for m7, two, one in diverging[:10]:
    print(f"  man_7bit={m7:3d} (0b{m7:07b})  2stg={two}  1stg={one}")

if not diverging:
    print("Models are equivalent for ed=0! Checking ed>0...")
    for ed in range(1, 8):
        for m7 in range(128):
            man_full = (1 << 7) | m7
            # carry align
            man_aligned = (man_full + 1) >> ed
            # 2-stage
            man_bfp8 = (man_aligned >> 1) + (man_aligned & 1)
            man_bfp8 = min(man_bfp8, 127)
            two_stg = min(man_bfp8 >> 4, 7)
            # 1-stage
            one_stg = min(man_aligned >> 5, 7)
            if two_stg != one_stg:
                diverging.append((m7, ed, two_stg, one_stg))
        if diverging:
            print(f"  Found {len(diverging)} differences at ed={ed}")
            break

if not diverging:
    print("\n*** Models are ALGEBRAICALLY EQUIVALENT for all ed and man_7bit values ***")
    print("*** The 2-stage claim cannot be verified — both produce identical results ***")
else:
    # Create a tile of bf16 values that all have the same exponent and use a diverging mantissa
    m7_test = diverging[0][0]
    exp_test = 127

    # All 16 elements in each group must have same exponent (ed=0 for all)
    test_values = [make_bf16_with_man7(m7_test, exp_test)] * 16
    # Fill a 32x32 tile
    tile_data = test_values * (32 * 32 // 16)
    torch_input = torch.tensor(tile_data, dtype=torch.float32).to(torch.bfloat16).reshape(32, 32)

    # Verify our input has the expected mantissa
    x_bits = torch_input.to(torch.float32).view(torch.int32)
    actual_m7 = ((x_bits >> 16) & 0x7F).flatten()[0].item()
    print(f"\nTest value: man_7bit={actual_m7}, exp={exp_test}")
    print(f"  Expected 2stg result: {diverging[0][1]}")
    print(f"  Expected 1stg result: {diverging[0][2]}")

    # Run on device via FP32 path (no SFPU kernel)
    device = ttnn.open_device(device_id=0)
    torch_fp32 = torch_input.to(torch.float32)
    input_tensor = ttnn.from_torch(torch_fp32, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.typecast(input_tensor, dtype=ttnn.bfloat4_b)
    actual = ttnn.to_torch(input_tensor)
    ttnn.close_device(device)

    # Check result
    actual_f32 = actual.to(torch.float32).reshape(-1, 16)
    se_raw = exp_test  # all same exponent, so shared_exp = element_exp
    hw_man = infer_bfp4_mantissa(actual_f32[0, 0].item(), se_raw)
    print(f"  Device result mantissa: {hw_man}")

    if hw_man == diverging[0][1]:
        print("\n*** CONFIRMED: HW packer uses 2-STAGE conversion (8->7 RHU then 7->3 truncation) ***")
    elif hw_man == diverging[0][2]:
        print("\n*** CONFIRMED: HW packer uses 1-STAGE conversion (direct 8->3 truncation) ***")
    else:
        print(f"\n*** UNEXPECTED: device mantissa {hw_man} matches neither model ***")
