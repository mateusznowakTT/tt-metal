// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "lltt.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size = 64;
    const uint offset0 = dst_index_in0 * dst_tile_size;
    const uint offset1 = dst_index_in1 * dst_tile_size;
    const uint offset2 = dst_index_out * dst_tile_size;

#ifdef DISABLE_SFPLOADMACRO
    // Non-LOADMACRO: sequential load-swap-store.
    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG1;

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        TT_SFPLOAD(a, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0);
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);
        // After SFPSWAP: VD=a gets max (mod1=9) or min (mod1=8); VC=b gets the other.
        TTI_SFPSWAP(0, b, a, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TT_SFPSTORE(a, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset2);
        sfpi::dst_reg++;
    }
#else
    // This uses SFPLOADMACRO to achieve a throughput of 3 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple              | MAD | Round     | Store   |
    // - | ---- | ------------------- | --- | --------- | ------- |
    // 0 | [a]  |                     |     |           |         |
    // 1 |  b   |                     |     |           |         |
    // 2 | [c]  | swap_minmax([a], b) |     |           |         |
    // 0 | ...  |                     |     |           |         |
    // 1 | ...  |                     |     | L16 = [a] |         |
    // 2 | ...  |                     |     |           | [c] L16 |
    const uint lm_offset0 = (dst_index_in0 * 32) << 1;
    const uint lm_offset1 = (dst_index_in1 * 32) << 1;
    const uint lm_offset2 = (dst_index_out * 32) << 1;

    constexpr int b = p_sfpu::LREG2;
    constexpr int c = p_sfpu::LREG3;

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        int a = i & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, lm_offset0 | (a >> 2));
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_7, lm_offset1);
        TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, lm_offset2 | (c >> 2));
    }

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size = 64;
    const uint offset0 = dst_index_in0 * dst_tile_size;
    const uint offset1 = dst_index_in1 * dst_tile_size;
    const uint offset2 = dst_index_out * dst_tile_size;

#ifdef DISABLE_SFPLOADMACRO
    // Non-LOADMACRO: sequential overflow-safe signed/unsigned comparison + masked selection.
    //
    // Computes is_a_less_than_b flag using the same CC-predicated technique as
    // calculate_binary_comp_int32, then selects a or b based on the flag.

    // Register allocation:
    constexpr int a = p_sfpu::LREG0;       // operand a (from in0)
    constexpr int flag = p_sfpu::LREG1;    // operand b, then is_a_less_than_b flag, then mask
    constexpr int sign_a = p_sfpu::LREG2;  // bit31(a), later reused as ~mask
    constexpr int diff = p_sfpu::LREG3;    // diff_sign = bit31(a) XOR bit31(b)
    constexpr int b_copy = p_sfpu::LREG4;  // preserved copy of b

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        TT_SFPLOAD(a, InstrModLoadStore::INT32, ADDR_MOD_7, offset0);
        TT_SFPLOAD(flag, InstrModLoadStore::INT32, ADDR_MOD_7, offset1);  // flag = b

        TTI_SFPMOV(0, flag, b_copy, 0);  // b_copy = b (save before overwriting)

        // Extract bit31 of a and b for sign/overflow check
        TTI_SFPMOV(0, a, sign_a, 0);
        TTI_SFPMOV(0, flag, diff, 0);
        TTI_SFPSHFT((-31) & 0xfff, sign_a, sign_a, 1);  // sign_a = bit31(a)
        TTI_SFPSHFT((-31) & 0xfff, diff, diff, 1);      // diff   = bit31(b)
        TTI_SFPXOR(0, sign_a, diff, 0);                 // diff = bit31(a) XOR bit31(b)

        // CC region:
        // Same-sign path (diff == 0): flag = a - b, then extract sign bit
        TTI_SFPSETCC(0, diff, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPIADD(0, a, flag, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSHFT((-31) & 0xfff, flag, flag, 1);  // flag = sign(a-b): 1 if a<b
        // Diff-sign path: flag = sign bit that indicates a<b
        TTI_SFPCOMPC(0, 0, 0, 0);                              // CC = diff-sign
        TTI_SFPLOADI(flag, sfpi::SFPLOADI_MOD0_USHORT, 0x00);  // flag=0 (diff-sign default)
        if constexpr (!IS_UNSIGNED) {
            TTI_SFPSETCC(0, sign_a, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);  // CC = diff-sign && a<0
        } else {
            TTI_SFPSETCC(0, sign_a, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // CC = diff-sign && a<2^31
        }
        TTI_SFPLOADI(flag, sfpi::SFPLOADI_MOD0_USHORT, 0x01);  // flag=1 where a<b (diff-sign)
        TTI_SFPENCC(0, 0, 0, 0);                               // end CC region; flag = is_a_less_than_b (0 or 1)

        // Generate mask = -flag: 0x00000000 if a>=b, 0xFFFFFFFF if a<b
        TTI_SFPLOADI(diff, sfpi::SFPLOADI_MOD0_USHORT, 0x00);  // diff = 0
        TTI_SFPIADD(0, diff, flag, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // flag = 0 - flag = -flag = mask

        TTI_SFPNOT(0, flag, sign_a, 0);  // sign_a = ~mask

        // Select result using mask:
        // IS_MAX_OP: result = (a & ~mask) | (b & mask)  [b when a<b, else a]
        // IS_MIN_OP: result = (a & mask)  | (b & ~mask) [a when a<b, else b]
        if constexpr (IS_MAX_OP) {
            TTI_SFPAND(0, sign_a, a, 0);     // a = a & ~mask
            TTI_SFPAND(0, flag, b_copy, 0);  // b_copy = b & mask
        } else {
            TTI_SFPAND(0, flag, a, 0);         // a = a & mask
            TTI_SFPAND(0, sign_a, b_copy, 0);  // b_copy = b & ~mask
        }
        TTI_SFPOR(0, b_copy, a, 0);  // a = result

        TT_SFPSTORE(a, InstrModLoadStore::INT32, ADDR_MOD_7, offset2);
        sfpi::dst_reg++;
    }
#else
    // This uses SFPLOADMACRO to achieve a throughput of 5 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple                | MAD | Round     | Store   |
    // - | ---- | --------------------- | --- | --------- | ------- |
    // 0 | [a0] |                       |     |           |         |
    // 1 | [b0] |                       |     |           |         |
    // 2 |      | setcc  a1  (<0 or ≥0) |     |           |         |
    // 3 |      | encc                  |     |           |         |
    // 4 | [c]  | swap_minmax([a0], b0) |     |           |         |
    // 0 | ...  |                       |     |           |         |
    // 1 | ...  | setcc [b0] (<0 or ≥0) |     | L16 = [a] |         |
    // 2 | ...  | setcc  a0  (<0 or ≥0) |     |           |         |
    // 3 | ...  | encc                  |     | L16 = [b] |         |
    // 4 | ...  |                       |     |           | [c] L16 |

    constexpr int a0 = p_sfpu::LREG0;
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int a1 = p_sfpu::LREG2;
    constexpr int b1 = p_sfpu::LREG3;
    constexpr int c = p_sfpu::LREG7;

    load_replay_buf(0, 10, [offset0, offset1, offset2] {
        // first iteration, with a0, b0, c
        TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset0 | (a0 >> 2));
        TT_SFPLOADMACRO((2 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset1 | (b0 >> 2));
        TTI_SFPSETCC(0, a1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPENCC(0, 0, 0, 0);
        TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset2 | (c >> 2));

        // second iteration, with a1, b1, c
        TT_SFPLOADMACRO((1 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset0 | (a1 >> 2));
        TT_SFPLOADMACRO((2 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset1 | (b1 >> 2));
        TTI_SFPSETCC(0, a0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPENCC(0, 0, 0, 0);
        TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset2 | (c >> 2));
    });

#pragma GCC unroll 4
    for (int i = 0; i < ITERATIONS / 2; ++i) {
        lltt::replay(0, 10);
    }

    if constexpr (ITERATIONS & 1) {
        lltt::replay(0, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(5 + 2, 2);
    } else {
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(2, 2);
    }

    TTI_SFPNOP;
#endif  // DISABLE_SFPLOADMACRO
}

template <bool IS_MAX_OP = true>
inline void binary_max_min_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b = p_sfpu::LREG2;

    // InstructionTemplate[0]
    TTI_SFPSWAP(0, b, 12, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=9 means set VD=max and VC=min

    // InstructionTemplate[1]
    TTI_SFPSHFT2(0, 0, 13, 6);  // SFPSHFT2_MOD1_SHFT_IMM

    // Macro 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (3 << 3) | 5;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {1,1},
    //   UnitDelayKind: {1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x330, 8, 1);
#endif
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void binary_max_min_int32_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int b1 = p_sfpu::LREG3;

    // InstructionTemplate[0]
    TTI_SFPSWAP(
        0, b0, 12, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=9 means set VD=max and VC=min

    // InstructionTemplate[1]
    TTI_SFPSWAP(
        0, b1, 13, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=9 means set VD=max and VC=min

    // InstructionTemplate[2]
    TTI_SFPSETCC(0, 0, 14, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);

    // InstructionTemplate[3]
    TTI_SFPSHFT2(0, 0, 15, 6);  // SFPSHFT2_MOD1_SHFT_IMM

    // Macro 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 5;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Macro 2:
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (4 << 3) | 6;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (6 << 3) | 7;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Macro 3:
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (4 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {1,1,1,1},
    //   UnitDelayKind: {1,1,1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0xff0, 8, 1);
#endif
}

}  // namespace sfpu
}  // namespace ckernel
