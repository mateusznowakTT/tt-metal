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
    // Non-LOADMACRO: sequential load-compare-store using sfpi vFloat API.
    // Uses sfpi comparison instead of SFPSWAP to avoid pipeline hazards with two fresh loads.
    constexpr uint sfpi_tile_size = 32;

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * sfpi_tile_size];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * sfpi_tile_size];
        sfpi::vFloat result = a;
        if constexpr (IS_MAX_OP) {
            v_if(a < b) { result = b; }
            v_endif;
        } else {
            v_if(b < a) { result = b; }
            v_endif;
        }
        sfpi::dst_reg[dst_index_out * sfpi_tile_size] = result;
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
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_3, lm_offset0 | (a >> 2));
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_3, lm_offset1);
        TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, lm_offset2 | (c >> 2));
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
    constexpr int a = p_sfpu::LREG0;
    constexpr int flag = p_sfpu::LREG1;
    constexpr int sign_a = p_sfpu::LREG2;
    constexpr int diff = p_sfpu::LREG3;
    constexpr int b_copy = p_sfpu::LREG4;

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        TT_SFPLOAD(a, InstrModLoadStore::INT32, ADDR_MOD_3, offset0);
        TT_SFPLOAD(flag, InstrModLoadStore::INT32, ADDR_MOD_3, offset1);

        TTI_SFPMOV(0, flag, b_copy, 0);

        TTI_SFPMOV(0, a, sign_a, 0);
        TTI_SFPMOV(0, flag, diff, 0);
        TTI_SFPSHFT((-31) & 0xfff, sign_a, sign_a, 1);
        TTI_SFPSHFT((-31) & 0xfff, diff, diff, 1);
        TTI_SFPXOR(0, sign_a, diff, 0);

        TTI_SFPSETCC(0, diff, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPIADD(0, a, flag, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSHFT((-31) & 0xfff, flag, flag, 1);
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPLOADI(flag, sfpi::SFPLOADI_MOD0_USHORT, 0x00);
        if constexpr (!IS_UNSIGNED) {
            TTI_SFPSETCC(0, sign_a, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        } else {
            TTI_SFPSETCC(0, sign_a, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        }
        TTI_SFPLOADI(flag, sfpi::SFPLOADI_MOD0_USHORT, 0x01);
        TTI_SFPENCC(0, 0, 0, 0);

        // Normalize flag: same-sign a<b path sets flag=0xFFFFFFFF (via SFPSHFT(-31) of negative
        // a-b), diff-sign a<b path sets flag=0x00000001. Normalize to 0 or 1 so that -flag gives
        // the correct mask (0 or 0xFFFFFFFF). Without this, -(0xFFFFFFFF) = 1, not 0xFFFFFFFF.
        TTI_SFPSETCC(0, flag, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPLOADI(flag, sfpi::SFPLOADI_MOD0_USHORT, 0x01);  // flag = 1 wherever flag != 0
        TTI_SFPENCC(0, 0, 0, 0);

        // Generate mask: -flag gives 0x00000000 (a>=b) or 0xFFFFFFFF (a<b).
        TTI_SFPLOADI(diff, sfpi::SFPLOADI_MOD0_USHORT, 0x00);
        TTI_SFPIADD(0, diff, flag, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPNOT(0, flag, sign_a, 0);

        if constexpr (IS_MAX_OP) {
            TTI_SFPAND(0, sign_a, a, 0);
            TTI_SFPAND(0, flag, b_copy, 0);
        } else {
            TTI_SFPAND(0, flag, a, 0);
            TTI_SFPAND(0, sign_a, b_copy, 0);
        }
        TTI_SFPOR(0, b_copy, a, 0);

        TT_SFPSTORE(a, InstrModLoadStore::INT32, ADDR_MOD_3, offset2);
        sfpi::dst_reg++;
    }
#else
    const uint lm_offset0 = (dst_index_in0 * 32) << 1;
    const uint lm_offset1 = (dst_index_in1 * 32) << 1;
    const uint lm_offset2 = (dst_index_out * 32) << 1;

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

    lltt::record<lltt::NoExec>(0, 10);

    // first iteration, with a0, b0, c
    TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, lm_offset0 | (a0 >> 2));
    TT_SFPLOADMACRO((2 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, lm_offset1 | (b0 >> 2));
    TTI_SFPSETCC(0, a1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_2, lm_offset2 | (c >> 2));

    // second iteration, with a1, b1, c
    TT_SFPLOADMACRO((1 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, lm_offset0 | (a1 >> 2));
    TT_SFPLOADMACRO((2 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, lm_offset1 | (b1 >> 2));
    TTI_SFPSETCC(0, a0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_2, lm_offset2 | (c >> 2));

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
