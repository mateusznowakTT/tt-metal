// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_typecast.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint16() {
    _calculate_typecast_fp32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp16b() {
    _calculate_typecast_uint16_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp16b() {
    _calculate_typecast_int32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_int32() {
    _calculate_typecast_fp32_to_int32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_fp16b() {
    _calculate_typecast_fp32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp32() {
    _calculate_typecast_uint16_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp32() {
    _calculate_typecast_int32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint32() {
    _calculate_typecast_fp32_to_uint32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp16b() {
    _calculate_typecast_uint32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp32() {
    _calculate_typecast_uint32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_uint32() {
    _calculate_typecast_uint16_to_uint32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint16() {
    _calculate_typecast_uint32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_uint16() {
    _calculate_typecast_int32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_fp16b() {
    _init_typecast_fp32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_uint32() {
    _init_typecast_uint16_to_uint32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp32() {
    _init_typecast_uint32_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp32() {
    _init_typecast_int32_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp32() {
    _init_typecast_uint16_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp16b() {
    _init_typecast_uint16_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp16b() {
    _init_typecast_int32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp16b() {
    _init_typecast_uint32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint16() {
    _init_typecast_fp32_to_uint16_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_uint16() {
    _init_typecast_uint32_to_uint16_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_uint16() {
    _init_typecast_int32_to_uint16_<APPROXIMATION_MODE>();
}

// Configure SFPLOADMACRO for BFP4 pre-rounding kernel.
// Sets up Sequence[0] to schedule SFPSTORE at delay=7 (fires after 7 stream instructions).
// Uses WaitForElapsedInstructions mode so SFPLOADMACRO itself doesn't count for delay.
template <bool APPROXIMATION_MODE>
inline void _init_typecast_fp16b_to_bfp4b_() {
    // Sequence[0]: schedule Store sub-unit only
    constexpr std::uint32_t simple_bits = 0;                          // skip
    constexpr std::uint32_t mad_bits = 0;                             // skip
    constexpr std::uint32_t round_bits = 0;                           // skip
    constexpr std::uint32_t store_bits = 0x00 | 0x00 | (7 << 3) | 3;  // SFPSTORE, delay=7
    // store_bits = 0b00_111_011 = 0x3B

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
    TTI_SFPCONFIG(0, 4 + 0, 0);  // Write Sequence[0] from LReg[0]

    // Misc: StoreMod0=DEFAULT, UsesLoadMod0ForStore=0, UnitDelayKind=WaitForElapsedInstructions(macro 0)
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::DEFAULT, 8, 1);
}

// Pre-round bf16 mantissa for BFP4 packer output using SFPLOADMACRO.
// The HW packer converts 7-bit mantissa to 3-bit via two steps:
//   Step 1: 8->7 bit round-half-up (BFP8 intermediate)
//   Step 2: 7->3 bit truncation (BFP4 final)
// This kernel adds a rounding bias of 2^(element_exp - 4 - 127) to each
// element before the packer runs, using integer add on the LREG representation.
//
// SFPLOADMACRO schedules the Store at delay=7, which fires after all 7 stream
// instructions complete (on the same cycle as SFPENCC). This overlaps Store
// with SFPENCC and allows 9 cycles/iter throughput (vs 11 without SFPLOADMACRO).
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp16b_to_bfp4b_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Load v into L0 and schedule Store(L0) at delay=7
        TT_SFPLOADMACRO(0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);

        // L2 = raw exponent of v
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_NODEBIAS);

        // L3 = exp - 4; LaneEnabled &= (exp >= 4)
        // Also handles v=0: exp=0 -> 0-4=-4 < 0 -> disabled by GTE0
        TTI_SFPIADD(-4 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);

        // LaneEnabled &= (exp < 255) — guard against inf/NaN
        TTI_SFPIADD(-251 & 0xfff, p_sfpu::LREG3, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);

        // L0 = v + 64 (integer bias = 2^(-4) of significand), in-place
        // CC_NONE preserves accumulated lane flags; disabled lanes keep original L0
        TTI_SFPIADD(64, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);

        // L4 = raw exponent of biased value
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG4, sfpi::SFPEXEXP_MOD1_NODEBIAS);

        // L4 = orig_exp - biased_exp
        TTI_SFPIADD(
            0, p_sfpu::LREG2, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);

        // LaneEnabled &= (exponent unchanged) — reject mantissa overflow
        // Stream instruction 7: delay=7 elapsed, scheduled Store fires next cycle
        TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);

        // Reset all lanes to enabled for next iteration
        // Scheduled Store fires on this cycle (Store sub-unit), reading LF before SFPENCC resets it
        TTI_SFPENCC(0, 0, 0, 0);
    }
    // Let last iteration's scheduled Store fire
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp16b_to_bfp4b() {
    _init_typecast_fp16b_to_bfp4b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp16b_to_bfp4b() {
    _calculate_typecast_fp16b_to_bfp4b_<APPROXIMATION_MODE, ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
