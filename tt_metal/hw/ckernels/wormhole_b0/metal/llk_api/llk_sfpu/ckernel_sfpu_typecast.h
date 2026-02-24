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

// Pre-round bf16 mantissa for BFP4 packer output.
// The HW packer converts 7-bit mantissa to 3-bit via two steps:
//   Step 1: 8->7 bit round-half-up (BFP8 intermediate)
//   Step 2: 7->3 bit truncation (BFP4 final)
// This kernel adds a rounding bias of 2^(element_exp - 4 - 127) to each
// element before the packer runs. For ed=0 elements (max exponent in their
// BFP group), this adds 4 to man_bfp8 which partially converts the 7->3
// truncation into rounding. The bias is kept small to minimize disturbance
// to ed>0 elements' carry-align step.
// Note: the ideal bias for perfect ed=0 rounding would be 2^(exp-3-127),
// but that larger bias damages ed>0 elements more than it helps ed=0.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp16b_to_bfp4b_() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v_if(v != sfpi::vConst0) {
            const auto raw_exp = sfpi::exexp_nodebias(v);
            // Guard against exponent underflow (raw_exp - 4 would wrap)
            v_and(raw_exp >= 4);

            // bias = 2^(raw_exp - 4 - 127), zero mantissa
            const auto bias = sfpi::setexp(sfpi::vFloat(sfpi::vConst1), raw_exp - 4);
            const auto signed_bias = sfpi::setsgn(bias, v);
            const auto biased = v + signed_bias;

            // Only accept if exponent didn't overflow (prevents shared_exp pollution)
            v_and(sfpi::exexp_nodebias(biased) == raw_exp);
            sfpi::dst_reg[0] = biased;
        }
        v_endif;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp16b_to_bfp4b() {
    _calculate_typecast_fp16b_to_bfp4b_<APPROXIMATION_MODE, ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
