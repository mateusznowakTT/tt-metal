# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

mem_configs = [
    ttnn.DRAM_MEMORY_CONFIG,
    ttnn.L1_MEMORY_CONFIG,
]

TILE_HEIGHT = 32
TILE_WIDTH = 32

cpu_layout = ttnn.Layout.ROW_MAJOR
npu_layout = ttnn.Layout.TILE


@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    (
        (torch.bfloat16, ttnn.bfloat16, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.bfloat16),
        (torch.float32, ttnn.float32, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b),
        (torch.int, ttnn.uint16, ttnn.uint32),
        (torch.int, ttnn.uint16, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.uint16),
        (torch.int, ttnn.uint32, ttnn.uint16),
    ),
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],  # Single core
        [[1, 1, 320, 320]],  # multi core
        [[1, 3, 320, 320]],  # multi core
    ],
)
@pytest.mark.parametrize(
    "input_mem_config",
    mem_configs,
)
@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
class TestTypecast:
    def test_run_eltwise_typecast_op(
        self,
        tt_output_dtype,
        pt_input_dtype,
        tt_input_dtype,
        input_shapes,
        input_mem_config,
        dst_mem_config,
        device,
        function_level_defaults,
    ):
        if tt_input_dtype == tt_output_dtype:
            pytest.skip("Same I/O data types. Skip.")
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=0, high=100), pt_input_dtype)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["tt_input_dtype"] = [tt_input_dtype]
        test_args["tt_output_dtype"] = [tt_output_dtype]
        test_args["input_mem_config"] = [input_mem_config]
        test_args.update({"output_mem_config": dst_mem_config})
        comparison_func = comparison_funcs.comp_pcc
        if tt_input_dtype == ttnn.bfloat4_b or tt_output_dtype == ttnn.bfloat4_b:
            comparison_func = partial(comparison_funcs.comp_pcc, pcc=0.98)

        run_single_pytorch_test(
            "eltwise-typecast",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )


@pytest.mark.skip("Issue #17237: Does not work with new mantissa rounding")
def test_typecast_bf16_to_bfp8_b(device):
    torch.manual_seed(0)
    shape = [32, 32]

    # bf16 --> bfp8_b by cpu.
    torch_bf16 = torch.randn(shape, dtype=torch.bfloat16)
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def print_mismatches(cpu, npu, num_max_print):
    different_indices = (cpu != npu).nonzero(as_tuple=True)
    count = 0
    for idx in zip(*different_indices):
        count = count + 1
        print(f"idx={idx} cpu={cpu[idx]} npu={npu[idx]}")
        if count > num_max_print:
            break


@pytest.mark.skip("Issue #17237: Does not work with new mantissa rounding")
@pytest.mark.parametrize("seed", [0, 2, 4, 6, 8])
@pytest.mark.parametrize("scale", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("bias", [0, 1, 2, 4, 8, 16, 32, 64, 128])
def test_typecast_bf16_to_bfp8_b_various_input(seed, scale, bias, device):
    torch.manual_seed(seed)
    shape = [1024, 1024]

    bias = bias
    low = bias - scale
    high = bias + scale
    torch_bf16 = random_tensor = torch.empty(shape).uniform_(low, high).to(torch.bfloat16)

    random_signs = torch.randint(0, 2, shape) * 2 - 1
    torch_bf16 = torch_bf16 * random_signs

    # bf16 --> bfp8_b by cpu.
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    if not passed:
        print_mismatches(cpu_version, npu_version, 16)
    assert passed


@pytest.mark.skip("Issue #17237: Does not work with new mantissa rounding")
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("scale", [4])
@pytest.mark.parametrize("bias", [2])
# NaN becomes -Inf when converted to bfloat8_b format, skip testing
@pytest.mark.parametrize("insert_inf, insert_nan", [[True, False]])  # , [False, True], [True, True]])
def test_typecast_bf16_to_bfp8_b_with_inf_nan(seed, scale, bias, insert_inf, insert_nan, device):
    torch.manual_seed(seed)
    shape = [1024, 1024]

    bias = bias
    low = bias - scale
    high = bias + scale

    torch_bf16 = random_tensor = torch.empty(shape).uniform_(low, high).to(torch.bfloat16)
    if insert_inf:
        num_inf = torch_bf16.numel() // 8  # 16 elements are pcked into
        inf_indices = torch.randint(0, torch_bf16.numel(), (num_inf,))
        torch_bf16.view(-1)[inf_indices] = float("inf")
    if insert_nan:
        num_nan = torch_bf16.numel() // 8
        nan_indices = torch.randint(0, torch_bf16.numel(), (num_nan,))
        torch_bf16.view(-1)[nan_indices] = float("nan")
    random_signs = torch.randint(0, 2, shape) * 2 - 1
    torch_bf16 = torch_bf16 * random_signs

    # bf16 --> bfp8_b by cpu.
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    if not passed:
        print_mismatches(cpu_version, npu_version, 16)
    assert passed


def test_typecast_bfp8_b_to_bf16(device):
    torch.manual_seed(0)
    shape = [1024, 1024]

    # bfp8_b --> bf16 by cpu.
    torch_bf16 = torch.randn(shape, dtype=torch.bfloat16)
    bfp8_b = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b.to(cpu_layout).to_torch()

    # bfp8_b --> bf16 by npu.
    bf16_by_npu = ttnn.typecast(bfp8_b.to(device), ttnn.bfloat16)
    npu_version = bf16_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def test_typecast_fp32_to_bfp8_b(device):
    torch.manual_seed(0)
    shape = [32, 32]

    # fp32 --> bfp8_b by cpu.
    torch_fp32 = torch.randn(shape, dtype=torch.float32)
    bfp8_b_by_cpu = ttnn.Tensor(torch_fp32, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # fp32 --> bfp8_b by npu
    tt_fp32 = ttnn.Tensor(torch_fp32, ttnn.float32).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_fp32, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def test_typecast_bfp8_b_to_fp32(device):
    torch.manual_seed(0)
    shape = [1024, 1024]

    # bfp8_b --> fp32 by cpu.
    torch_fp32 = torch.randn(shape, dtype=torch.float32)
    bfp8_b = ttnn.Tensor(torch_fp32, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b.to(cpu_layout).to_torch()

    # bfp8_b --> fp32 by npu.
    fp32_by_npu = ttnn.typecast(bfp8_b.to(device), ttnn.float32)
    npu_version = fp32_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def _simulate_bfp_quantization(x, man_bits):
    """Simulate bfp round-trip (float -> bfp -> float) on CPU.

    Args:
        x: Input tensor.
        man_bits: Number of mantissa bits (3 for bfp4, 7 for bfp8).
    """
    orig_shape = x.shape
    x_f32 = x.to(torch.float32).contiguous().reshape(-1, 16)
    x_bits = x_f32.view(torch.int32)

    sign = (x_bits >> 31) & 1
    exp = (x_bits >> 23) & 0xFF
    man = x_bits & 0x7FFFFF

    is_zero = exp == 0
    exp_bfp = torch.clamp(exp - 112, 0, 31)  # rebias: -127 + 15 = -112
    exp_bfp = torch.where(is_zero, torch.zeros_like(exp_bfp), exp_bfp)
    man_full = torch.where(is_zero, torch.zeros_like(man), man | (1 << 23))

    shared_exp = exp_bfp.max(dim=-1, keepdim=True).values
    exp_diff = (shared_exp - exp_bfp).clamp(0, 31)
    man_aligned = man_full >> exp_diff

    # Round to man_bits (round half-up)
    shift = 24 - man_bits
    remainder = man_aligned & ((1 << shift) - 1)
    tie = 1 << (shift - 1)
    man_n = man_aligned >> shift
    man_n = man_n + (remainder >= tie).to(torch.int32)
    man_n = man_n.clamp(max=(1 << man_bits) - 1)
    sign = torch.where(man_n == 0, torch.zeros_like(sign), sign)

    # Unpack: normalize mantissa (find leading zeros in man_bits-wide field)
    bfp_zero = man_n == 0
    lz = torch.zeros_like(man_n)
    for i in range(1, man_bits):
        lz = torch.where(
            (man_n >= 1) & (man_n < (1 << (man_bits - i))),
            torch.tensor(i, dtype=torch.int32),
            lz,
        )
    lz = torch.where(bfp_zero, torch.zeros_like(lz), lz)
    man_out = ((man_n << lz) << 1) & ((1 << man_bits) - 1)
    exp_out = shared_exp - lz + 112  # rebias to FP32: +127 - 15 = +112

    result = (sign << 31) | (exp_out << 23) | (man_out << (23 - man_bits))
    result = torch.where(bfp_zero, torch.zeros_like(result), result)
    return result.view(torch.float32).reshape(orig_shape).to(torch.bfloat16)


@pytest.mark.parametrize("tile_h", [32])  # [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("tile_w", [32])  # [16, 32])
@pytest.mark.parametrize("dtype", [ttnn.bfloat4_b])  # [ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize("transpose_tile", [False])  # [True, False])
@pytest.mark.parametrize("on_device", [True])  # [True, False])
def test_tiny_tiles_bfloat_on_device_conversion(device, tile_h, tile_w, dtype, transpose_tile, on_device, capsys):
    if tile_h < 16 and transpose_tile:
        pytest.skip("transpose tile does not support tile height less than 16")
    # minimum tile_h = 4 for fbloat, as exponents are packed into uint32 (4 exponents minmum)
    torch.manual_seed(0)
    torch_input_tensor = torch.randn((64, 64), dtype=torch.bfloat16)

    def verify_tensor():
        output_tensor = ttnn.to_torch(input_tensor)

        if dtype == ttnn.bfloat16 or dtype == ttnn.bfloat8_b:
            expected_pcc = 0.9999
        elif dtype == ttnn.bfloat4_b:
            expected_pcc = 0.989

        assert_with_pcc(torch_input_tensor, output_tensor, expected_pcc)

    if on_device:
        input_tensor = ttnn.from_torch(
            torch_input_tensor,
            tile=ttnn.Tile((tile_h, tile_w), transpose_tile=transpose_tile),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        with capsys.disabled():
            print("After from_torch")
        verify_tensor()
        input_tensor = ttnn.to_device(input_tensor, device)
        with capsys.disabled():
            print("After to_device")
        verify_tensor()
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
        with capsys.disabled():
            print("After to_layout")
        verify_tensor()
        input_tensor = ttnn.typecast(input_tensor, dtype=dtype)
        with capsys.disabled():
            print("After typecast")
        verify_tensor()

    else:
        input_tensor = ttnn.from_torch(
            torch_input_tensor,
            tile=ttnn.Tile((tile_h, tile_w), transpose_tile=transpose_tile),
            dtype=dtype,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        verify_tensor()
