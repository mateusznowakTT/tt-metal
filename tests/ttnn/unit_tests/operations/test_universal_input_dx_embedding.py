# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for DX team OPs - Embedding (#31713).
"""

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device

TILE_H = 32
TILE_W = 32


def make_memory_config(strategy, shape):
    H, W = shape[-2], shape[-1]
    if strategy == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    elif strategy == "l1":
        return ttnn.L1_MEMORY_CONFIG
    elif strategy == "height_sharded":
        num_cores = max(1, H // TILE_H)
        num_cores = min(num_cores, 8)
        shard_h = H // num_cores
        return ttnn.create_sharded_memory_config(
            [shard_h, W],
            core_grid=ttnn.CoreGrid(y=num_cores, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    elif strategy == "width_sharded":
        num_cores = max(1, W // TILE_W)
        num_cores = min(num_cores, 8)
        shard_w = W // num_cores
        return ttnn.create_sharded_memory_config(
            [H, shard_w],
            core_grid=ttnn.CoreGrid(y=1, x=num_cores),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
    elif strategy == "block_sharded":
        num_cores_h = max(1, H // TILE_H)
        num_cores_h = min(num_cores_h, 4)
        num_cores_w = max(1, W // TILE_W)
        num_cores_w = min(num_cores_w, 4)
        shard_h = H // num_cores_h
        shard_w = W // num_cores_w
        return ttnn.create_sharded_memory_config(
            [shard_h, shard_w],
            core_grid=ttnn.CoreGrid(y=num_cores_h, x=num_cores_w),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


ALL_MEMORY_STRATEGIES = ["dram", "l1", "height_sharded", "width_sharded", "block_sharded"]


# =============================================================================
# Embedding - weight tensor memory config
# =============================================================================


@pytest.mark.parametrize("weight_memory_strategy", ALL_MEMORY_STRATEGIES)
@pytest.mark.parametrize("weight_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_embedding_weight_memory(device, weight_memory_strategy, weight_layout):
    """Test embedding with weights in different memory configs."""
    vocab_size = 256
    embedding_dim = 128
    seq_len = 32

    torch_indices = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int32)
    torch_weights = torch.randn(vocab_size, embedding_dim, dtype=torch.bfloat16)

    tt_indices = ttnn.from_torch(torch_indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    weight_shape = [1, 1, vocab_size, embedding_dim]
    torch_weights_4d = torch_weights.reshape(weight_shape)
    weight_mem_config = make_memory_config(weight_memory_strategy, weight_shape)
    tt_weights = ttnn.from_torch(
        torch_weights_4d, dtype=ttnn.bfloat16, layout=weight_layout, device=device, memory_config=weight_mem_config
    )

    tt_output = ttnn.embedding(tt_indices, tt_weights)

    torch_output = torch.nn.functional.embedding(torch_indices.long(), torch_weights)
    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.999)


# =============================================================================
# Embedding - index tensor memory config
# =============================================================================


@pytest.mark.parametrize("index_memory_strategy", ALL_MEMORY_STRATEGIES)
def test_embedding_index_memory(device, index_memory_strategy):
    """Test embedding with index tensor in different memory configs."""
    vocab_size = 256
    embedding_dim = 128
    seq_len = 32

    torch_indices = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int32)
    torch_weights = torch.randn(vocab_size, embedding_dim, dtype=torch.bfloat16)

    index_shape = [1, 1, 1, seq_len]
    torch_indices_4d = torch_indices.reshape(index_shape)
    index_mem_config = make_memory_config(index_memory_strategy, index_shape)
    tt_indices = ttnn.from_torch(
        torch_indices_4d, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=index_mem_config
    )

    tt_weights = ttnn.from_torch(
        torch_weights.reshape(1, 1, vocab_size, embedding_dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = ttnn.embedding(tt_indices, tt_weights)

    torch_output = torch.nn.functional.embedding(torch_indices.long(), torch_weights)
    tt_result = ttnn.to_torch(tt_output)
    if tt_result.shape != torch_output.shape:
        torch_output = torch_output.reshape(tt_result.shape)

    assert_with_pcc(torch_output, tt_result, 0.999)
