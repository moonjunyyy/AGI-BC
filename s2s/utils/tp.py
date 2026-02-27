"""
Tensor-parallel helpers wrapping m00nny_utils/parallel/sharded_modules.py.
"""
import sys
import os

# Add m00nny_utils to path if not already available
_m00nny_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "m00nny_utils")
if _m00nny_path not in sys.path:
    sys.path.insert(0, os.path.dirname(_m00nny_path))

import re
import torch
import torch.nn as nn

try:
    from m00nny_utils.parallel.sharded_modules import (
        ShardedLinear,
        ShardedEmbedding,
        shardedConv1D,
        _convert_to_sharded_module_recursive,
        convert_to_sharded_module,
    )
    _m00nny_available = True
except ImportError:
    _m00nny_available = False
    # Fallback: identity wrappers when m00nny_utils not available
    class ShardedLinear(nn.Linear):
        def __init__(self, linear, row_parallel=False):
            super().__init__(linear.in_features, linear.out_features, bias=linear.bias is not None)
            self.weight = linear.weight
            if linear.bias is not None:
                self.bias = linear.bias

    class ShardedEmbedding(nn.Embedding):
        def __init__(self, embedding):
            super().__init__(embedding.num_embeddings, embedding.embedding_dim)
            self.weight = embedding.weight

    def _convert_to_sharded_module_recursive(model, **kwargs):
        return model

    def convert_to_sharded_module(module, **kwargs):
        pass


# Regex patterns for attention/MLP layers
_ATTN_COL_PATTERNS = [
    r".*\.q_proj$",
    r".*\.k_proj$",
    r".*\.v_proj$",
    r".*\.gate_proj$",
    r".*\.up_proj$",
]

_ATTN_ROW_PATTERNS = [
    r".*\.o_proj$",
    r".*\.down_proj$",
]

_EMBED_PATTERNS = [
    r".*embed_tokens$",
]


def shard_model(model: nn.Module, tp_degree: int = 1) -> nn.Module:
    """Apply tensor parallelism to a model (requires torch.distributed to be init'd).

    Must be called from inside a worker process where dist.init_process_group()
    has already been called.  Use TensorParallelPool (s2s/utils/tp_worker.py)
    to manage the worker processes — it spawns N processes, each calls this
    function with its own rank, and NCCL handles the all-reduce/all-gather.

    Sharding map:
      col_parallel: q_proj, k_proj, v_proj, gate_proj, up_proj
      row_parallel: o_proj, down_proj
      embed:        embed_tokens

    Args:
        model:     The model to shard (should be on CPU, weights already loaded).
        tp_degree: Total number of ranks; used only for the guard below.

    Returns:
        Model with ShardedLinear / ShardedEmbedding layers replacing originals.
    """
    if tp_degree <= 1:
        return model

    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialised.  "
            "Call shard_model only from inside a TensorParallelPool worker process."
        )

    if not _m00nny_available:
        raise RuntimeError(
            "m00nny_utils not found; cannot apply tensor parallelism."
        )

    return _convert_to_sharded_module_recursive(
        model,
        embed_parallel_ids=_EMBED_PATTERNS,
        col_parallel_ids=_ATTN_COL_PATTERNS,
        row_parallel_ids=_ATTN_ROW_PATTERNS,
    )
