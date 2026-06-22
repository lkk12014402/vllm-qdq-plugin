# SPDX-License-Identifier: Apache-2.0
"""QuaRot per-Linear block-Hadamard + MXFP4 ``LinearMethodBase`` (standalone).

Replays auto-round's per-Linear inverse block-Hadamard on the input activation,
then MXFP4 activation QDQ, then a dequantized GEMM. Reuses the vendored MXFP4
weight-dequant helpers and the ``vllm_qdq_plugin.hadamard_mxfp4_act_qdq`` op from
:mod:`.mxfp4`.

The block-Hadamard ``H`` (block_size x block_size, 32 for MXFP4) is **per
Linear**:
  * random_hadamard -> loaded from the checkpoint ``hadamard_matrix`` buffers.
  * hadamard (deterministic) -> regenerated as a normalized Sylvester matrix
    (byte-identical to auto-round's ``deterministic_hadamard_matrix``).

Per-partition (merged-layer) handling
--------------------------------------
In auto-round HF inference q/k/v (and gate/up) are **separate** modules, each
with its own input pre-hook applying its own inverse Hadamard ``Hᵢᵀ`` and its own
MXFP4 activation QDQ before its GEMM:

    yᵢ = qdq(x @ Hᵢᵀ) @ Sᵢᵀ          (Sᵢ = stored, already-rotated MXFP4 weight)

vLLM merges these into one ``qkv_proj`` / ``gate_up_proj`` Linear. Because the
Hadamard sits on the **input** (contraction) dimension and the activation is
quantized *after* rotation, a single shared input rotation is only correct when
every partition shares the same ``Hᵢ``. To stay faithful to auto-round in the
**true random** case (a different random Hadamard per Linear), this method stores
one matrix **per output partition** and, when they differ, processes each
partition independently (rotate -> qdq -> partial GEMM -> concat), exactly
reproducing the separate-module HF math.

When all partition matrices are identical (the common case — and what the
current auto-round export produces, since its random Hadamard uses a fixed
default RNG seed) a uniform fast path does a single rotation + single merged
GEMM.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from ._mxfp4_common import (
    clear_packed_storage,
    dense_linear_stable_dtype,
    register_mxfp4_packed_weight,
)
from .hadamard import deterministic_hadamard_matrix
from .mxfp4 import dequant_packed_mxfp4_weight
from .perlinear_config import HADAMARD_TYPE_RANDOM

logger = init_logger(__name__)

__all__ = ["HadamardMXFP4LinearMethod"]

# Maps a vLLM ``loaded_shard_id`` to a partition index for the per-partition
# Hadamard buffer. QKVParallelLinear passes "q"/"k"/"v"; MergedColumnParallelLinear
# passes an int already equal to the partition index; a plain Linear passes None.
_QKV_SHARD_TO_IDX = {"q": 0, "k": 1, "v": 2}


class HadamardMXFP4LinearMethod(LinearMethodBase):
    """Per-Linear block-Hadamard + MXFP4 activation QDQ + dequantized GEMM."""

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.quant_config.group_size
        layer.params_dtype = params_dtype

        # Partition layout (q/k/v or gate/up or single). Drives both the
        # per-partition Hadamard buffer slots and the per-partition forward.
        layer._output_partition_sizes = list(output_partition_sizes)
        num_partitions = len(output_partition_sizes)

        # Packed MXFP4 weight + e8m0 scale (shared layout with the SpinQuant method).
        register_mxfp4_packed_weight(
            layer,
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            group_size=group_size,
            extra_weight_attrs=extra_weight_attrs,
        )

        # Per-partition block Hadamard buffer: [num_partitions, bs, bs].
        # auto-round serializes one ``hadamard_matrix`` per ORIGINAL Linear
        # (q_proj/k_proj/v_proj/...). vLLM's stacked-param mapping routes each
        # of those to this merged param with a ``loaded_shard_id``; the custom
        # loader below stores each into its own slot so true per-Linear random
        # matrices are preserved (no last-shard-wins overwrite).
        bs = self.quant_config.block_size
        hadamard_matrix = Parameter(
            torch.eye(bs, dtype=torch.float32).unsqueeze(0).repeat(num_partitions, 1, 1),
            requires_grad=False,
        )
        set_weight_attrs(
            hadamard_matrix,
            {"weight_loader": _hadamard_weight_loader, "ignore_warning": True},
        )
        layer.register_parameter("hadamard_matrix", hadamard_matrix)

        # Pre-unpacked bf16 dense weight placeholder (filled after loading).
        weight_dense_qdq = Parameter(torch.empty(1, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(weight_dense_qdq, {"ignore_warning": True})
        layer.register_parameter("weight_dense_qdq", weight_dense_qdq)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.weight_packed.device
        bs = self.quant_config.block_size
        partition_sizes = layer._output_partition_sizes
        num_partitions = len(partition_sizes)
        target_dtype = getattr(layer, "params_dtype", torch.bfloat16)

        # Resolve one block Hadamard Hᵢ per partition, then precompute Rᵢ = Hᵢᵀ
        # for the online inverse input rotation (Hᵢ orthonormal -> Hᵢ⁻¹ = Hᵢᵀ).
        # Uniformity is detected in fp32 for an exact comparison; the rotation is
        # then stored in the activation dtype so the per-forward ``.to(x.dtype)``
        # is a no-op (no copy) in the common case.
        if self.quant_config.hadamard_type == HADAMARD_TYPE_RANDOM:
            mats = layer.hadamard_matrix.data.to(device=device, dtype=torch.float32)
            R_list_fp32 = [mats[i].t().contiguous() for i in range(num_partitions)]
        else:
            # Deterministic Sylvester is shared and symmetric (Hᵀ = H).
            H = deterministic_hadamard_matrix(bs, dtype=torch.float32, device=device)
            R_list_fp32 = [H.contiguous() for _ in range(num_partitions)]

        uniform = all(torch.equal(R_list_fp32[0], R) for R in R_list_fp32)
        layer._block_size = bs
        layer._rotation_uniform = uniform

        # Pre-unpack MXFP4 weight to dense bf16 once (preunpack_bf16 runtime).
        weight_dense = dequant_packed_mxfp4_weight(
            layer.weight_packed.data,
            layer.weight_scale.data,
            self.quant_config.group_size,
            target_dtype=target_dtype,
        )
        layer.weight_dense_qdq = Parameter(weight_dense, requires_grad=False)

        if uniform:
            layer._hadamard_R = R_list_fp32[0].to(target_dtype)
            layer._hadamard_R_list = None
            layer._weight_partitions = None
        else:
            layer._hadamard_R = None
            layer._hadamard_R_list = torch.stack(R_list_fp32, dim=0).to(target_dtype)
            # Pre-slice the dense weight into per-partition contiguous views and
            # precompute bias row offsets, so the slow path avoids re-slicing the
            # weight on every forward. Row slices of a row-major 2D tensor are
            # already contiguous and share storage (no extra memory).
            offsets = []
            partitions = []
            start = 0
            for psize in partition_sizes:
                offsets.append((start, start + psize))
                partitions.append(layer.weight_dense_qdq.data[start:start + psize])
                start += psize
            layer._partition_offsets = offsets
            layer._weight_partitions = partitions

        # Drop the now-unused packed storage and the raw Hadamard buffer.
        clear_packed_storage(layer)
        layer.hadamard_matrix = Parameter(
            torch.empty(1, dtype=torch.float32, device=device), requires_grad=False
        )

        self._log_state_once(layer, uniform, num_partitions)

    def _log_state_once(self, layer: torch.nn.Module, uniform: bool, num_partitions: int) -> None:
        logger.info_once(
            "vllm-qdq-plugin: Hadamard MXFP4 rotation prepared "
            "(type=%s, block_size=%d, online input rotation=Hᵀ, "
            "per-partition=%s, runtime=preunpack_bf16)",
            self.quant_config.hadamard_type,
            self.quant_config.block_size,
            "uniform(fast)" if uniform else f"distinct x{num_partitions}(faithful)",
        )

    def _rotate_qdq(self, x: torch.Tensor, R: torch.Tensor, bs: int) -> torch.Tensor:
        """Block-diagonal inverse Hadamard (x @ Hᵀ per bs-group) + MXFP4 act QDQ."""
        shape = x.shape
        x = (x.reshape(*shape[:-1], -1, bs) @ R).reshape(shape)
        return torch.ops.vllm_qdq_plugin.hadamard_mxfp4_act_qdq(x, self.quant_config.group_size)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # No logging here: this runs inside vLLM's torch.compile graph.
        bs = layer._block_size

        if layer._rotation_uniform:
            # Fast path: single rotation + qdq + merged GEMM. ``_hadamard_R`` is
            # stored in the activation dtype, so ``.to(x.dtype)`` is a no-op.
            R = layer._hadamard_R.to(dtype=x.dtype)
            xq = self._rotate_qdq(x, R, bs)
            return dense_linear_stable_dtype(layer.weight_dense_qdq, xq, bias)

        # Faithful path: distinct per-partition Hadamards. Process each output
        # partition independently (matching auto-round's separate q/k/v modules):
        #   yᵢ = qdq(x @ Hᵢᵀ) @ Sᵢᵀ
        # Weight partitions are pre-sliced contiguous views (see load step).
        R_list = layer._hadamard_R_list.to(dtype=x.dtype)
        partitions = layer._weight_partitions
        offsets = layer._partition_offsets
        outputs = []
        for idx, W_part in enumerate(partitions):
            xq = self._rotate_qdq(x, R_list[idx], bs)
            b_part = None
            if bias is not None:
                row_start, row_end = offsets[idx]
                b_part = bias[row_start:row_end]
            outputs.append(dense_linear_stable_dtype(W_part, xq, b_part))
        return torch.cat(outputs, dim=-1)


def _hadamard_weight_loader(param, loaded_weight, loaded_shard_id=None, *args, **kwargs):
    """Load a per-Linear Hadamard matrix into its partition slot.

    auto-round stores one ``hadamard_matrix`` (block_size x block_size) per
    ORIGINAL Linear. vLLM merges q/k/v -> qkv_proj and gate/up -> gate_up_proj
    and calls this loader once per constituent with a ``loaded_shard_id``:
      * QKVParallelLinear           -> "q" / "k" / "v"
      * MergedColumnParallelLinear  -> 0 / 1 (already the partition index)
      * plain Column/RowParallelLinear -> None (single partition)
    We place each matrix into its own slot so distinct per-Linear (true random)
    matrices are preserved rather than overwritten last-shard-wins.
    """
    data = param.data
    bs = data.shape[-1]
    mat = loaded_weight.to(dtype=data.dtype).reshape(bs, bs)

    if loaded_shard_id is None:
        idx = 0
    elif isinstance(loaded_shard_id, str):
        idx = _QKV_SHARD_TO_IDX.get(loaded_shard_id, 0)
    else:
        idx = int(loaded_shard_id)

    if data.dim() == 3 and 0 <= idx < data.shape[0]:
        data[idx].copy_(mat)
    elif data.dim() == 2:
        data.copy_(mat)
