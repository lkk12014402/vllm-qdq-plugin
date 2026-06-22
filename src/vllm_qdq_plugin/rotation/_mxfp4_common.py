# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the MXFP4 rotation LinearMethods (SpinQuant + Hadamard).

Both :mod:`.linear_method` (SpinQuant/QuaRot R1/R4) and
:mod:`.perlinear_linear_method` (per-Linear block Hadamard) register the same
packed MXFP4 weight/scale parameters, run the same dtype-stable dense GEMM, and
free the packed storage the same way once a pre-unpacked backend is prepared.
These helpers centralize that logic so the two methods cannot drift apart.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.utils import set_weight_attrs

__all__ = [
    "register_mxfp4_packed_weight",
    "dense_linear_stable_dtype",
    "clear_packed_storage",
]


def register_mxfp4_packed_weight(
    layer: torch.nn.Module,
    *,
    output_size_per_partition: int,
    input_size_per_partition: int,
    group_size: int,
    extra_weight_attrs: dict,
) -> None:
    """Register the packed MXFP4 ``weight_packed`` + e8m0 ``weight_scale`` params.

    Layout (shared by both rotation methods):
      * ``weight_packed``: ``[N, K//2]`` uint8 (two E2M1 nibbles per byte).
      * ``weight_scale`` : ``[N, K//group_size]`` uint8 (e8m0 exponents).
    """
    weight_packed = Parameter(
        torch.empty(output_size_per_partition, input_size_per_partition // 2, dtype=torch.uint8),
        requires_grad=False,
    )
    set_weight_attrs(
        weight_packed,
        {"input_dim": 1, "output_dim": 0, "packed_dim": 1, "pack_factor": 2} | extra_weight_attrs,
    )
    layer.register_parameter("weight_packed", weight_packed)

    weight_scale = Parameter(
        torch.empty(output_size_per_partition, input_size_per_partition // group_size, dtype=torch.uint8),
        requires_grad=False,
    )
    set_weight_attrs(weight_scale, {"input_dim": 1, "output_dim": 0} | extra_weight_attrs)
    layer.register_parameter("weight_scale", weight_scale)


def dense_linear_stable_dtype(
    weight: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``F.linear`` in the weight's dtype while preserving the input dtype.

    Activations may arrive in a different dtype than the (pre-dequantized) dense
    weight; compute in the weight's dtype and cast the result back so the layer
    output dtype is stable for the rest of the graph.
    """
    input_dtype = x.dtype
    compute_dtype = weight.dtype
    if x.dtype != compute_dtype:
        x = x.to(compute_dtype)
    if bias is not None and bias.dtype != compute_dtype:
        bias = bias.to(compute_dtype)
    output = torch.nn.functional.linear(x, weight, bias)
    if output.dtype != input_dtype:
        output = output.to(input_dtype)
    return output


def clear_packed_storage(layer: torch.nn.Module) -> None:
    """Replace packed weight/scale storage with 1-element dummies.

    Used after a pre-unpacked backend (bf16/fp8) has consumed the packed tensors.
    Keeping 1-element placeholders (instead of deleting the attributes) means
    torch.compile cached graphs never hit a missing key — a shape guard simply
    triggers a healthy re-compilation.
    """
    device = "cpu"
    if getattr(layer, "weight_packed", None) is not None:
        device = layer.weight_packed.device
        layer.weight_packed = Parameter(
            torch.empty(1, dtype=torch.uint8, device=device), requires_grad=False
        )
    if getattr(layer, "weight_scale", None) is not None:
        layer.weight_scale = Parameter(
            torch.empty(1, dtype=torch.uint8, device=device), requires_grad=False
        )
