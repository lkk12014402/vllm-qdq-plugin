# SPDX-License-Identifier: Apache-2.0
"""QuaRot per-Linear (block-diagonal) Hadamard + MXFP4 vLLM ``QuantizationConfig``.

Targets auto-round checkpoints quantized with ``rotation_config`` (the QuaRot
``transform`` backend), i.e. ``rotation_config="default"`` (deterministic
Hadamard) or ``rotation_config="random_hadamard"``. This is **distinct** from
SpinQuant (``spinquant_config`` / R1-R4) handled by :mod:`.config`:

  * SpinQuant rotates the residual stream with one large hidden-size /
    intermediate-size rotation at a few specific points.
  * Per-Linear Hadamard fuses a small ``block_size x block_size`` (= group_size,
    32 for MXFP4) **block-diagonal** Hadamard into **every** Linear's weight and
    replays the inverse Hadamard on that Linear's input activation online.

Equivalence with auto-round HF inference
----------------------------------------
auto-round registers, per quantized Linear, a forward pre-hook that applies the
inverse block-Hadamard to the input (``apply_rotation_hooks_from_config`` ->
``apply_rotation_transform(..., location="input")``); the qmodule itself only
does activation QDQ + weight dequant + GEMM. This plugin reproduces exactly that
math inside the vLLM ``LinearMethod`` forward:

    y = mxfp4_act_qdq(block_rotate(x, Hᵀ)) @ dequant(W_packed, W_scale)ᵀ

where ``H`` is the 32x32 Hadamard fused into the weight at quantization time and
``W`` on disk is already ``W @ Hᵀ`` (block-diagonal). Since ``H`` is orthonormal,
``(x Hᵀ)(W Hᵀ)ᵀ = x Wᵀ`` up to MXFP4 quantization error.

Enable with ``VLLM_HADAMARD_MXFP4=1``.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)

HADAMARD_TYPE_DETERMINISTIC = "hadamard"
HADAMARD_TYPE_RANDOM = "random_hadamard"
_VALID_HADAMARD_TYPES = {HADAMARD_TYPE_DETERMINISTIC, HADAMARD_TYPE_RANDOM}


@register_quantization_config("hadamard_mxfp4")
class HadamardMXFP4Config(QuantizationConfig):
    """Quantization config for QuaRot per-Linear block-Hadamard + MXFP4."""

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 32,
        block_size: int = 32,
        hadamard_type: str = HADAMARD_TYPE_DETERMINISTIC,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.block_size = block_size
        if hadamard_type not in _VALID_HADAMARD_TYPES:
            raise ValueError(
                f"Unsupported hadamard_type {hadamard_type!r}. "
                f"Expected one of {sorted(_VALID_HADAMARD_TYPES)}."
            )
        self.hadamard_type = hadamard_type

    def __repr__(self) -> str:
        return (
            f"HadamardMXFP4Config(bits={self.bits}, group_size={self.group_size}, "
            f"block_size={self.block_size}, hadamard_type={self.hadamard_type})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "hadamard_mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # Volta+

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HadamardMXFP4Config":
        rot = config.get("rotation_config", {}) or {}
        hadamard_type = rot.get("hadamard_type", HADAMARD_TYPE_DETERMINISTIC)
        block_size = int(rot.get("block_size", config.get("group_size", 32)))
        instance = cls(
            bits=config.get("bits", 4),
            group_size=config.get("group_size", 32),
            block_size=block_size,
            hadamard_type=hadamard_type,
        )
        matrix_src = "loaded from checkpoint" if hadamard_type == HADAMARD_TYPE_RANDOM \
            else "regenerated (Sylvester)"
        logger.info(
            "vllm-qdq-plugin: Hadamard MXFP%d (group_size=%d) | per-Linear block-diagonal "
            "Hadamard [type=%s, block_size=%d, matrix=%s] | online input rotation (Hᵀ) + "
            "activation_qdq=enabled(triton-match) | runtime=preunpack_bf16 (cuBLAS F.linear)",
            instance.bits, instance.group_size, instance.hadamard_type,
            instance.block_size, matrix_src,
        )
        return instance

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
    ) -> str | None:
        """Auto-detect per-Linear Hadamard MXFP4 checkpoints."""
        if user_quant == "hadamard_mxfp4":
            return "hadamard_mxfp4"
        rot = hf_quant_cfg.get("rotation_config", {}) or {}
        if not rot or rot.get("hadamard_type") not in _VALID_HADAMARD_TYPES:
            return None
        # Do not collide with the SpinQuant config (spinquant_config present).
        if hf_quant_cfg.get("spinquant_config"):
            return None
        data_type = str(hf_quant_cfg.get("data_type", "")).lower()
        if "mxfp" in data_type or "mx_fp" in data_type:
            return "hadamard_mxfp4"
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from .perlinear_linear_method import HadamardMXFP4LinearMethod

        if isinstance(layer, LinearBase):
            return HadamardMXFP4LinearMethod(self)
        return None
