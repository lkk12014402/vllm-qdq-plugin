# SPDX-License-Identifier: Apache-2.0
"""SpinQuant/QuaRot MXFP4 vLLM ``QuantizationConfig`` (vendored, standalone).

Ported from auto-round's ``auto_round.vllm_plugin.spinquant_mxfp4`` with zero
auto_round dependency. Registers a ``spinquant_mxfp4`` quantization method that
auto-detects auto-round SpinQuant/QuaRot + MXFP4 checkpoints and replays the
online R1/R4 activation rotation at inference time.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .constants import (
    RUNTIME_BACKEND_PACKED_FUSED,
    RUNTIME_BACKEND_PREUNPACK_BF16,
    RUNTIME_BACKEND_PREUNPACK_FP8,
    VALID_RUNTIME_BACKENDS,
)
from .hadamard import describe_hadamard

logger = init_logger(__name__)


def _resolve_runtime_backend(config: dict[str, Any]) -> str:
    """Resolve runtime backend from env or config.

    Precedence: ``VLLM_SPINQUANT_RUNTIME_BACKEND`` env (or legacy
    ``AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND``) > ``spinquant_config.runtime_backend``
    > top-level ``runtime_backend`` > default ``preunpack_bf16``.
    """
    sq_config = config.get("spinquant_config", {})
    backend = (
        os.getenv("VLLM_SPINQUANT_RUNTIME_BACKEND")
        or os.getenv("AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND")
        or sq_config.get("runtime_backend")
        or config.get("runtime_backend")
        or RUNTIME_BACKEND_PREUNPACK_BF16
    )
    if backend not in VALID_RUNTIME_BACKENDS:
        raise ValueError(
            f"Unsupported SpinQuant runtime backend {backend!r}. "
            f"Expected one of {sorted(VALID_RUNTIME_BACKENDS)}."
        )
    return backend


@register_quantization_config("spinquant_mxfp4")
class SpinQuantMXFP4Config(QuantizationConfig):
    """Quantization config for SpinQuant/QuaRot online rotation + MXFP4."""

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 32,
        online_r1: bool = True,
        online_r4: bool = False,
        online_r3: bool = False,
        r1_type: str = "hadamard",
        r4_type: str = "hadamard",
        r3_type: str = "hadamard",
        rotation_size: int | None = None,
        hidden_size: int = 0,
        head_dim: int = 128,
        intermediate_size: int = 0,
        runtime_backend: str = RUNTIME_BACKEND_PREUNPACK_BF16,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.online_r1 = online_r1
        self.online_r4 = online_r4
        self.online_r3 = online_r3
        self.r1_type = r1_type
        self.r4_type = r4_type
        self.r3_type = r3_type
        self.rotation_size = rotation_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.runtime_backend = runtime_backend
        # Populated by from_config().
        self.selective_rotation = False
        self.rotated_layers: set[str] = set()
        self._unrotated_suffixes: set[str] = set()

    def __repr__(self) -> str:
        parts = [
            f"bits={self.bits}",
            f"group_size={self.group_size}",
            f"online_r1={self.online_r1}",
            f"r1_type={self.r1_type}",
            f"runtime_backend={self.runtime_backend}",
        ]
        if self.online_r4:
            parts.append(f"online_r4={self.online_r4}, r4_type={self.r4_type}")
        if self.online_r3:
            parts.append(f"online_r3={self.online_r3}, r3_type={self.r3_type}")
        return f"SpinQuantMXFP4Config({', '.join(parts)})"

    @classmethod
    def get_name(cls) -> str:
        return "spinquant_mxfp4"

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
    def from_config(cls, config: dict[str, Any]) -> "SpinQuantMXFP4Config":
        """Parse from a model's ``quantization_config``."""
        sq_config = config.get("spinquant_config", {})
        instance = cls(
            bits=config.get("bits", 4),
            group_size=config.get("group_size", 32),
            online_r1=sq_config.get("online_r1_rotation", True),
            online_r4=sq_config.get("r4", False),
            online_r3=sq_config.get("r3", False),
            r1_type="random" if sq_config.get("random_r1", False) else "hadamard",
            r4_type="random" if sq_config.get("random_r4", False) else "hadamard",
            r3_type="random" if sq_config.get("random_r3", False) else "hadamard",
            rotation_size=sq_config.get("rotation_size", None),
            hidden_size=sq_config.get("hidden_size", 0),
            head_dim=sq_config.get("head_dim", 128),
            intermediate_size=sq_config.get("intermediate_size", 0),
            runtime_backend=_resolve_runtime_backend(config),
        )
        # Selective rotation info for load-time weight compensation.
        instance.selective_rotation = sq_config.get("selective_rotation", False)
        instance.rotated_layers = set(sq_config.get("rotated_layers", []))
        # Determine which layer-type suffixes are NOT rotated (merged-layer compensation).
        instance._unrotated_suffixes = set()
        if instance.selective_rotation and instance.rotated_layers:
            for suffix in ("v_proj", "q_proj", "k_proj", "gate_proj", "up_proj"):
                if not any(name.endswith(suffix) for name in instance.rotated_layers):
                    instance._unrotated_suffixes.add(suffix)
            if instance._unrotated_suffixes:
                logger.info(
                    "vllm-qdq-plugin: Selective rotation: unrotated layer types "
                    "%s will be compensated in merged layers",
                    instance._unrotated_suffixes,
                )

        # Log only active rotations for this specific model.
        active = []
        if instance.online_r1:
            r1_size = instance.rotation_size or instance.hidden_size
            active.append(f"R1(online, {describe_hadamard(instance.r1_type, r1_size)})")
        if sq_config.get("r2", False):
            active.append("R2(offline, fused into weights)")
        if instance.online_r3:
            active.append(f"R3(online, {describe_hadamard(instance.r3_type, instance.head_dim)})")
        if instance.online_r4:
            r4_size = instance.rotation_size or instance.intermediate_size
            active.append(f"R4(online, {describe_hadamard(instance.r4_type, r4_size)})")
        rot_str = ", ".join(active) if active else "none"
        if sq_config.get("selective_rotation", False):
            num_rotated = sq_config.get("num_rotated_layers", "?")
            rot_str += f" [SELECTIVE: {num_rotated} layers rotated]"

        gemm_lib_map = {
            RUNTIME_BACKEND_PACKED_FUSED: "Triton fused kernel",
            RUNTIME_BACKEND_PREUNPACK_BF16: "cuBLAS (torch.nn.functional.linear)",
            RUNTIME_BACKEND_PREUNPACK_FP8: "CUTLASS FP8 (vllm.cutlass_scaled_mm)",
        }
        gemm_lib = gemm_lib_map.get(instance.runtime_backend, "unknown")
        logger.info(
            "vllm-qdq-plugin: SpinQuant MXFP%d (group_size=%d) | Online rotations: [%s] | "
            "activation_qdq=enabled(even) | runtime_backend=%s (%s) | hidden_size=%d, intermediate_size=%d",
            instance.bits, instance.group_size, rot_str, instance.runtime_backend, gemm_lib,
            instance.hidden_size, instance.intermediate_size,
        )
        return instance

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
    ) -> str | None:
        """Auto-detect spinquant_mxfp4 models from their config."""
        if user_quant == "spinquant_mxfp4":
            return "spinquant_mxfp4"
        sq = hf_quant_cfg.get("spinquant_config", {})
        if not sq or not sq.get("online_r1_rotation", False):
            return None
        data_type = hf_quant_cfg.get("data_type", "").lower()
        if "mxfp" in data_type or "mx_fp" in data_type:
            return "spinquant_mxfp4"
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # Lazy import to avoid a circular import (linear_method imports config
        # indirectly via constants/mxfp4).
        from .linear_method import (
            SpinQuantMXFP4PackedFusedLinearMethod,
            SpinQuantMXFP4PreunpackBF16LinearMethod,
            SpinQuantMXFP4PreunpackFP8LinearMethod,
        )

        if isinstance(layer, LinearBase):
            if self.runtime_backend == RUNTIME_BACKEND_PREUNPACK_BF16:
                return SpinQuantMXFP4PreunpackBF16LinearMethod(self)
            if self.runtime_backend == RUNTIME_BACKEND_PREUNPACK_FP8:
                return SpinQuantMXFP4PreunpackFP8LinearMethod(self)
            return SpinQuantMXFP4PackedFusedLinearMethod(self)
        return None
