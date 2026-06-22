# SPDX-License-Identifier: Apache-2.0
"""Shared base for the MXFP4 rotation ``QuantizationConfig`` classes.

Both :class:`.config.SpinQuantMXFP4Config` and
:class:`.perlinear_config.HadamardMXFP4Config` expose the same vLLM
``QuantizationConfig`` capability surface (supported act dtypes, min capability,
config filename) and both gate auto-detection on the checkpoint's ``data_type``
being an MXFP variant. This base centralizes that shared boilerplate.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

__all__ = ["MXFP4RotationConfigBase", "data_type_is_mxfp"]


def data_type_is_mxfp(hf_quant_cfg: dict[str, Any]) -> bool:
    """True when the checkpoint's ``data_type`` denotes an MXFP format."""
    data_type = str(hf_quant_cfg.get("data_type", "")).lower()
    return "mxfp" in data_type or "mx_fp" in data_type


class MXFP4RotationConfigBase(QuantizationConfig):
    """Shared capability surface for the MXFP4 rotation configs."""

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # Volta+

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json"]
