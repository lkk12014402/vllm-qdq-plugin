# SPDX-License-Identifier: Apache-2.0
"""QuaRot/SpinQuant rotation + MXFP4 vLLM inference support (standalone).

Vendored from auto-round's ``auto_round.vllm_plugin`` with zero auto_round
dependency. Provides a vLLM ``QuantizationConfig`` (``spinquant_mxfp4``) plus
``LinearMethodBase`` implementations that perform online R1/R4 rotation and
MXFP4 activation QDQ at inference time. R2 is fused offline into the weights at
quantization time, and R3 (post-RoPE) is not supported here.

Enable by setting ``VLLM_SPINQUANT_MXFP4=1`` before vLLM imports the plugin.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_REGISTERED = False


def register_spinquant_mxfp4() -> None:
    """Register the ``spinquant_mxfp4`` quantization config and supporting patches.

    Idempotent. Safe to call in every process (main + workers).
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    from .mxfp4 import register_custom_ops
    from .weight_loading_patch import apply_weight_loading_patch

    # Register torch custom ops (spinquant_mxfp4_act_qdq / spinquant_mxfp4_linear).
    register_custom_ops()

    # Importing config triggers @register_quantization_config("spinquant_mxfp4").
    from . import config  # noqa: F401

    # Tolerate top-level spinquant_R* keys in the checkpoint.
    apply_weight_loading_patch()

    logger.info(
        "vllm-qdq-plugin: registered 'spinquant_mxfp4' quantization config "
        "(VLLM_SPINQUANT_MXFP4 enabled)"
    )


def __getattr__(name: str):
    # Lazy export to avoid importing vLLM quantization machinery at package import.
    if name == "SpinQuantMXFP4Config":
        from .config import SpinQuantMXFP4Config

        return SpinQuantMXFP4Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["register_spinquant_mxfp4", "SpinQuantMXFP4Config"]
