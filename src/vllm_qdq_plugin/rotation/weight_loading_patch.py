# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch for vLLM's ``AutoWeightsLoader`` to tolerate SpinQuant keys.

auto-round serializes some rotation matrices (e.g. ``spinquant_R2_head``) as
top-level keys in the safetensors checkpoint. vLLM's ``AutoWeightsLoader`` cannot
map these to any model module/parameter and raises ``ValueError``. We patch its
``__init__`` to always include ``"spinquant_R"`` in ``ignore_unexpected_prefixes``
so those top-level keys are silently skipped.

Note: R2 itself is already fused offline into v_proj/o_proj weights at
quantization time, so the leftover ``spinquant_R2_head`` key is informational
only and safe to ignore at inference time.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCH_APPLIED = False


def apply_weight_loading_patch() -> None:
    """Patch ``AutoWeightsLoader`` to ignore spinquant rotation matrix keys."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    _PATCH_APPLIED = True

    try:
        from vllm.model_executor.models.utils import AutoWeightsLoader
    except ImportError:
        logger.warning(
            "vllm-qdq-plugin: could not import AutoWeightsLoader from vllm; "
            "weight loading patch not applied."
        )
        return

    _orig_init = AutoWeightsLoader.__init__

    def _patched_init(
        self,
        module,
        *,
        skip_prefixes=None,
        skip_substrs=None,
        ignore_unexpected_prefixes=None,
        ignore_unexpected_suffixes=None,
    ):
        if ignore_unexpected_prefixes is None:
            ignore_unexpected_prefixes = []
        else:
            ignore_unexpected_prefixes = list(ignore_unexpected_prefixes)

        if not any(p.startswith("spinquant_R") for p in ignore_unexpected_prefixes):
            ignore_unexpected_prefixes.append("spinquant_R")

        _orig_init(
            self,
            module,
            skip_prefixes=skip_prefixes,
            skip_substrs=skip_substrs,
            ignore_unexpected_prefixes=ignore_unexpected_prefixes,
            ignore_unexpected_suffixes=ignore_unexpected_suffixes,
        )

    AutoWeightsLoader.__init__ = _patched_init
    logger.info(
        "vllm-qdq-plugin: applied SpinQuant weight loading patch "
        "(AutoWeightsLoader will ignore 'spinquant_R*' top-level keys)."
    )
