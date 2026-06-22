# SPDX-License-Identifier: Apache-2.0
"""MXFP4 (microscaling FP4) dequant + activation QDQ (vendored, standalone).

Ported from auto-round's ``auto_round.vllm_plugin.spinquant_mxfp4`` so this
plugin has **zero auto_round dependency**.

Format (as exported by auto-round):
  - ``weight_packed``: ``[N, K//2]`` uint8 -- two E2M1 FP4 values per byte
    (low nibble = even column, high nibble = odd column).
  - ``weight_scale``:  ``[N, K//32]`` uint8 -- e8m0 shared exponents
    (``scale = 2^(e - 127)``), one per group of 32.

Two custom ops are registered under the ``vllm_qdq_plugin`` torch library:
  - ``vllm_qdq_plugin::spinquant_mxfp4_act_qdq`` -- MXFP4 activation QDQ.
  - ``vllm_qdq_plugin::spinquant_mxfp4_linear``  -- packed MXFP4 dequant + GEMM.

The activation QDQ uses Quark's "even" scale mode (round the group max before
extracting the exponent) so it matches the quantization-time semantics exactly.
The packed linear path uses the vendored Triton fused kernel when available,
falling back to a pure-PyTorch dequant + ``F.linear``.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MXFP4_BLOCK_SIZE = 32
FP4_E2M1_MAX = 6.0
F32_MIN_NORMAL = 2 ** (-126)

# E2M1 FP4 magnitude lookup (index 0..7 -> value); sign is the high bit.
_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


# =============================================================================
# Weight unpack / dequant helpers
# =============================================================================


def unpack_mxfp4_values(weight_packed: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Unpack packed E2M1 MXFP4 values to unscaled floating-point values."""
    N, K_half = weight_packed.shape
    K = K_half * 2

    low = (weight_packed & 0x0F).to(torch.int32)
    high = ((weight_packed >> 4) & 0x0F).to(torch.int32)
    unpacked = torch.stack([low, high], dim=-1).reshape(N, K)

    e2m1_lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=weight_packed.device)
    sign = torch.where(unpacked >= 8, -1.0, 1.0)
    mag_idx = unpacked & 0x07
    abs_val = e2m1_lut[mag_idx]
    return (abs_val * sign).to(target_dtype)


def e8m0_to_scale(weight_scale: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Convert e8m0 exponents to floating-point scale values."""
    return torch.pow(2.0, weight_scale.to(torch.int32).float() - 127.0).to(target_dtype)


def dequant_packed_mxfp4_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Fully dequantize packed MXFP4 weights to a dense floating-point tensor."""
    N, K_half = weight_packed.shape
    K = K_half * 2
    fp_values = unpack_mxfp4_values(weight_packed, target_dtype=target_dtype)
    scale_float = e8m0_to_scale(weight_scale, target_dtype=target_dtype)
    fp_values = fp_values.reshape(N, -1, group_size)
    return (fp_values * scale_float.unsqueeze(-1)).reshape(N, K)


def preunpack_mxfp4_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert packed MXFP4 weight to an unpacked FP8 + scale representation."""
    fp_values = unpack_mxfp4_values(weight_packed, target_dtype=torch.float32)
    weight_fp8 = fp_values.to(torch.float8_e4m3fn)
    scale_bf16 = e8m0_to_scale(weight_scale, target_dtype=torch.bfloat16).reshape(-1, 1)
    if group_size != MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"preunpack_fp8 backend currently expects group_size={MXFP4_BLOCK_SIZE}, got {group_size}"
        )
    return weight_fp8, scale_bf16


def dequant_preunpacked_mxfp4_weight(
    weight_fp8: torch.Tensor,
    scale_bf16: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize unpacked FP8 + scale weights back to a dense tensor for GEMM."""
    origin_shape = weight_fp8.shape
    weight_fp8 = weight_fp8.reshape(-1, MXFP4_BLOCK_SIZE)
    scale = scale_bf16.reshape(-1, 1).to(target_dtype)
    return (weight_fp8.to(target_dtype) * scale).reshape(origin_shape)


# =============================================================================
# Activation QDQ (pure PyTorch, Quark "even" mode aligned)
# =============================================================================


def _fp4_121_positive(x: torch.Tensor) -> torch.Tensor:
    """Round positive values to the E2M1 FP4 grid."""
    half_step = torch.round(2.0 * x) / 2.0
    unit_step = torch.round(x)
    two_step = 2.0 * torch.round(x / 2.0)

    below_two = x < 2.0
    below_four = x < 4.0
    return (
        half_step * below_two
        + unit_step * (~below_two) * below_four
        + two_step * (~below_two) * (~below_four)
    )


def mxfp4_act_qdq(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pure-PyTorch MXFP4 activation QDQ aligned with Quark/vllm-ext "even" mode.

    Equivalence note:
        This is **bit-exact identical** to the native QDQ-simulation path
        ``vllm_qdq_plugin.qdq.mxfp4.mxfp4_qdq`` (vLLM's reference MXFP4) for all
        fp16/bf16 inputs — both quantize to E2M1 values with E8M0 (power-of-2,
        round-max-then-floor-log2-minus-2) per-group scales. This is verified in
        ``tests/test_rotation_qdq_equivalence.py``. We keep a separate vendored
        copy here so the rotation subpackage stays fully standalone (zero
        cross-subpackage dependency), and because the "even" naming makes the
        alignment with auto-round/Quark's export semantics explicit.

    Unlike the native 2D-only helper, this accepts arbitrary-rank inputs (the
    last dim must be divisible by ``group_size``) and any input dtype.
    """
    if group_size <= 0 or x.shape[-1] % group_size != 0:
        raise ValueError(
            f"MXFP4 activation qdq requires the last dim to be divisible by group_size, "
            f"got shape={tuple(x.shape)}, group_size={group_size}"
        )

    original_dtype = x.dtype
    original_shape = x.shape
    x_fp32 = x.to(torch.float32).reshape(-1, group_size)

    sign = x_fp32.sign()
    x_abs = x_fp32.abs()
    amax = x_abs.amax(dim=-1, keepdim=True)

    # Match Quark's "even" scale mode: round the max value before extracting the exponent.
    rounded_bits = (amax.contiguous().view(torch.int32) + 0x200000) & 0x7F800000
    rounded_max = rounded_bits.view(torch.float32)
    safe_max = torch.where(rounded_max > 0, rounded_max, torch.full_like(rounded_max, F32_MIN_NORMAL))

    scale_exp = torch.floor(torch.log2(safe_max)) - 2.0
    scale_exp = torch.clamp(scale_exp, min=-127, max=127)
    scale = torch.pow(2.0, scale_exp)
    scale = torch.where(torch.isfinite(scale) & (scale > 0), scale, torch.ones_like(scale))

    x_scaled = x_abs / scale
    x_fp4 = _fp4_121_positive(x_scaled).clamp(max=FP4_E2M1_MAX)
    x_qdq = (sign * x_fp4 * scale).reshape(original_shape)
    return x_qdq.to(original_dtype)


def mxfp4_act_qdq_hadamard(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pure-PyTorch MXFP4 activation QDQ matching auto-round's QuaRot ``transform``
    (triton) inference path bit-for-bit.

    Equivalence note:
        This reproduces ``auto_round.algorithms.transforms.quarot.utils.triton.
        mxfp4.mxfp4_forward_kernel`` (``quest=False`` branch) — the kernel used by
        the per-Linear / random Hadamard HF inference path when triton is
        available (``pre_dequantized_input=True``). Verified against the real
        forward (cosine ~0.9999995, max abs diff ~6e-4, the residual being the
        kernel's bf16 ``tl.dot`` rotation vs this fp32 rotation).

        Two details distinguish it from both :func:`mxfp4_act_qdq` (Quark "even")
        and auto-round's non-triton ``quant_mx``:
          1. per-group scale = ``2^(floor(log2(amax)) - 2) / 0.75`` (the ``/0.75``
             expands the dynamic range so amax maps near 3-4 on the FP4 grid);
          2. round-to-nearest with explicit midpoint thresholds on the E2M1 grid
             ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}`` (NOT round-half-to-even).

    The Hadamard rotation itself is applied by the caller (the LinearMethod), so
    this function only does the per-group scale + FP4 round + dequant.
    """
    if group_size <= 0 or x.shape[-1] % group_size != 0:
        raise ValueError(
            f"MXFP4 activation qdq requires the last dim to be divisible by group_size, "
            f"got shape={tuple(x.shape)}, group_size={group_size}"
        )

    original_dtype = x.dtype
    original_shape = x.shape
    x_fp32 = x.to(torch.float32).reshape(-1, group_size)

    amax = x_fp32.abs().amax(dim=-1, keepdim=True)
    safe_max = torch.where(amax == 0, torch.ones_like(amax), amax)
    # shared_exp = 2^(floor(log2(amax)) - 2) / (3/4)   [matches triton kernel]
    shared_exp = torch.exp2(torch.floor(torch.log2(safe_max)) - 2.0) / 0.75
    shared_exp = torch.where(shared_exp > 0, shared_exp, torch.ones_like(shared_exp))

    x_scaled = x_fp32 / shared_exp
    a = x_scaled.abs()
    sign = torch.where(x_scaled > 0, 1.0, -1.0)
    # Round to nearest E2M1 grid point with the kernel's midpoint thresholds.
    fp4 = torch.where(
        a > 5.0, 6.0,
        torch.where(
            a > 3.5, 4.0,
            torch.where(
                a > 2.5, 3.0,
                torch.where(
                    a > 1.75, 2.0,
                    torch.where(
                        a > 1.25, 1.5,
                        torch.where(
                            a > 0.75, 1.0,
                            torch.where(a > 0.25, 0.5, torch.zeros_like(a)),
                        ),
                    ),
                ),
            ),
        ),
    )
    x_qdq = (sign * fp4 * shared_exp).reshape(original_shape)
    return x_qdq.to(original_dtype)


# =============================================================================
# Optional Triton fused dequant + GEMM
# =============================================================================

try:
    from .triton_gemm import triton_mxfp4_gemm as _triton_mxfp4_gemm

    _HAS_TRITON_MXFP4 = True
except (ImportError, RuntimeError):
    _HAS_TRITON_MXFP4 = False
    _triton_mxfp4_gemm = None


def _mxfp4_dequant_linear_fallback(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Pure-PyTorch fallback for MXFP4 dequant + linear."""
    w_dequant = dequant_packed_mxfp4_weight(
        weight_packed, weight_scale, group_size, target_dtype=x.dtype
    )
    return torch.nn.functional.linear(x, w_dequant, None)


# =============================================================================
# Custom op implementations
# =============================================================================

_qdq_backend_logged = False

# Activation-QDQ backend selection for the SpinQuant op. Default "even" (Quark
# semantics, what SpinQuant checkpoints expect). "triton" selects the
# triton-kernel-matching rounding (same as the Hadamard path) as an advanced
# override. Resolved once from VLLM_SPINQUANT_MXFP4_QDQ_BACKEND (legacy
# AUTO_ROUND_MXFP4_QDQ_BACKEND); the value changes FP4 rounding semantics.
_QDQ_BACKEND_EVEN = "even"
_QDQ_BACKEND_TRITON = "triton"
_resolved_qdq_backend: str | None = None


def _resolve_qdq_backend() -> str:
    global _resolved_qdq_backend
    if _resolved_qdq_backend is not None:
        return _resolved_qdq_backend
    val = (
        os.getenv("VLLM_SPINQUANT_MXFP4_QDQ_BACKEND")
        or os.getenv("AUTO_ROUND_MXFP4_QDQ_BACKEND")
        or ""
    ).strip().lower()
    if val in ("", "even", "quark"):
        backend = _QDQ_BACKEND_EVEN
    elif val in ("triton", "triton-match", "hadamard"):
        backend = _QDQ_BACKEND_TRITON
    else:
        logger.warning(
            "Unknown VLLM_SPINQUANT_MXFP4_QDQ_BACKEND=%r; falling back to 'even'.", val
        )
        backend = _QDQ_BACKEND_EVEN
    _resolved_qdq_backend = backend
    return backend


def _log_qdq_backend(name: str) -> None:
    global _qdq_backend_logged
    if not _qdq_backend_logged:
        logger.info("MXFP4 activation QDQ backend: %s", name)
        _qdq_backend_logged = True


def _spinquant_mxfp4_act_qdq_impl(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """QDQ the rotated activation to MXFP4 semantics before GEMM.

    Backend selected by ``VLLM_SPINQUANT_MXFP4_QDQ_BACKEND`` (legacy
    ``AUTO_ROUND_MXFP4_QDQ_BACKEND``):
      * ``even`` (default) — Quark "even" mode; what SpinQuant exports expect.
      * ``triton`` — the triton-kernel-matching rounding (advanced override).
    """
    if _resolve_qdq_backend() == _QDQ_BACKEND_TRITON:
        _log_qdq_backend("pytorch (triton-match) [override]")
        return mxfp4_act_qdq_hadamard(x, group_size)
    _log_qdq_backend("pytorch (even)")
    return mxfp4_act_qdq(x, group_size)


def _spinquant_mxfp4_act_qdq_fake(x: torch.Tensor, group_size: int) -> torch.Tensor:
    return torch.empty_like(x)


def _hadamard_mxfp4_act_qdq_impl(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """MXFP4 activation QDQ for the per-Linear Hadamard path.

    Matches auto-round's triton ``mxfp4_forward_kernel`` inference path
    (see :func:`mxfp4_act_qdq_hadamard`).
    """
    _log_qdq_backend("pytorch (hadamard/triton-match)")
    return mxfp4_act_qdq_hadamard(x, group_size)


def _hadamard_mxfp4_act_qdq_fake(x: torch.Tensor, group_size: int) -> torch.Tensor:
    return torch.empty_like(x)


def _spinquant_mxfp4_linear_impl(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Actual MXFP4 dequant + GEMM implementation."""
    if _HAS_TRITON_MXFP4 and x.is_cuda:
        output = _triton_mxfp4_gemm(
            x.float(),
            weight_packed,
            weight_scale,
            bias=None,
            group_size=group_size,
            fp32_precision="tf32",
        )
        return output.to(x.dtype)
    return _mxfp4_dequant_linear_fallback(x, weight_packed, weight_scale, group_size)


def _spinquant_mxfp4_linear_fake(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    N = weight_packed.shape[0]
    return torch.empty(*x.shape[:-1], N, dtype=x.dtype, device=x.device)


# =============================================================================
# Register custom ops with vLLM's library system (idempotent)
# =============================================================================

_OPS_REGISTERED = False
# Keep the Library object alive at module scope: a torch.library.Library that
# is garbage-collected deregisters all of its ops, so it must not be a local.
_CUSTOM_OP_LIB = None


def register_custom_ops() -> None:
    """Register the spinquant MXFP4 custom ops once."""
    global _OPS_REGISTERED, _CUSTOM_OP_LIB
    if _OPS_REGISTERED:
        return
    _OPS_REGISTERED = True

    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import direct_register_custom_op

    lib = torch.library.Library("vllm_qdq_plugin", "DEF")
    _CUSTOM_OP_LIB = lib

    direct_register_custom_op(
        op_name="spinquant_mxfp4_act_qdq",
        op_func=_spinquant_mxfp4_act_qdq_impl,
        mutates_args=[],
        fake_impl=_spinquant_mxfp4_act_qdq_fake,
        target_lib=lib,
        dispatch_key=current_platform.dispatch_key,
    )
    direct_register_custom_op(
        op_name="spinquant_mxfp4_linear",
        op_func=_spinquant_mxfp4_linear_impl,
        mutates_args=[],
        fake_impl=_spinquant_mxfp4_linear_fake,
        target_lib=lib,
        dispatch_key=current_platform.dispatch_key,
    )
    direct_register_custom_op(
        op_name="hadamard_mxfp4_act_qdq",
        op_func=_hadamard_mxfp4_act_qdq_impl,
        mutates_args=[],
        fake_impl=_hadamard_mxfp4_act_qdq_fake,
        target_lib=lib,
        dispatch_key=current_platform.dispatch_key,
    )
