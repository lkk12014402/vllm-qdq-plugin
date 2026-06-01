"""
Pre-quantization transforms: operations applied to Q, K, V before quantization.

Each transform has the signature:
    (q, k, v, ctx: TransformContext) → (q, k, v, ctx: TransformContext)

Transforms are composable via the pre_transforms list on AttentionConfig.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import functools

# Type alias for transform functions
TransformFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, "TransformContext"],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, "TransformContext"],
]


@dataclass
class TransformContext:
    """
    Typed context passed between transforms and to the kernel launcher.

    Carries metadata produced by transforms that downstream stages need
    (e.g., delta_s from smoothing, v_mean from V-smoothing).
    """
    delta_s: Optional[torch.Tensor] = None
    v_mean: Optional[torch.Tensor] = None
    R: Optional[torch.Tensor] = None  # Hadamard rotation matrix [D, D]


# ============================================================================
# QK Smoothing
# ============================================================================

def _pad_128(x: torch.Tensor) -> torch.Tensor:
    """Pad tensor's sequence dimension to a multiple of 128."""
    L = x.size(2)
    pad_len = (128 - L % 128) % 128
    if pad_len == 0:
        return x.contiguous()
    return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()


def qk_smoothing(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: TransformContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, TransformContext]:
    """
    Apply QK smoothing to reduce quantization outliers.

    1. K centering: k_centered = k - mean(k, dim=sequence)
    2. Q per-block smoothing: subtract per-128-token-group mean
    3. Delta correction: delta_s = q_means @ k_centered^T

    The delta_s correction is added back during attention to maintain equivalence.
    """
    B, H, N, D = q.shape

    # Step 1: K centering (lossless)
    k_centered = k - k.mean(dim=-2, keepdim=True)

    # Step 2: Pad to multiple of 128
    q_padded = _pad_128(q)
    k_padded = _pad_128(k_centered)

    # Step 3: Q smoothing with per-block means
    if N >= 128:
        L_pad = q_padded.size(2)
        GROUP_SIZE = 128
        num_groups = L_pad // GROUP_SIZE

        q_grouped = q_padded.view(B, H, num_groups, GROUP_SIZE, D)
        q_means = q_grouped.mean(dim=3, keepdim=False)  # [B, H, num_groups, D]
        q_smoothed_grouped = q_grouped - q_means.unsqueeze(3)
        q_smoothed_full = q_smoothed_grouped.view(B, H, L_pad, D)
    else:
        q_means = q_padded.mean(dim=-2, keepdim=True)  # [B, H, 1, D]
        q_smoothed_full = q_padded - q_means
        num_groups = 1

    # Remove padding
    q_smoothed = q_smoothed_full[:, :, :N, :]
    k_smoothed = k_padded[:, :, :N, :]

    # Step 4: Compute delta_s = q_means @ k^T
    ctx.delta_s = torch.matmul(
        q_means, k_smoothed.transpose(-2, -1)
    ).to(torch.float32).contiguous()

    return q_smoothed, k_smoothed, v, ctx


# ============================================================================
# V Smoothing
# ============================================================================

def v_smoothing(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: TransformContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, TransformContext]:
    """
    Subtract V's mean to reduce outlier impact before quantization.

    The correction (output += v_mean) is applied post-kernel in api.py,
    reading ctx.v_mean. This works because sum(softmax_weights) = 1:
    output = sum(softmax * (v - v_mean)) + v_mean = sum(softmax * v).
    """
    ctx.v_mean = v.mean(dim=-2, keepdim=True)  # [B, H, 1, D]
    v_centered = v - ctx.v_mean
    return q, k, v_centered, ctx


# ============================================================================
# Hadamard Rotation
# ============================================================================

@functools.lru_cache(maxsize=16)
def create_hadamard_matrix(
    block_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    randomized: bool = True,
) -> torch.Tensor:
    """
    Create orthogonal Hadamard matrix using Sylvester construction.

    H(1) = [[1]]
    H(2n) = [[H(n),  H(n) ],
             [H(n), -H(n)]]

    Normalized by 1/sqrt(block_size) to satisfy H × H^T = I.

    Optionally multiply columns by random ±1 signs to prevent pathological alignment.

    Args:
        block_size: Power of 2, typically 32
        device: CUDA device
        dtype: Data type (usually bfloat16 for fast rotation)
        randomized: If True, multiply each column by random ±1 sign

    Returns:
        R: Orthogonal matrix [block_size, block_size]
    """
    assert (block_size & (block_size - 1)) == 0, "block_size must be power of 2"

    def _sylvester(n: int, dev: torch.device, dt: torch.dtype) -> torch.Tensor:
        """Recursive Sylvester construction."""
        if n == 1:
            return torch.ones(1, 1, device=dev, dtype=dt)
        H_half = _sylvester(n // 2, dev, dt)
        half = n // 2
        H = torch.zeros(n, n, device=dev, dtype=dt)
        H[:half, :half] = H_half
        H[:half, half:] = H_half
        H[half:, :half] = H_half
        H[half:, half:] = -H_half
        return H

    dev = torch.device(device)
    H = _sylvester(block_size, dev, dtype)
    R = H / (block_size ** 0.5)

    # Apply randomized Hadamard: multiply each column by random ±1
    if randomized:
        signs = (torch.randint(0, 2, (block_size,), device=dev) * 2 - 1).to(dtype)
        R = R * signs[None, :]  # [block_size, block_size] * [block_size]

    return R


def hadamard_rotation(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: TransformContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, TransformContext]:
    """
    Apply Hadamard rotation to Q and K before quantization.

    Rotation redistributes magnitudes evenly across head_dim, improving
    low-bit quantization accuracy by reducing outlier channels.

    Algorithm:
    1. Create Hadamard matrix R [D, D]
    2. Rotate Q: q_rot = Q @ R
    3. Rotate K: k_rot = K @ R
    4. K-smoothing: k_rot = k_rot - mean(k_rot, dim=sequence)

    Args:
        q: [B, H, N, D] queries
        k: [B, H, N, D] keys
        v: [B, H, N, D] values (unchanged)
        ctx: TransformContext

    Returns:
        q_rot, k_rot, v, ctx with rotation matrix stored in ctx.R
    """
    B, H, N, D = q.shape

    # Create Hadamard rotation matrix (cached after first call)
    # Use full head_dim for rotation to redistribute all dimensions
    BLOCK_R = D

    R = create_hadamard_matrix(
        block_size=BLOCK_R,
        device=q.device.type,
        dtype=torch.bfloat16,
        randomized=True,
    )
    ctx.R = R

    # Rotate Q and K
    q_float = q.to(torch.float32)
    k_float = k.to(torch.float32)
    R_float = R.to(torch.float32)

    # Apply rotation: [B, H, N, D] @ [D, D] → [B, H, N, D]
    q_rot = torch.matmul(q_float, R_float)
    k_rot = torch.matmul(k_float, R_float)

    # K-smoothing: subtract sequence-dim mean to reduce DC offset
    k_rot = k_rot - k_rot.mean(dim=2, keepdim=True)

    return q_rot, k_rot, v, ctx

