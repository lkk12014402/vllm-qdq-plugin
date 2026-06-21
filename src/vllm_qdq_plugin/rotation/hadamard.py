# SPDX-License-Identifier: Apache-2.0
"""Hadamard / orthogonal rotation utilities (vendored, standalone).

Ported from auto-round
(``auto_round.algorithms.transforms.spinquant.rotation_utils`` and the helper
functions in ``auto_round.vllm_plugin.spinquant_mxfp4``) so that this plugin has
**zero auto_round dependency**.

These utilities reconstruct, at model load time, the same Hadamard / random /
trained rotation matrices that auto-round fused or registered during
quantization, so the online R1/R4 activation rotation can be replayed exactly.

Non-power-of-2 dimensions (e.g. 3072 = 12 x 256) are supported via the
pre-computed ``known_hadamard.KNOWN_HADAMARD_MATRICES`` table.
"""

from __future__ import annotations

import math

import torch

from .known_hadamard import KNOWN_HADAMARD_MATRICES

__all__ = [
    "is_pow2",
    "get_hadamard_K",
    "matmul_hadU",
    "deterministic_hadamard_matrix",
    "random_hadamard_matrix",
    "build_full_hadamard",
    "build_block_hadamard",
    "generate_random_orthogonal",
    "describe_hadamard",
]


def is_pow2(n: int) -> bool:
    """Check if ``n`` is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def get_hadamard_K(n: int) -> tuple[torch.Tensor, int]:
    """Return the Hadamard matrix ``H_K`` and block dimension ``K`` for size ``n``.

    For power-of-2 sizes, ``K=1`` (full Walsh-Hadamard via butterfly).
    For non-power-of-2 sizes, ``K`` is the largest known Hadamard size that
    divides ``n`` such that ``n / K`` is a power of 2.

    Examples:
        - ``n=3072``: ``3072 = 12 x 256`` -> ``K=12``, returns a 12x12 Hadamard.
        - ``n=1024``: power-of-2          -> ``K=1``,  returns a 1024x1024 Hadamard.
    """
    if is_pow2(n):
        # Sylvester construction (unnormalized).
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        return H, 1

    for size in KNOWN_HADAMARD_MATRICES:
        if n % size == 0 and is_pow2(n // size):
            had_K = KNOWN_HADAMARD_MATRICES[size]()
            return had_K, size

    raise ValueError(
        f"Cannot find suitable Hadamard decomposition for n={n}. "
        f"Known non-pow2 sizes: {sorted(KNOWN_HADAMARD_MATRICES.keys())}. "
        f"n must be a power of 2, or n = K x 2^m for a known K."
    )


def matmul_hadU(
    X: torch.Tensor,
    hadamard_K: torch.Tensor | None = None,
    K: int | None = None,
) -> torch.Tensor:
    """Apply the normalized Hadamard transform to the last dimension of ``X``.

    Equivalent to ``X @ H`` (``H`` normalized) but computed via the efficient
    recursive butterfly algorithm, with an explicit ``K x K`` Hadamard block for
    any non-power-of-2 residual. Based on QuaRot/Quark's implementation.
    """
    n = X.shape[-1]

    if hadamard_K is None or K is None:
        hadamard_K, K = get_hadamard_K(n)
        hadamard_K = hadamard_K.to(dtype=X.dtype, device=X.device)

    inp = X.clone().reshape(-1, n, 1)
    output = inp.clone()

    # Butterfly decomposition (Walsh-Hadamard for the n/K part).
    while inp.shape[1] > K:
        inp = inp.view(inp.shape[0], inp.shape[1] // 2, 2, inp.shape[2])
        output = output.view(inp.shape)
        output[:, :, 0, :] = inp[:, :, 0, :] + inp[:, :, 1, :]
        output[:, :, 1, :] = inp[:, :, 0, :] - inp[:, :, 1, :]
        output = output.view(inp.shape[0], inp.shape[1], -1)
        inp, output = (output, inp)
    del output

    # Apply the K x K Hadamard block (if K > 1).
    if K > 1:
        had = hadamard_K.to(inp.device).to(inp.dtype)
        inp = had.view(1, K, K) @ inp

    # Normalize: butterfly + K-block gives an unnormalized result, divide by sqrt(n).
    return inp.view(X.shape) / math.sqrt(n)


def deterministic_hadamard_matrix(
    size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a normalized Sylvester Hadamard matrix (``H / sqrt(N)``)."""
    if size <= 0 or size & (size - 1) != 0:
        raise ValueError(f"deterministic_hadamard_matrix requires power-of-2 size, got {size}")
    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    while H.size(0) < size:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(size)


def random_hadamard_matrix(
    size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a random Hadamard matrix: ``H @ diag(+-1) / sqrt(N)``."""
    D = torch.randint(0, 2, (size,), dtype=torch.float64, device=device) * 2 - 1
    Q = torch.diag(D)
    hadamard_K, K = get_hadamard_K(size)
    hadamard_K = hadamard_K.to(dtype=torch.float64, device=device)
    result = matmul_hadU(Q, hadamard_K=hadamard_K, K=K)
    return result.to(dtype=dtype)


def build_full_hadamard(n: int, device: torch.device) -> torch.Tensor:
    """Build the full ``[n, n]`` normalized Hadamard matrix on ``device``.

    Called once at model load time (NOT in the forward path). Applies
    :func:`matmul_hadU` to the identity so the explicit matrix is byte-identical
    to the butterfly algorithm used during save-time fuse / online hooks. This is
    critical for non-power-of-2 dimensions (e.g. 3072 with K=12), where the
    butterfly interleaving pattern differs from a naive first/second-half split.
    """
    eye = torch.eye(n, device=device, dtype=torch.float32)
    return matmul_hadU(eye)


def build_block_hadamard(
    rotation_size: int,
    hadamard_K: torch.Tensor,
    K: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a normalized explicit Hadamard block for block-wise rotation."""
    rot_mat = hadamard_K.to(device=device, dtype=torch.float32)
    if rot_mat.shape[0] != rotation_size:
        had_1, _ = get_hadamard_K(rotation_size // K)
        rot_mat = torch.kron(
            rot_mat.to(device="cpu", dtype=torch.float32),
            had_1.to(device="cpu", dtype=torch.float32),
        ).to(device=device)
    return rot_mat / math.sqrt(rotation_size)


def generate_random_orthogonal(n: int, device: torch.device) -> torch.Tensor:
    """Generate a deterministic random orthogonal matrix ``[n, n]`` via QR.

    Used as a fallback when ``rot_type`` is random/trained but no matrix is in
    the checkpoint. Deterministic per ``n`` (fixed seed) so all TP ranks agree.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42 + n)  # deterministic across ranks
    rand_mat = torch.randn(n, n, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(rand_mat)
    return Q.to(device=device)


def describe_hadamard(rot_type: str, size: int) -> str:
    """Build a human-readable description of a rotation configuration."""
    if rot_type != "hadamard":
        return f"{rot_type}, size={size}"
    if is_pow2(size):
        return f"hadamard, size={size}, power-of-2"
    try:
        _, K = get_hadamard_K(size)
        return f"hadamard, size={size}, K={K} (non-power-of-2)"
    except (ValueError, ImportError):
        return f"hadamard, size={size}, non-power-of-2"
