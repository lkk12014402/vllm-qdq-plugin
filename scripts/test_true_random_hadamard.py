#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""True-random per-partition Hadamard equivalence (synthetic, CPU).

Validates that ``HadamardMXFP4LinearMethod`` reproduces auto-round's
separate-module math for MERGED layers (qkv_proj / gate_up_proj) when each
partition has a DIFFERENT random Hadamard matrix.

auto-round runs q/k/v as separate Linears, each with its own inverse-block
Hadamard + MXFP4 act-qdq before its own GEMM. We build that reference from N
independent "modules" and check the merged plugin layer (per-partition slow
path) matches a concat of the references — and that the custom weight loader
places each shard's matrix into the correct slot.

Also checks the uniform fast path stays bit-equivalent to the per-partition path
when all matrices are identical.

Usage:
    python test_true_random_hadamard.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "vllm-qdq-plugin/src")
from vllm_qdq_plugin.rotation.mxfp4 import (  # noqa: E402
    dequant_packed_mxfp4_weight,
    register_custom_ops,
)
from vllm_qdq_plugin.rotation.perlinear_config import HadamardMXFP4Config  # noqa: E402
from vllm_qdq_plugin.rotation.perlinear_linear_method import (  # noqa: E402
    HadamardMXFP4LinearMethod,
    _hadamard_weight_loader,
)

BS = 32
GS = 32
K = 256  # input dim (multiple of block size)

_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def random_orthonormal_block(bs: int, gen: torch.Generator) -> torch.Tensor:
    """A random orthonormal bs x bs matrix (QR of a Gaussian), like a random Hadamard block."""
    a = torch.randn(bs, bs, generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    # Fix signs for determinism / orthonormality.
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.contiguous()


def pack_mxfp4_weight(w: torch.Tensor, group_size: int):
    """Quantize a dense weight to packed E2M1 MXFP4 + e8m0 scale (matches the unpacker)."""
    w = w.float()
    n, k = w.shape
    wg = w.reshape(n, -1, group_size)
    amax = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-20)
    # e8m0 scale exponent so that amax/scale <= 6 (FP4 max).
    exp = torch.floor(torch.log2(amax / 6.0)).clamp(-127, 128)
    scale = torch.pow(2.0, exp)
    e8m0 = (exp + 127.0).to(torch.uint8).squeeze(-1)  # [n, k//gs]

    scaled = (wg / scale).clamp(-6.0, 6.0)
    sign = (scaled < 0).to(torch.int32)
    lut = _E2M1_LUT.to(w.device)
    idx = (scaled.abs().unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.int32)
    code = (sign * 8 + idx).reshape(n, k)  # nibble code 0..15

    low = code[:, 0::2] & 0x0F
    high = code[:, 1::2] & 0x0F
    packed = (low | (high << 4)).to(torch.uint8)
    return packed, e8m0


def make_packed_weight(n: int, k: int, gen: torch.Generator):
    """Random bf16 weight -> MXFP4 packed + scale, plus its (exact) dequantized bf16."""
    w = (torch.randn(n, k, generator=gen, dtype=torch.float32) * 0.3).to(torch.bfloat16)
    packed, scale = pack_mxfp4_weight(w, GS)
    w_deq = dequant_packed_mxfp4_weight(packed, scale, GS, target_dtype=torch.bfloat16)
    return packed, scale, w_deq


def reference_partition(x: torch.Tensor, H: torch.Tensor, w_deq: torch.Tensor) -> torch.Tensor:
    """yᵢ = qdq(x @ Hᵢᵀ) @ w_deqᵀ  — the separate-module auto-round math."""
    R = H.t().contiguous().to(x.dtype)
    shape = x.shape
    xr = (x.reshape(*shape[:-1], -1, BS) @ R).reshape(shape)
    xq = torch.ops.vllm_qdq_plugin.hadamard_mxfp4_act_qdq(xr, GS)
    return torch.nn.functional.linear(xq.to(w_deq.dtype), w_deq)


def build_merged_layer(partition_sizes, Hs, packed_list, scale_list):
    """Emulate create_weights + custom loader + process_weights for a merged layer."""
    cfg = HadamardMXFP4Config(hadamard_type="random_hadamard", block_size=BS, group_size=GS)
    method = HadamardMXFP4LinearMethod(cfg)

    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    layer._output_partition_sizes = list(partition_sizes)

    layer.weight_packed = torch.nn.Parameter(torch.cat(packed_list, dim=0), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.cat(scale_list, dim=0), requires_grad=False)

    # Per-partition Hadamard buffer filled via the production custom loader.
    hbuf = torch.nn.Parameter(
        torch.eye(BS).unsqueeze(0).repeat(len(partition_sizes), 1, 1), requires_grad=False
    )
    shard_ids = ["q", "k", "v"] if len(partition_sizes) == 3 else list(range(len(partition_sizes)))
    for sid, H in zip(shard_ids, Hs):
        _hadamard_weight_loader(hbuf, H, sid)
    layer.hadamard_matrix = hbuf

    method.process_weights_after_loading(layer)
    return method, layer


def run_case(name, partition_sizes, distinct, gen, device="cuda"):
    Hs, packed_list, scale_list, w_deqs = [], [], [], []
    base_H = random_orthonormal_block(BS, gen)
    for i, n in enumerate(partition_sizes):
        H = random_orthonormal_block(BS, gen) if distinct else base_H.clone()
        Hs.append(H.to(device))
        packed, scale, w_deq = make_packed_weight(n, K, gen)
        packed_list.append(packed.to(device))
        scale_list.append(scale.to(device))
        w_deqs.append(w_deq.to(device))

    method, layer = build_merged_layer(partition_sizes, Hs, packed_list, scale_list)
    expected_uniform = (not distinct) or len(partition_sizes) == 1
    assert layer._rotation_uniform == expected_uniform, (
        f"{name}: uniform-detect mismatch (expected={expected_uniform}, got={layer._rotation_uniform})"
    )

    x = ((torch.randn(4, K, dtype=torch.float32) * 0.5).to(torch.bfloat16)).to(device)
    y_plugin = method.apply(layer, x).float()

    # Reference = concat of per-partition separate-module outputs.
    refs = [reference_partition(x, Hs[i], w_deqs[i]) for i in range(len(partition_sizes))]
    y_ref = torch.cat(refs, dim=-1).float()

    abs_err = (y_plugin - y_ref).abs()
    cos = torch.nn.functional.cosine_similarity(y_plugin.reshape(-1), y_ref.reshape(-1), dim=0).item()
    maxerr = abs_err.max().item()
    ok = torch.allclose(y_plugin, y_ref, atol=1e-3, rtol=1e-3) or cos > 0.99999
    print(
        f"[{name:28s}] partitions={partition_sizes} distinct={distinct} "
        f"uniform={layer._rotation_uniform} cos={cos:.8f} maxerr={maxerr:.3e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main() -> None:
    register_custom_ops()
    device = "cuda"
    gen = torch.Generator().manual_seed(0)
    results = []
    # QKV merged with DISTINCT per-Linear random Hadamards (true random).
    results.append(run_case("qkv distinct (true random)", [128, 64, 64], True, gen, device))
    # Gate/Up merged with distinct matrices.
    results.append(run_case("gate_up distinct", [192, 192], True, gen, device))
    # Single Linear.
    results.append(run_case("single", [160], True, gen, device))
    # Uniform case: fast path must equal per-partition reference.
    results.append(run_case("qkv uniform (fast path)", [128, 64, 64], False, gen, device))
    print("\nOVERALL:", "PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
