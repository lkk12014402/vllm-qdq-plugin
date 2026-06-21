# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fused MXFP4 weight-dequantization + GEMM Triton kernel.

Performs: output = input @ dequant(packed_weight).T + bias
without materializing the full dequantized weight matrix in global memory.

Weight dequantization is done on-the-fly within the GEMM tile loop, which:
  - Reduces memory bandwidth (no intermediate bf16 weight read/write)
  - Enables better GPU utilization for memory-bound workloads
  - Reduces peak memory usage

This kernel assumes:
  - input: [M, K] in float16/bfloat16
  - packed_weight: [N, K/2] uint8 (two FP4 values per byte, row-major)
  - weight_scale: [N, K/GROUP_SIZE] uint8 (e8m0 format)
  - output: [M, N]
"""

import torch
import triton
import triton.language as tl


E8M0_BIAS = tl.constexpr(127)


@triton.jit
def _fp4_e2m1_to_float(idx):
    """Convert 3-bit E2M1 magnitude index to float value."""
    exp_bits = (idx >> 1) & 0x3
    man_bit = (idx & 0x1).to(tl.float32)
    is_subnorm = exp_bits == 0
    normal_val = (1.0 + man_bit * 0.5) * tl.exp2((exp_bits - 1).to(tl.float32))
    subnorm_val = man_bit * 0.5
    return tl.where(is_subnorm, subnorm_val, normal_val)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_stages=5, num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_stages=5, num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _mxfp4_gemm_kernel(
    # Pointers
    input_ptr,        # [M, K]
    packed_w_ptr,     # [N, K_half] uint8
    scale_ptr,        # [N, K_groups] uint8
    bias_ptr,         # [N] or None
    output_ptr,       # [M, N]
    # Dimensions
    M, N, K,
    K_half,           # K // 2
    K_groups,         # K // QUANT_GROUP_SIZE
    # Strides
    stride_im, stride_ik,
    stride_wn, stride_wk,   # packed weight strides (in uint8 elements)
    stride_sn, stride_sk,   # scale strides
    stride_om, stride_on,
    # Config
    HAS_BIAS: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,  # 32
    FP32_PRECISION: tl.constexpr,    # "ieee" for full precision, "tf32" for speed
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,           # must be multiple of QUANT_GROUP_SIZE
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused MXFP4 dequant + GEMM.

    Computes: output[m, n] = sum_k(input[m, k] * dequant(W[n, k])) + bias[n]
    Weight is stored transposed: packed_w[n, k/2].
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in tiles of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        # --- Load input tile [BLOCK_M, BLOCK_K] ---
        input_offsets = rm[:, None] * stride_im + rk[None, :] * stride_ik
        input_mask = (rm[:, None] < M) & (rk[None, :] < K)
        input_tile = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)

        # --- Dequantize weight tile [BLOCK_N, BLOCK_K] -> transpose to [BLOCK_K, BLOCK_N] ---
        # packed_w is [N, K/2], we need W[rn, rk]
        # Packed column index: rk // 2
        packed_k = rk // 2  # [BLOCK_K]
        is_high = (rk % 2).to(tl.int32)  # [BLOCK_K]

        # Load packed bytes: [BLOCK_N, BLOCK_K]
        pw_offsets = rn[:, None] * stride_wn + packed_k[None, :] * stride_wk
        pw_mask = (rn[:, None] < N) & (packed_k[None, :] < K_half)
        packed_bytes = tl.load(packed_w_ptr + pw_offsets, mask=pw_mask, other=0).to(tl.int32)

        # Extract nibble
        nibble = tl.where(
            is_high[None, :] == 1,
            (packed_bytes >> 4) & 0x0F,
            packed_bytes & 0x0F,
        )

        # Decode FP4 E2M1
        sign_bit = (nibble >> 3) & 0x1
        mag_idx = nibble & 0x07
        mag_float = _fp4_e2m1_to_float(mag_idx)
        fp4_val = tl.where(sign_bit == 1, -mag_float, mag_float)

        # Load scale: [BLOCK_N, BLOCK_K] (broadcast per group)
        scale_k = rk // QUANT_GROUP_SIZE  # [BLOCK_K]
        s_offsets = rn[:, None] * stride_sn + scale_k[None, :] * stride_sk
        s_mask = (rn[:, None] < N) & (scale_k[None, :] < K_groups)
        scale_e8m0 = tl.load(scale_ptr + s_offsets, mask=s_mask, other=127).to(tl.int32)
        scale_float = tl.exp2((scale_e8m0 - E8M0_BIAS).to(tl.float32))

        # Dequantized weight tile: [BLOCK_N, BLOCK_K]
        w_dequant = fp4_val * scale_float

        # GEMM accumulation: input[M,K] @ W[N,K].T = input[M,K] @ W.T[K,N]
        # acc += input_tile @ w_dequant.T
        acc += tl.dot(input_tile.to(tl.float32), tl.trans(w_dequant.to(tl.float32)),
                      input_precision=FP32_PRECISION)

    # --- Add bias ---
    if HAS_BIAS:
        bias_offsets = rn
        bias_mask = rn < N
        bias_val = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0).to(tl.float32)
        acc += bias_val[None, :]

    # --- Store output ---
    out_offsets = rm[:, None] * stride_om + rn[None, :] * stride_on
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


@torch.compiler.disable()
def triton_mxfp4_gemm(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 32,
    fp32_precision: str = "ieee",
) -> torch.Tensor:
    """Fused MXFP4 weight dequantization + GEMM using Triton.

    Computes: output = input @ dequant(packed_weight).T + bias
    without materializing the full dequantized weight matrix.

    Args:
        input: [M, K] activation tensor (float16/bfloat16/float32).
        packed_weight: [N, K/2] uint8 — two FP4 values per byte.
        weight_scale: [N, K/GROUP_SIZE] uint8 — e8m0 shared exponents.
        bias: Optional [N] bias tensor.
        group_size: Elements per quantization group (default: 32).
        fp32_precision: "ieee" for exact float32 dot, "tf32" for TF32 speed.

    Returns:
        Output tensor [M, N] in same dtype as input.
    """
    # Normalize fp32_precision: accept bool for convenience
    if isinstance(fp32_precision, bool):
        fp32_precision = "ieee" if fp32_precision else "tf32"

    assert input.device.type == "cuda", "triton_mxfp4_gemm requires CUDA tensors"
    assert packed_weight.dtype == torch.uint8
    assert weight_scale.dtype == torch.uint8

    # Handle batched input: flatten to 2D, compute, reshape back
    orig_shape = input.shape
    if input.ndim > 2:
        input = input.reshape(-1, input.shape[-1])

    M, K = input.shape
    N, K_half = packed_weight.shape
    K_groups = K // group_size

    assert K_half == K // 2, f"packed_weight K dim mismatch: {K_half} != {K}//2"
    assert weight_scale.shape == (N, K_groups), (
        f"Scale shape mismatch: expected ({N}, {K_groups}), got {weight_scale.shape}"
    )

    # Ensure contiguous
    input = input.contiguous()
    packed_weight = packed_weight.contiguous()
    weight_scale = weight_scale.contiguous()

    # Output in float32 for accumulation precision
    output = torch.empty((M, N), dtype=torch.float32, device=input.device)

    has_bias = bias is not None
    if has_bias:
        bias = bias.contiguous()

    # Grid
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    _mxfp4_gemm_kernel[grid](
        input_ptr=input,
        packed_w_ptr=packed_weight,
        scale_ptr=weight_scale,
        bias_ptr=bias if has_bias else input,  # dummy pointer when no bias
        output_ptr=output,
        M=M, N=N, K=K,
        K_half=K_half,
        K_groups=K_groups,
        stride_im=input.stride(0), stride_ik=input.stride(1),
        stride_wn=packed_weight.stride(0), stride_wk=packed_weight.stride(1),
        stride_sn=weight_scale.stride(0), stride_sk=weight_scale.stride(1),
        stride_om=output.stride(0), stride_on=output.stride(1),
        HAS_BIAS=has_bias,
        QUANT_GROUP_SIZE=group_size,
        FP32_PRECISION=fp32_precision,
    )

    # Cast to input dtype and reshape
    output = output.to(input.dtype)
    if len(orig_shape) > 2:
        output = output.reshape(*orig_shape[:-1], N)

    return output
