"""
NVFP4 (E2M1) Flash Attention on NVIDIA B200 (sm_100)
=====================================================

Hardware attention using NVFP4 format:
- Data format: E2M1 (4-bit FP, packed 2 per byte) for Q, K, V, and P
- Scale format: E4M3 (fp8_e4m3fn, finer granularity than E8M0)
- Scale grouping: group_size=16 along reduction dimension
- Hardware: NVIDIA Blackwell (sm_100) via tl.dot_scaled -> tcgen05.mma

Compared to mxfp4_hw (E8M0, group_size=32), nvfp4 provides:
- 2x finer scale granularity (16 vs 32 elements per scale)
- More expressive scales (E4M3 has mantissa bits, not just power-of-2)
- Better accuracy for the same data format
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Helper Functions
# ============================================================================

# E2M1 representable positive values (magnitude only):
# 0b000=0, 0b001=0.5, 0b010=1.0, 0b011=1.5, 0b100=2.0, 0b101=3.0, 0b110=4.0, 0b111=6.0
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _float_to_e2m1_nibble(x: torch.Tensor) -> torch.Tensor:
    """Convert float tensor to E2M1 4-bit encoding (uint8, values 0-15)."""
    sign = (x < 0).to(torch.uint8) << 3
    ax = x.abs().clamp(max=6.0)

    code = torch.zeros_like(ax, dtype=torch.uint8)
    code = torch.where(ax >= 0.25, 1, code)
    code = torch.where(ax >= 0.75, 2, code)
    code = torch.where(ax >= 1.25, 3, code)
    code = torch.where(ax >= 1.75, 4, code)
    code = torch.where(ax >= 2.5, 5, code)
    code = torch.where(ax >= 3.5, 6, code)
    code = torch.where(ax >= 5.0, 7, code)

    return sign | code


def _e2m1_nibble_to_float(nibble: torch.Tensor) -> torch.Tensor:
    """Convert E2M1 4-bit encoding back to float32."""
    sign = ((nibble >> 3) & 1).float() * -2.0 + 1.0
    mag_code = (nibble & 0x7).long()
    values = _E2M1_VALUES.to(nibble.device)
    mag = values[mag_code]
    return sign * mag


def quantize_to_nvfp4(x: torch.Tensor, group_size: int = 16) -> tuple:
    """
    Quantize float tensor to NVFP4 (E2M1) with E4M3 block scales.
    Groups along the last dimension with group_size=16.

    Returns:
        x_packed: [..., K//2] uint8 (two e2m1 values per byte)
        scales: [..., K // group_size] float8_e4m3fn
    """
    assert x.shape[-1] % group_size == 0
    assert x.shape[-1] % 2 == 0

    original_shape = x.shape
    x_f32 = x.float()

    *batch_dims, K = x_f32.shape
    num_groups = K // group_size
    x_grouped = x_f32.reshape(*batch_dims, num_groups, group_size)

    # Compute per-group amax
    amax = x_grouped.abs().amax(dim=-1)  # [..., num_groups]
    e2m1_max = 6.0

    # E4M3 scale: amax / 6.0, cast to fp8_e4m3fn
    # E4M3 max is 448.0, min positive normal is 2^-6=0.015625
    scale_f32 = amax / e2m1_max
    scale_f32 = scale_f32.clamp(min=torch.finfo(torch.float8_e4m3fn).tiny)
    # Cast to E4M3 and back to get the actual quantized scale
    scale_e4m3 = scale_f32.to(torch.float8_e4m3fn)
    scale_f32_actual = scale_e4m3.to(torch.float32)

    # Divide by scale
    scale_expanded = scale_f32_actual.unsqueeze(-1).expand_as(x_grouped)
    x_scaled = x_grouped / scale_expanded  # values in [-6, 6]
    x_scaled = x_scaled.reshape(original_shape)

    # Encode to E2M1 nibbles
    nibbles = _float_to_e2m1_nibble(x_scaled)

    # Pack pairs: low nibble = even index, high nibble = odd index
    even = nibbles[..., 0::2]
    odd = nibbles[..., 1::2]
    packed = (odd << 4) | even  # [..., K//2] uint8

    return packed, scale_e4m3


def dequantize_nvfp4(x_packed: torch.Tensor, scales: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Dequantize NVFP4 packed tensor back to float32."""
    even = x_packed & 0x0F
    odd = (x_packed >> 4) & 0x0F

    *batch_dims, K_half = even.shape
    K = K_half * 2
    full = torch.zeros(*batch_dims, K, dtype=torch.uint8, device=x_packed.device)
    full[..., 0::2] = even
    full[..., 1::2] = odd

    x_f32 = _e2m1_nibble_to_float(full)

    scale_f32 = scales.to(torch.float32)
    scale_expanded = scale_f32.repeat_interleave(group_size, dim=-1)
    scale_expanded = scale_expanded[..., :K]
    return x_f32 * scale_expanded


# ============================================================================
# Triton Kernel
# ============================================================================


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    """Convert two f32 values to packed E2M1x2 byte via PTX hardware instruction."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def compute_p_scale_e4m3(
    p_amax,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Compute E4M3 scale and inverse for P quantization with group_size=16.

    For nvfp4, scales are fp8_e4m3 (not E8M0 power-of-2).
    We compute scale = amax (clamped to E4M3 range), inv_scale = 1/scale.

    Since P is always non-negative (post-softmax), scale = max value in group.
    We divide by amax directly (the E4M3 rounding happens in hardware).

    Args:
        p_amax: [BLOCK_M, BLOCK_N // GROUP_SIZE] per-group max of P
        BLOCK_M: tile height
        BLOCK_N: tile width
        GROUP_SIZE: 16 for nvfp4

    Returns:
        p_scale_e4m3: [BLOCK_M, BLOCK_N // GROUP_SIZE] fp8_e4m3 scales
        inv_scale_expanded: [BLOCK_M, BLOCK_N] float32 inverse scales
    """
    FP4_E2M1_MAX: tl.constexpr = 6.0
    # Scale = amax / 6.0 (so that max value maps to 6.0 in E2M1)
    # Clamp to avoid division by zero
    p_scale = p_amax / FP4_E2M1_MAX
    p_scale = tl.maximum(p_scale, 0.001953125)  # E4M3 min subnormal = 2^-9

    # Cast to E4M3 (the hardware will use this scale directly)
    p_scale_e4m3 = p_scale.to(tl.float8e4nv)

    # Compute inverse from unrounded amax for better numerical precision in quantization
    p_scale_f32 = p_scale_e4m3.to(tl.float32)
    inv_scale = FP4_E2M1_MAX / tl.maximum(p_amax, 0.001953125 * FP4_E2M1_MAX)

    NUM_GROUPS: tl.constexpr = BLOCK_N // GROUP_SIZE
    inv_scale_expanded = tl.reshape(
        tl.broadcast_to(inv_scale[:, :, None], [BLOCK_M, NUM_GROUPS, GROUP_SIZE]),
        [BLOCK_M, BLOCK_N],
    )

    return p_scale_e4m3, inv_scale_expanded


@triton.jit
def _nvfp4_attn_fwd_inner(
    acc, l_i, m_i, q, q_scale,
    K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,
    Delta_s_ptr,
    stride_kn, stride_kk,
    stride_ks_n, stride_ks_k,
    stride_vd, stride_vn,
    stride_vs_d, stride_vs_n,
    stride_delta_g, stride_delta_n,
    start_m, qk_scale,
    offs_m, offs_n,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    HAS_DELTA_S: tl.constexpr,
):
    # Determine loop bounds based on causal stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:  # STAGE == 3: non-causal, full range
        lo, hi = 0, N_CTX

    HEAD_DIM_PACKED: tl.constexpr = HEAD_DIM // 2
    BLOCK_N_PACKED: tl.constexpr = BLOCK_N // 2
    # nvfp4: group_size=16
    SCALE_GROUPS_K: tl.constexpr = HEAD_DIM // 16
    SCALE_GROUPS_N: tl.constexpr = BLOCK_N // 16

    offs_k_head_packed = tl.arange(0, HEAD_DIM_PACKED)
    offs_k_n = tl.arange(0, BLOCK_N)
    offs_scale_k = tl.arange(0, SCALE_GROUPS_K)
    offs_scale_n = tl.arange(0, SCALE_GROUPS_N)
    offs_head_full = tl.arange(0, HEAD_DIM)
    offs_n_packed = tl.arange(0, BLOCK_N_PACKED)

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- Load K block [BLOCK_N, HEAD_DIM//2] packed uint8 --
        k_ptrs = K_ptr + (start_n + offs_k_n[:, None]) * stride_kn + offs_k_head_packed[None, :] * stride_kk
        k = tl.load(k_ptrs)

        # -- Load K scale [BLOCK_N, HEAD_DIM // 16] fp8_e4m3 --
        ks_ptrs = K_scale_ptr + (start_n + offs_k_n[:, None]) * stride_ks_n + offs_scale_k[None, :] * stride_ks_k
        k_scale = tl.load(ks_ptrs)

        # -- Q @ K^T via dot_scaled e2m1 --
        qk = tl.dot_scaled(q, q_scale, "e2m1", tl.trans(k), k_scale, "e2m1")

        # -- Add delta_s correction --
        if HAS_DELTA_S:
            group_id = start_m
            ds_ptrs = Delta_s_ptr + group_id * stride_delta_g + (start_n + offs_n) * stride_delta_n
            ds_tile = tl.load(ds_ptrs)
            qk = qk + ds_tile[None, :]

        # -- Online softmax --
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)

        acc = acc * alpha[:, None]

        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij

        # -- Quantize P to NVFP4 (E2M1) with E4M3 scales, group_size=16 --
        p_reshaped = tl.reshape(p, [BLOCK_M, BLOCK_N // 16, 16])
        p_amax = tl.max(p_reshaped, 2)  # [BLOCK_M, BLOCK_N // 16]

        p_scale_e4m3, inv_scale_expanded = compute_p_scale_e4m3(
            p_amax, BLOCK_M, BLOCK_N, 16,
        )

        # Scale P values
        p_scaled = p * inv_scale_expanded

        # Pack to FP4
        p_pairs = tl.reshape(p_scaled, [BLOCK_M, BLOCK_N_PACKED, 2])
        p_even, p_odd = tl.split(p_pairs)

        p_packed = _fp32x2_to_fp4x2(p_even, p_odd)  # [BLOCK_M, BLOCK_N//2] uint8

        # -- Load V block [HEAD_DIM, BLOCK_N//2] packed along N (col-major) --
        v_ptrs = V_ptr + offs_head_full[:, None] * stride_vd + (start_n // 2 + offs_n_packed[None, :]) * stride_vn
        v = tl.load(v_ptrs)

        # -- Load V scale [HEAD_DIM, BLOCK_N // 16] fp8_e4m3 --
        vs_ptrs = V_scale_ptr + offs_head_full[:, None] * stride_vs_d + (start_n // 16 + offs_scale_n[None, :]) * stride_vs_n
        v_scale = tl.load(vs_ptrs)

        # -- P @ V via dot_scaled e2m1 --
        acc = tl.dot_scaled(p_packed, p_scale_e4m3, "e2m1", tl.trans(v), v_scale, "e2m1", acc)

        m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def _nvfp4_attn_fwd(
    Q, K, V, Out,
    Q_scale, K_scale, V_scale,
    Delta_s,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vd, stride_vn,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_qsz, stride_qsh, stride_qsm, stride_qsk,
    stride_ksz, stride_ksh, stride_ksn, stride_ksk,
    stride_vsz, stride_vsh, stride_vsd, stride_vsn,
    stride_dsz, stride_dsh, stride_dsg, stride_dsn,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    HAS_DELTA_S: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Base pointers for this batch/head
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    qs_offset = off_z * stride_qsz + off_h * stride_qsh
    ks_offset = off_z * stride_ksz + off_h * stride_ksh
    vs_offset = off_z * stride_vsz + off_h * stride_vsh

    Q_ptr = Q + q_offset
    K_ptr = K + k_offset
    V_ptr = V + v_offset
    Q_scale_ptr = Q_scale + qs_offset
    K_scale_ptr = K_scale + ks_offset
    V_scale_ptr = V_scale + vs_offset

    Delta_s_ptr = Delta_s + off_z * stride_dsz + off_h * stride_dsh

    HEAD_DIM_PACKED: tl.constexpr = HEAD_DIM // 2
    SCALE_GROUPS_K: tl.constexpr = HEAD_DIM // 16

    # Load Q block [BLOCK_M, HEAD_DIM//2] packed
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_head_packed = tl.arange(0, HEAD_DIM_PACKED)
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_head_packed[None, :] * stride_qk
    q = tl.load(q_ptrs)

    # Load Q scale [BLOCK_M, HEAD_DIM // 16] fp8_e4m3
    offs_scale_k = tl.arange(0, SCALE_GROUPS_K)
    qs_ptrs = Q_scale_ptr + offs_m[:, None] * stride_qsm + offs_scale_k[None, :] * stride_qsk
    q_scale = tl.load(qs_ptrs)

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Scale with log2(e) for exp2-based softmax
    qk_scale = sm_scale * 1.44269504

    offs_n = tl.arange(0, BLOCK_N)

    # Run attention inner loop
    if STAGE & 1:
        acc, l_i, m_i = _nvfp4_attn_fwd_inner(
            acc, l_i, m_i, q, q_scale,
            K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,
            Delta_s_ptr,
            stride_kn, stride_kk,
            stride_ksn, stride_ksk,
            stride_vd, stride_vn,
            stride_vsd, stride_vsn,
            stride_dsg, stride_dsn,
            start_m, qk_scale,
            offs_m, offs_n,
            N_CTX, BLOCK_M, HEAD_DIM, BLOCK_N,
            4 - STAGE,
            HAS_DELTA_S,
        )
    if STAGE & 2:
        acc, l_i, m_i = _nvfp4_attn_fwd_inner(
            acc, l_i, m_i, q, q_scale,
            K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,
            Delta_s_ptr,
            stride_kn, stride_kk,
            stride_ksn, stride_ksk,
            stride_vd, stride_vn,
            stride_vsd, stride_vsn,
            stride_dsg, stride_dsn,
            start_m, qk_scale,
            offs_m, offs_n,
            N_CTX, BLOCK_M, HEAD_DIM, BLOCK_N,
            2,
            HAS_DELTA_S,
        )

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output
    offs_head = tl.arange(0, HEAD_DIM)
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_head[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty))


# ============================================================================
# Wrapper Function
# ============================================================================


def nvfp4_flash_attention(
    q_packed: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    delta_s: torch.Tensor = None,
) -> torch.Tensor:
    """
    NVFP4 (E2M1) Flash Attention forward pass with E4M3 block scales (group_size=16).

    Args:
        q_packed: [B, H, M, D//2] uint8 — packed E2M1 queries
        k_packed: [B, H, N, D//2] uint8 — packed E2M1 keys
        v_packed: [B, H, D, N//2] uint8 — packed E2M1 values (col-major)
        q_scale: [B, H, M, D//16] float8_e4m3fn — scales for Q
        k_scale: [B, H, N, D//16] float8_e4m3fn — scales for K
        v_scale: [B, H, D, N//16] float8_e4m3fn — scales for V
        causal: whether to apply causal mask
        sm_scale: softmax scale (default: 1/sqrt(HEAD_DIM))
        delta_s: [B, H, num_groups, N] float32 — QK smoothing correction (optional)

    Returns:
        output: [B, H, M, D] float32
    """
    B, H, M, D_packed = q_packed.shape
    D = D_packed * 2
    N = k_packed.shape[2]

    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 64

    assert D in (64, 128), f"HEAD_DIM must be 64 or 128, got {D}"
    assert M % BLOCK_M == 0, f"M={M} must be divisible by BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be divisible by BLOCK_N={BLOCK_N}"

    output = torch.empty((B, H, M, D), dtype=torch.float32, device=q_packed.device)

    STAGE = 3 if causal else 1

    has_delta_s = delta_s is not None
    if not has_delta_s:
        delta_s = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=q_packed.device)

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    _nvfp4_attn_fwd[grid](
        q_packed, k_packed, v_packed, output,
        q_scale, k_scale, v_scale,
        delta_s,
        sm_scale,
        # Q strides [B, H, M, D//2]
        q_packed.stride(0), q_packed.stride(1), q_packed.stride(2), q_packed.stride(3),
        # K strides [B, H, N, D//2]
        k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
        # V strides [B, H, D, N//2]
        v_packed.stride(0), v_packed.stride(1), v_packed.stride(2), v_packed.stride(3),
        # O strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Q_scale strides [B, H, M, D//16]
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2), q_scale.stride(3),
        # K_scale strides [B, H, N, D//16]
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), k_scale.stride(3),
        # V_scale strides [B, H, D, N//16]
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2), v_scale.stride(3),
        # Delta_s strides [B, H, num_groups, N]
        delta_s.stride(0), delta_s.stride(1), delta_s.stride(2), delta_s.stride(3),
        # Dimensions
        B, H, N,
        # Compile-time constants
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=STAGE,
        HAS_DELTA_S=has_delta_s,
        num_warps=4,
        num_stages=4,
    )

    return output
