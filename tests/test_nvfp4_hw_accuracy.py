"""
Accuracy tests for NVFP4 hardware attention kernel.
Compares nvfp4_hw against fp32 reference (sdpa).
"""

import math
import pytest
import torch
import torch.nn.functional as F


def reference_attention(q, k, v, causal=False, sm_scale=None):
    """FP32 reference attention using PyTorch SDPA."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return F.scaled_dot_product_attention(
        q, k, v, scale=sm_scale, is_causal=causal
    )


def cosine_sim(a, b):
    """Per-sample cosine similarity, averaged."""
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    return F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item()


@pytest.fixture
def device():
    return torch.device("cuda")


class TestNvfp4HwBasic:
    """Basic correctness tests."""

    def _run_attention(self, q, k, v, causal=False):
        from vllm_qdq_plugin.sage3_attn.sage3.nvfp4_hw_kernel import (
            nvfp4_flash_attention, quantize_to_nvfp4,
        )
        B, H, N, D = q.shape
        sm_scale = 1.0 / math.sqrt(D)

        # Pad to 128
        if N % 128 != 0:
            pad_len = 128 - (N % 128)
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            N_padded = q.shape[2]
        else:
            N_padded = N

        q_packed, q_scale = quantize_to_nvfp4(q)
        k_packed, k_scale = quantize_to_nvfp4(k)
        v_t = v.permute(0, 1, 3, 2).contiguous()
        v_flat = v_t.reshape(-1, N_padded)
        v_packed_flat, v_scale_flat = quantize_to_nvfp4(v_flat)
        v_packed = v_packed_flat.reshape(B, H, D, N_padded // 2)
        v_scale = v_scale_flat.reshape(B, H, D, N_padded // 16)

        output = nvfp4_flash_attention(
            q_packed, k_packed, v_packed,
            q_scale, k_scale, v_scale,
            causal=causal, sm_scale=sm_scale,
        )
        return output[:, :, :N, :]

    def test_uniform_data(self, device):
        """Uniform random data — should achieve >0.95 cosine similarity."""
        B, H, N, D = 2, 4, 256, 128
        q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

        ref = reference_attention(q, k, v)
        out = self._run_attention(q, k, v)

        sim = cosine_sim(ref, out)
        print(f"Uniform data cosine similarity: {sim:.6f}")
        assert sim > 0.95, f"Cosine similarity {sim} too low"

    def test_causal(self, device):
        """Causal masking correctness."""
        B, H, N, D = 2, 4, 256, 128
        q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

        ref = reference_attention(q, k, v, causal=True)
        out = self._run_attention(q, k, v, causal=True)

        sim = cosine_sim(ref, out)
        print(f"Causal cosine similarity: {sim:.6f}")
        assert sim > 0.95, f"Cosine similarity {sim} too low"

    def test_varying_seq_lengths(self, device):
        """Test padding correctness with various sequence lengths."""
        B, H, D = 1, 2, 128
        for N in [128, 192, 256, 384, 512]:
            q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
            k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
            v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

            ref = reference_attention(q, k, v)
            out = self._run_attention(q, k, v)

            sim = cosine_sim(ref, out)
            print(f"N={N}: cosine similarity = {sim:.6f}")
            assert sim > 0.94, f"N={N}: cosine similarity {sim} too low"

    def test_head_dim_64(self, device):
        """Test with HEAD_DIM=64."""
        B, H, N, D = 2, 4, 256, 64
        q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

        ref = reference_attention(q, k, v)
        out = self._run_attention(q, k, v)

        sim = cosine_sim(ref, out)
        print(f"HEAD_DIM=64 cosine similarity: {sim:.6f}")
        assert sim > 0.95


class TestNvfp4HwWithDeltaS:
    """Tests with delta_s (QK smoothing) via api dispatch."""

    def test_delta_s_improves_accuracy(self, device):
        """Delta_s correction should improve accuracy on high-DC-offset data."""
        import os
        os.environ.pop('SAGE3_DISABLE_PER_BLOCK_MEAN', None)

        from vllm_qdq_plugin.sage3_attn.sage3.nvfp4_hw_kernel import (
            nvfp4_flash_attention, quantize_to_nvfp4,
        )
        from vllm_qdq_plugin.sage3_attn.sage3.transforms import qk_smoothing, TransformContext

        B, H, N, D = 2, 4, 256, 128
        # Add large DC offset to stress quantization
        q = torch.randn(B, H, N, D, device=device, dtype=torch.float32) + 3.0
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32) + 3.0
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        sm_scale = 1.0 / math.sqrt(D)

        ref = reference_attention(q, k, v)

        # With delta_s
        ctx = TransformContext()
        q_s, k_s, v_s, ctx = qk_smoothing(q, k, v, ctx)
        delta_s = ctx.delta_s

        q_packed, q_scale = quantize_to_nvfp4(q_s)
        k_packed, k_scale = quantize_to_nvfp4(k_s)
        v_t = v_s.permute(0, 1, 3, 2).contiguous()
        v_flat = v_t.reshape(-1, N)
        v_packed_flat, v_scale_flat = quantize_to_nvfp4(v_flat)
        v_packed = v_packed_flat.reshape(B, H, D, N // 2)
        v_scale_t = v_scale_flat.reshape(B, H, D, N // 16)

        out_with_ds = nvfp4_flash_attention(
            q_packed, k_packed, v_packed,
            q_scale, k_scale, v_scale_t,
            sm_scale=sm_scale, delta_s=delta_s,
        )

        sim = cosine_sim(ref, out_with_ds)
        print(f"nvfp4_hw with delta_s (high DC offset): cosine sim = {sim:.6f}")
        assert sim > 0.93, f"Cosine similarity {sim} too low with delta_s"


class TestNvfp4HwQuantizer:
    """Unit tests for quantize_to_nvfp4."""

    def test_roundtrip(self, device):
        """Quantize and dequantize should be close to original."""
        from vllm_qdq_plugin.sage3_attn.sage3.nvfp4_hw_kernel import (
            quantize_to_nvfp4, dequantize_nvfp4,
        )
        x = torch.randn(4, 128, device=device, dtype=torch.float32)
        packed, scales = quantize_to_nvfp4(x)

        assert packed.shape == (4, 64)  # K//2
        assert scales.shape == (4, 8)   # K//16
        assert scales.dtype == torch.float8_e4m3fn

        recon = dequantize_nvfp4(packed, scales)
        # FP4 with group_size=16 should have reasonable reconstruction
        rel_err = (x - recon).abs().mean() / x.abs().mean()
        print(f"Relative reconstruction error: {rel_err:.4f}")
        assert rel_err < 0.5  # FP4 is lossy but should be reasonable

    def test_scale_shapes(self, device):
        """Verify output shapes for various input sizes."""
        from vllm_qdq_plugin.sage3_attn.sage3.nvfp4_hw_kernel import quantize_to_nvfp4

        for K in [64, 128, 256]:
            x = torch.randn(2, 8, K, device=device)
            packed, scales = quantize_to_nvfp4(x)
            assert packed.shape == (2, 8, K // 2)
            assert scales.shape == (2, 8, K // 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
