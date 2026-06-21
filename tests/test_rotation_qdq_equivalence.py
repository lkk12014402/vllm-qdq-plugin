# SPDX-License-Identifier: Apache-2.0
"""Equivalence test: rotation MXFP4 activation QDQ == native QDQ-sim MXFP4 QDQ.

The rotation subpackage vendors its own pure-PyTorch MXFP4 activation QDQ
(``rotation.mxfp4.mxfp4_act_qdq``, Quark "even" mode) to stay fully standalone.
This test verifies it is **bit-exact identical** to the native QDQ-simulation
path (``qdq.mxfp4.mxfp4_qdq``, vLLM's reference MXFP4) for fp16/bf16 inputs,
so the duplication is intentional and safe.
"""

import unittest

import torch

from vllm_qdq_plugin.qdq.mxfp4 import mxfp4_qdq as native_qdq
from vllm_qdq_plugin.rotation.mxfp4 import mxfp4_act_qdq as rotation_qdq


class RotationQDQEquivalenceTests(unittest.TestCase):
    GROUP_SIZE = 32

    def _assert_bit_exact(self, x: torch.Tensor) -> None:
        # native_qdq requires 2D [M, K]; rotation_qdq accepts arbitrary rank.
        a = native_qdq(x.clone(), group_size=self.GROUP_SIZE)
        b = rotation_qdq(x.clone(), self.GROUP_SIZE)
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        self.assertTrue(
            torch.equal(a, b),
            msg=(
                f"QDQ outputs differ for shape={tuple(x.shape)} dtype={x.dtype}: "
                f"max_abs_diff={(a.float() - b.float()).abs().max().item():.3e}"
            ),
        )

    def test_random_inputs_bf16_and_fp16(self) -> None:
        torch.manual_seed(0)
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                x = torch.randn(8, 128, dtype=dtype) * 3.0
                self._assert_bit_exact(x)

    def test_magnitude_edge_cases(self) -> None:
        torch.manual_seed(1)
        cases = {
            "normal_x3": torch.randn(16, 256, dtype=torch.bfloat16) * 3,
            "large_x1e3": torch.randn(16, 256, dtype=torch.bfloat16) * 1e3,
            "tiny_x1e-3": torch.randn(16, 256, dtype=torch.bfloat16) * 1e-3,
            "huge_x1e5": torch.randn(8, 64, dtype=torch.bfloat16) * 1e5,
            "fp4_grid_values": torch.tensor(
                [[6.0, 4.0, 3.0, 2.0] * 8], dtype=torch.bfloat16
            ),
            "all_zeros": torch.zeros(4, 32, dtype=torch.bfloat16),
        }
        for name, x in cases.items():
            with self.subTest(case=name):
                self._assert_bit_exact(x)

    def test_multiple_groups_per_row(self) -> None:
        # K spanning several groups exercises per-group scale independence.
        torch.manual_seed(2)
        x = torch.randn(4, self.GROUP_SIZE * 5, dtype=torch.bfloat16) * 7.0
        self._assert_bit_exact(x)


if __name__ == "__main__":
    unittest.main()
