# SPDX-License-Identifier: Apache-2.0
"""Shared constants for the SpinQuant/QuaRot MXFP4 rotation plugin.

Values MUST match auto-round's serialization
(``auto_round.algorithms.transforms.spinquant.serialize``) so checkpoints
exported by auto-round load and run correctly.
"""

from __future__ import annotations

MXFP4_BLOCK_SIZE = 32

# Runtime weight backends.
RUNTIME_BACKEND_PACKED_FUSED = "packed_fused"
RUNTIME_BACKEND_PREUNPACK_BF16 = "preunpack_bf16"
RUNTIME_BACKEND_PREUNPACK_FP8 = "preunpack_fp8"
VALID_RUNTIME_BACKENDS = {
    RUNTIME_BACKEND_PACKED_FUSED,
    RUNTIME_BACKEND_PREUNPACK_BF16,
    RUNTIME_BACKEND_PREUNPACK_FP8,
}

# Rotation type codes (stored in the ``spinquant_{r1,r4}_type`` buffer).
ROTATION_TYPE_HADAMARD = 0
ROTATION_TYPE_RANDOM = 1
ROTATION_TYPE_TRAINED = 2

# Rotation runtime modes prepared once at load time.
ROTATION_RUNTIME_NONE = 0
ROTATION_RUNTIME_MATRIX = 1
ROTATION_RUNTIME_HADAMARD = 2
