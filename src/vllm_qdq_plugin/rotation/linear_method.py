# SPDX-License-Identifier: Apache-2.0
"""SpinQuant/QuaRot MXFP4 ``LinearMethodBase`` implementations (vendored).

Ported from auto-round's ``auto_round.vllm_plugin.spinquant_mxfp4`` with zero
auto_round dependency. Handles, per Linear layer:

  - Load-time reconstruction of R1/R4 rotation state from checkpoint buffers.
  - Backend-specific weight preparation (packed / pre-unpacked bf16 / fp8).
  - Selective-rotation compensation for unrotated partitions in merged layers.
  - Per-forward online activation rotation + MXFP4 activation QDQ + GEMM.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from .constants import (
    ROTATION_RUNTIME_HADAMARD,
    ROTATION_RUNTIME_MATRIX,
    ROTATION_RUNTIME_NONE,
    ROTATION_TYPE_HADAMARD,
    ROTATION_TYPE_RANDOM,
    ROTATION_TYPE_TRAINED,
    RUNTIME_BACKEND_PACKED_FUSED,
    RUNTIME_BACKEND_PREUNPACK_BF16,
    RUNTIME_BACKEND_PREUNPACK_FP8,
)
from ._mxfp4_common import (
    clear_packed_storage,
    dense_linear_stable_dtype,
    register_mxfp4_packed_weight,
)
from .hadamard import (
    build_block_hadamard,
    generate_random_orthogonal,
    get_hadamard_K,
    matmul_hadU,
)
from .mxfp4 import (
    dequant_packed_mxfp4_weight,
    dequant_preunpacked_mxfp4_weight,
    preunpack_mxfp4_weight,
)

logger = init_logger(__name__)

__all__ = [
    "SpinQuantMXFP4LinearMethod",
    "SpinQuantMXFP4PackedFusedLinearMethod",
    "SpinQuantMXFP4PreunpackBF16LinearMethod",
    "SpinQuantMXFP4PreunpackFP8LinearMethod",
]


class SpinQuantMXFP4LinearMethod(LinearMethodBase):
    """Base linear method implementing shared SpinQuant rotation handling."""

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.quant_config.group_size
        layer.params_dtype = params_dtype

        # Partition info for selective-rotation compensation at load time.
        # QKVParallelLinear -> [q, k, v]; MergedColumnParallelLinear -> [gate, up].
        layer._output_partition_sizes = output_partition_sizes

        # Packed MXFP4 weight + e8m0 scale (shared layout with the Hadamard method).
        register_mxfp4_packed_weight(
            layer,
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            group_size=group_size,
            extra_weight_attrs=extra_weight_attrs,
        )

        # R1 rotation metadata (scalar int32 buffers loaded from checkpoint).
        spinquant_r1_type = Parameter(torch.zeros((), dtype=torch.int32), requires_grad=False)
        set_weight_attrs(spinquant_r1_type, {"ignore_warning": True} | extra_weight_attrs)
        layer.register_parameter("spinquant_r1_type", spinquant_r1_type)

        spinquant_r1_size = Parameter(torch.zeros((), dtype=torch.int32), requires_grad=False)
        set_weight_attrs(spinquant_r1_size, {"ignore_warning": True} | extra_weight_attrs)
        layer.register_parameter("spinquant_r1_size", spinquant_r1_size)

        # R1 rotation matrix (for trained/random rotations). Hadamard stays zeros (unused).
        rot_size = self.quant_config.rotation_size or input_size_per_partition
        spinquant_r1_matrix = Parameter(
            torch.zeros(rot_size, rot_size, dtype=torch.float32), requires_grad=False
        )
        set_weight_attrs(spinquant_r1_matrix, {"ignore_warning": True} | extra_weight_attrs)
        layer.register_parameter("spinquant_r1_matrix", spinquant_r1_matrix)

        # R4 rotation metadata (down_proj). Shape (1,) for RowParallelLinear compat.
        spinquant_r4_type = Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(spinquant_r4_type, {"ignore_warning": True} | extra_weight_attrs)
        layer.register_parameter("spinquant_r4_type", spinquant_r4_type)

        spinquant_r4_size = Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(spinquant_r4_size, {"ignore_warning": True} | extra_weight_attrs)
        layer.register_parameter("spinquant_r4_size", spinquant_r4_size)

        r4_rot_size = self.quant_config.rotation_size or input_size_per_partition
        spinquant_r4_matrix = Parameter(
            torch.zeros(r4_rot_size, r4_rot_size, dtype=torch.float32), requires_grad=False
        )
        set_weight_attrs(spinquant_r4_matrix, {"ignore_warning": True} | extra_weight_attrs)
        layer.register_parameter("spinquant_r4_matrix", spinquant_r4_matrix)

        # Backend-specific weight placeholders (ALL backends register ALL names) so
        # torch.compile cached graphs never hit a missing key when switching backends.
        weight_dense_qdq = Parameter(torch.empty(1, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(weight_dense_qdq, {"ignore_warning": True})
        layer.register_parameter("weight_dense_qdq", weight_dense_qdq)

        weight_unpacked_fp8 = Parameter(torch.empty(1, dtype=torch.float8_e4m3fn), requires_grad=False)
        set_weight_attrs(weight_unpacked_fp8, {"ignore_warning": True})
        layer.register_parameter("weight_unpacked_fp8", weight_unpacked_fp8)

        weight_scale_bf16 = Parameter(torch.empty(1, dtype=torch.bfloat16), requires_grad=False)
        set_weight_attrs(weight_scale_bf16, {"ignore_warning": True})
        layer.register_parameter("weight_scale_bf16", weight_scale_bf16)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare runtime rotation state once after checkpoint loading."""
        self._process_rotation(layer, prefix="r1")
        self._process_rotation(layer, prefix="r4")
        self._prepare_runtime_weight_backend(layer)
        self._compensate_unrotated_partitions(layer)
        self._log_rotation_state_once(layer)

    @staticmethod
    def _runtime_name(code: int) -> str:
        return {
            ROTATION_RUNTIME_NONE: "none",
            ROTATION_RUNTIME_MATRIX: "matrix",
            ROTATION_RUNTIME_HADAMARD: "hadamard",
        }.get(code, str(code))

    def _log_rotation_state_once(self, layer: torch.nn.Module) -> None:
        """Emit a one-time summary of the prepared rotation/runtime state."""
        r1_runtime = getattr(layer, "_r1_rotation_runtime", ROTATION_RUNTIME_NONE)
        r4_runtime = getattr(layer, "_r4_rotation_runtime", ROTATION_RUNTIME_NONE)
        logger.info_once(
            "vllm-qdq-plugin: SpinQuant rotation prepared "
            "(backend=%s, r1_runtime=%s, r4_runtime=%s, online_r1=%s, online_r4=%s)",
            self.quant_config.runtime_backend,
            self._runtime_name(r1_runtime),
            self._runtime_name(r4_runtime),
            self.quant_config.online_r1,
            self.quant_config.online_r4,
        )

    def _process_rotation(self, layer: torch.nn.Module, prefix: str) -> None:
        """Prepare load-time rotation state for a given prefix (r1 or r4)."""
        type_attr = f"spinquant_{prefix}_type"
        size_attr = f"spinquant_{prefix}_size"
        matrix_attr = f"spinquant_{prefix}_matrix"
        result_attr = f"_{prefix}_rotation_matrix"
        rot_size_attr = f"_{prefix}_rot_size"
        runtime_attr = f"_{prefix}_rotation_runtime"
        hadamard_attr = f"_{prefix}_hadamard_K"
        hadamard_factor_attr = f"_{prefix}_hadamard_factor"

        setattr(layer, result_attr, None)
        setattr(layer, rot_size_attr, 0)
        setattr(layer, runtime_attr, ROTATION_RUNTIME_NONE)
        setattr(layer, hadamard_attr, None)
        setattr(layer, hadamard_factor_attr, 0)

        if not hasattr(layer, type_attr):
            return

        rot_type = int(getattr(layer, type_attr).item())
        rot_size = int(getattr(layer, size_attr).item())

        if rot_size == 0:
            if hasattr(layer, matrix_attr):
                delattr(layer, matrix_attr)
            return

        device = layer.weight_packed.device
        in_features = layer.weight_packed.shape[1] * 2
        R = None

        if rot_type == ROTATION_TYPE_HADAMARD:
            is_po2 = (rot_size & (rot_size - 1) == 0) and rot_size > 0
            if not is_po2 and not hasattr(self, f"_logged_{prefix}_non_po2"):
                try:
                    _, K = get_hadamard_K(rot_size)
                except (ValueError, ImportError):
                    K = "?"
                logger.info(
                    "vllm-qdq-plugin: %s rotation: size=%s (non-power-of-2, K=%s, using matmul_hadU butterfly)",
                    prefix.upper(), rot_size, K,
                )
                setattr(self, f"_logged_{prefix}_non_po2", True)
            hadamard_K, K = get_hadamard_K(rot_size)
            hadamard_K = hadamard_K.to(device=device, dtype=torch.float32)
            setattr(layer, rot_size_attr, rot_size)

            if rot_size == in_features:
                setattr(layer, runtime_attr, ROTATION_RUNTIME_HADAMARD)
                setattr(layer, hadamard_attr, hadamard_K)
                setattr(layer, hadamard_factor_attr, K)
                R = None
            elif in_features % rot_size == 0:
                R = build_block_hadamard(rot_size, hadamard_K, K, device)
            else:
                raise ValueError(
                    f"{prefix.upper()} rotation_size={rot_size} is not compatible "
                    f"with in_features={in_features}"
                )
        elif rot_type in (ROTATION_TYPE_RANDOM, ROTATION_TYPE_TRAINED):
            if hasattr(layer, matrix_attr):
                mat = getattr(layer, matrix_attr).data
                if mat.any():
                    if mat.shape[0] >= rot_size:
                        R = mat[:rot_size, :rot_size].to(device=device, dtype=torch.float32)
                    else:
                        logger.warning(
                            "%s shape %s < rot_size %s, falling back to random orthogonal matrix",
                            matrix_attr, tuple(mat.shape), rot_size,
                        )
                        R = generate_random_orthogonal(rot_size, device)
                else:
                    logger.warning(
                        "rot_type=%s (random/trained) but no matrix in checkpoint for %s, "
                        "generating random orthogonal matrix", rot_type, prefix,
                    )
                    R = generate_random_orthogonal(rot_size, device)
            else:
                R = generate_random_orthogonal(rot_size, device)

        if R is not None:
            # Store in the activation dtype so the per-forward ``.to(x.dtype)`` in
            # _apply_rotation is a no-op (no copy) in the common case, while still
            # being correct if activations arrive in a different dtype.
            setattr(layer, result_attr, R.to(getattr(layer, "params_dtype", torch.bfloat16)))
            setattr(layer, rot_size_attr, rot_size)
            setattr(layer, runtime_attr, ROTATION_RUNTIME_MATRIX)

        if hasattr(layer, matrix_attr):
            delattr(layer, matrix_attr)

    def _prepare_runtime_weight_backend(self, layer: torch.nn.Module) -> None:
        """Prepare the load-time weight representation for the selected backend."""
        backend = self.quant_config.runtime_backend
        if backend == RUNTIME_BACKEND_PACKED_FUSED:
            return

        target_dtype = getattr(layer, "params_dtype", torch.bfloat16)
        if backend == RUNTIME_BACKEND_PREUNPACK_BF16:
            weight_dense = dequant_packed_mxfp4_weight(
                layer.weight_packed.data,
                layer.weight_scale.data,
                self.quant_config.group_size,
                target_dtype=target_dtype,
            )
            layer.weight_dense_qdq = Parameter(weight_dense, requires_grad=False)
            clear_packed_storage(layer)
            return

        if backend == RUNTIME_BACKEND_PREUNPACK_FP8:
            weight_fp8, scale_bf16 = preunpack_mxfp4_weight(
                layer.weight_packed.data,
                layer.weight_scale.data,
                self.quant_config.group_size,
            )
            layer.weight_unpacked_fp8 = Parameter(weight_fp8, requires_grad=False)
            layer.weight_scale_bf16 = Parameter(scale_bf16, requires_grad=False)
            clear_packed_storage(layer)
            return

        raise ValueError(f"Unsupported SpinQuant runtime backend: {backend}")

    def _compensate_unrotated_partitions(self, layer: torch.nn.Module) -> None:
        """Apply load-time rotation to unrotated partitions in merged layers.

        When selective rotation skips a layer (e.g. v_proj) that vLLM merges with
        rotated layers (qkv_proj merges q+k+v), the online rotation is applied
        uniformly to the full input. We compensate by rotating the unrotated
        partition's weight at load time:

            x_rot @ W_comp.T = (x@H) @ (W@H).T = x @ H @ H.T @ W.T = x @ W.T

        which is mathematically exact (H orthogonal) with zero runtime overhead.
        """
        if not self.quant_config.selective_rotation:
            return
        if not self.quant_config._unrotated_suffixes:
            return

        rot_size = getattr(layer, "_r1_rot_size", 0)
        if rot_size == 0:
            return

        partition_sizes = getattr(layer, "_output_partition_sizes", None)
        if partition_sizes is None or len(partition_sizes) <= 1:
            return

        unrotated_suffixes = self.quant_config._unrotated_suffixes
        if len(partition_sizes) == 3:
            partition_suffixes = ["q_proj", "k_proj", "v_proj"]
        elif len(partition_sizes) == 2:
            partition_suffixes = ["gate_proj", "up_proj"]
        else:
            return

        compensate_indices = [
            i for i, suffix in enumerate(partition_suffixes) if suffix in unrotated_suffixes
        ]
        if not compensate_indices:
            return

        R = getattr(layer, "_r1_rotation_matrix", None)
        runtime = getattr(layer, "_r1_rotation_runtime", ROTATION_RUNTIME_NONE)

        if R is None and runtime == ROTATION_RUNTIME_HADAMARD:
            hadamard_K = getattr(layer, "_r1_hadamard_K", None)
            hadamard_factor = getattr(layer, "_r1_hadamard_factor", 0)
            if hadamard_K is not None:
                R = build_block_hadamard(rot_size, hadamard_K, hadamard_factor, hadamard_K.device)
        elif R is None:
            return

        backend = self.quant_config.runtime_backend
        if backend == RUNTIME_BACKEND_PREUNPACK_BF16:
            weight = layer.weight_dense_qdq.data
        elif backend == RUNTIME_BACKEND_PREUNPACK_FP8:
            weight = layer.weight_unpacked_fp8.data
        elif backend == RUNTIME_BACKEND_PACKED_FUSED:
            logger.warning_once(
                "vllm-qdq-plugin: Selective rotation with packed_fused backend cannot "
                "compensate unrotated partitions. Use preunpack_bf16 backend."
            )
            return
        else:
            return

        R_compute = R.to(dtype=torch.float32, device=weight.device)
        row_start = 0
        compensated_count = 0
        for i, psize in enumerate(partition_sizes):
            if i in compensate_indices:
                row_end = row_start + psize
                W_part = weight[row_start:row_end].float()
                in_features = W_part.shape[-1]
                if in_features % rot_size == 0:
                    W_reshaped = W_part.reshape(psize, -1, rot_size)
                    W_compensated = (W_reshaped @ R_compute).reshape(psize, in_features)
                    weight[row_start:row_end] = W_compensated.to(weight.dtype)
                    compensated_count += 1
            row_start += psize

        if compensated_count > 0 and not getattr(self, "_logged_compensation", False):
            logger.info(
                "vllm-qdq-plugin: Selective rotation: compensated %d unrotated partition(s) "
                "in merged layer (zero runtime overhead)", compensated_count,
            )
            self._logged_compensation = True

    def _prepare_activations(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply online rotation + activation QDQ before the backend GEMM.

        Note: no logging here — this runs inside vLLM's torch.compile graph, where
        a Python logging call would force a graph break / compile error. Rotation
        activity is reported once at load time via ``_log_rotation_state_once``.
        """
        if self.quant_config.online_r1:
            x = self._apply_rotation(layer, x, prefix="r1")
        if self.quant_config.online_r4:
            x = self._apply_rotation(layer, x, prefix="r4")
        x = torch.ops.vllm_qdq_plugin.spinquant_mxfp4_act_qdq(x, self.quant_config.group_size)
        return x

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compatibility dispatcher for direct instantiation in tests/local usage."""
        backend = self.quant_config.runtime_backend
        if backend == RUNTIME_BACKEND_PREUNPACK_BF16:
            return SpinQuantMXFP4PreunpackBF16LinearMethod(self.quant_config).apply(layer, x, bias)
        if backend == RUNTIME_BACKEND_PREUNPACK_FP8:
            return SpinQuantMXFP4PreunpackFP8LinearMethod(self.quant_config).apply(layer, x, bias)
        return SpinQuantMXFP4PackedFusedLinearMethod(self.quant_config).apply(layer, x, bias)

    @staticmethod
    def _apply_rotation(layer: torch.nn.Module, x: torch.Tensor, prefix: str) -> torch.Tensor:
        """Apply a prepared rotation in the lightest available runtime form."""
        runtime = getattr(layer, f"_{prefix}_rotation_runtime", ROTATION_RUNTIME_NONE)
        rot_size = getattr(layer, f"_{prefix}_rot_size", 0)

        if runtime == ROTATION_RUNTIME_HADAMARD:
            hadamard_K = getattr(layer, f"_{prefix}_hadamard_K", None)
            hadamard_factor = getattr(layer, f"_{prefix}_hadamard_factor", 0)
            if hadamard_K is None or rot_size == 0:
                return x
            return matmul_hadU(
                x,
                hadamard_K=hadamard_K.to(device=x.device, dtype=x.dtype),
                K=hadamard_factor,
            ).to(x.dtype)

        R = getattr(layer, f"_{prefix}_rotation_matrix", None)
        if R is None:
            return x
        R = R.to(dtype=x.dtype)
        in_features = x.shape[-1]
        if rot_size == in_features:
            return x @ R
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, rot_size)
        return (x @ R).reshape(shape)


class SpinQuantMXFP4PackedFusedLinearMethod(SpinQuantMXFP4LinearMethod):
    """Packed low-bit runtime: activation qdq + fused weight dequant/GEMM."""

    def apply(self, layer, x, bias=None):
        x = self._prepare_activations(layer, x)
        output = torch.ops.vllm_qdq_plugin.spinquant_mxfp4_linear(
            x, layer.weight_packed, layer.weight_scale, self.quant_config.group_size
        )
        if bias is not None:
            output = output + bias
        return output


class SpinQuantMXFP4PreunpackBF16LinearMethod(SpinQuantMXFP4LinearMethod):
    """Pre-unpack BF16 runtime: load-time full dequant to BF16 + activation qdq + F.linear."""

    def apply(self, layer, x, bias=None):
        x = self._prepare_activations(layer, x)
        return dense_linear_stable_dtype(layer.weight_dense_qdq, x, bias)


class SpinQuantMXFP4PreunpackFP8LinearMethod(SpinQuantMXFP4LinearMethod):
    """Pre-unpack runtime: load-time FP8 unpack + per-forward weight restore + F.linear."""

    def apply(self, layer, x, bias=None):
        x = self._prepare_activations(layer, x)
        weight_dense = dequant_preunpacked_mxfp4_weight(
            layer.weight_unpacked_fp8,
            layer.weight_scale_bf16,
            target_dtype=getattr(layer, "params_dtype", x.dtype),
        )
        return dense_linear_stable_dtype(weight_dense, x, bias)
