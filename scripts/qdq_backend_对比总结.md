# MXFP4 激活 QDQ 后端对比：`VLLM_SPINQUANT_MXFP4_QDQ_BACKEND` vs 原生 vllm-qdq-plugin vs auto-round

本文对比 **MXFP4 激活 QDQ（quant-dequant）的舍入/缩放模式** 在三处实现里的差异，
并说明我们迁移的 rotation plugin 新增的 `VLLM_SPINQUANT_MXFP4_QDQ_BACKEND` 配置与它们的关系。

---

## 0. 先厘清：存在三种 QDQ「模式」

| 模式 | per-group scale | FP4 舍入 | 别名 |
|---|---|---|---|
| **even** | `2^(floor(log2(round(amax))) − 2)`（先把 amax 舍入再取指数）| round-half-to-even（就近偶数）| Quark "even" |
| **triton / hadamard** | `2^(floor(log2(amax)) − 2) / 0.75` | 显式中点阈值 round-to-nearest | QuaRot transform 核 |
| **rceil** | `_to_mx_rceil`（OCP MX round-ceil）| ceil 方向 | MX 规范最小实现 |

三处实现都围绕这三种模式，区别在于**“谁内置哪几种、由什么开关选择”**。

---

## 1. 三处实现各自怎么做

### A. 原生 vllm-qdq-plugin（`VLLM_QDQ` 激活模拟路径）

- 实现：`qdq/mxfp4.py::mxfp4_qdq`，在 `patch.py` 里 patch Marlin GEMM 时**硬编码**调用。
- 模式：**只有 even**（位运算实现 amax 舍入 + round-half-to-even）。
- 开关：**没有** QDQ 舍入模式的环境变量。`mxfp4_qdq(a, group_size=32)` 写死。

### B. auto-round（`auto_round_extension/vllm_ext`）

- 实现：`mxfp4_qdq_utils.py` 里有两套：
  - `qdq_mxfp4` → `fp4_121_scaled_even_rounding`：**even** 模式，用于
    `mxfp4_gemm_with_unpacked_weight`（FP8 预解包 GEMM 路径）。
  - `to_mxfp4_rceil` → `_to_mx_rceil`：**rceil** 模式，用于 `run_mxfp4_emulations`（纯模拟路径）。
- 选择方式：**没有专门的 QDQ-backend 环境变量**；模式与**权重运行时后端耦合**，由
  `VLLM_MXFP4_PRE_UNPACK_TO_FP8`（默认 `1`）/ `VLLM_MXFP4_PRE_UNPACK_WEIGHTS` 间接决定。
  代码里有一句 `# TODO: select the rounding mode based on config`。
  实际上 `linear_impl_mxfp4.py` 里非-FP8 分支 `raise NotImplementedError("Only
  VLLM_MXFP4_PRE_UNPACK_TO_FP8 is supported now.")` → **目前线性层实际只走 even**。
- 另外，QuaRot **transform（per-Linear Hadamard）** 推理走的是独立的 triton 核
  `quarot/utils/triton/mxfp4.py::mxfp4_forward_kernel` —— 即 **triton/hadamard 模式**，
  也是写死的（无 env 选择）。

### C. 我们迁移的 rotation plugin（`vllm-qdq-plugin/.../rotation/`）

- 实现：`rotation/mxfp4.py` 里有两套 + 一个 env 开关：
  - `mxfp4_act_qdq`：**even** 模式 →（已验证与原生 `mxfp4_qdq` **位级一致**，见
    `tests/test_rotation_qdq_equivalence.py`）。
  - `mxfp4_act_qdq_hadamard`：**triton/hadamard** 模式 →（已验证复刻 auto-round transform 核，
    cos≈0.9999995）。
- 两个 torch custom op：
  - `spinquant_mxfp4_act_qdq`（SpinQuant 路径）→ 受 `VLLM_SPINQUANT_MXFP4_QDQ_BACKEND` 控制。
  - `hadamard_mxfp4_act_qdq`（per-Linear Hadamard 路径）→ **固定 triton-match**，与 auto-round
    transform 对齐，不受 env 影响。

---

## 2. `VLLM_SPINQUANT_MXFP4_QDQ_BACKEND` 是什么、与它们有何区别

| 取值 | 选择的模式 | 等价对象 |
|---|---|---|
| 不设 / `even` / `quark`（默认）| even | == 原生 `mxfp4_qdq` == auto-round `qdq_mxfp4`（FP8 路径）|
| `triton` / `triton-match` / `hadamard` | triton/hadamard | == auto-round QuaRot transform triton 核 == 我们的 `mxfp4_act_qdq_hadamard` |

**关键区别**：

1. **这是我们独有的、唯一显式的“QDQ 舍入模式”开关。**
   - 原生 vllm-qdq-plugin：**无此 env**，even 写死。
   - auto-round：**无此 env**，模式与权重后端耦合（`VLLM_MXFP4_PRE_UNPACK_TO_FP8`），
     线性层实际固定 even；transform 路径固定 triton。
   - 我们：把“原本死配置”的 `VLLM_SPINQUANT_MXFP4_QDQ_BACKEND` **接成真开关**，
     默认 `even`（保持与上面两者**位级/语义一致**），可选 `triton` 做高级覆盖/实验对比。

2. **作用域只限 SpinQuant 的激活 QDQ。** per-Linear Hadamard 路径用独立 op，固定 triton-match，
   不受该 env 影响（因为它必须严格对齐 auto-round transform 核才能等价）。

3. **不包含 rceil。** rceil 只在 auto-round 出现，且其线性层路径当前 NotImplemented，
   故我们没有迁移它。

4. **兼容别名**：代码同时识别旧名 `AUTO_ROUND_MXFP4_QDQ_BACKEND`，但当前 auto-round 仓库
   里并不存在该变量（属历史/前向兼容命名）。

---

## 3. 一句话总结

- **原生 vllm-qdq-plugin** = even（硬编码，无开关）。
- **auto-round** = even（线性层，模式与 FP8 预解包后端耦合，无专门开关）+ triton（transform 路径，硬编码）；另有未启用的 rceil。
- **我们的 rotation plugin** = 唯一提供 `VLLM_SPINQUANT_MXFP4_QDQ_BACKEND` 显式开关：默认 `even`（与前两者一致），可切 `triton`；且 Hadamard 路径固定 triton-match 对齐 auto-round transform。

---

## 附：源码位置

- 原生 even：`vllm-qdq-plugin/src/vllm_qdq_plugin/qdq/mxfp4.py::mxfp4_qdq`，调用点 `patch.py`
- auto-round even/rceil：`auto-round/auto_round_extension/vllm_ext/mxfp4_qdq_utils.py`
  （`qdq_mxfp4` / `to_mxfp4_rceil`），路径选择 `linear_impl_mxfp4.py` + `envs_ext.py`
- auto-round transform triton 核：`auto-round/auto_round/algorithms/transforms/quarot/utils/triton/mxfp4.py`
- 我们的实现 + env：`vllm-qdq-plugin/src/vllm_qdq_plugin/rotation/mxfp4.py`
  （`mxfp4_act_qdq` / `mxfp4_act_qdq_hadamard` / `_resolve_qdq_backend`），env 定义 `envs.py`
