# vllm-qdq-plugin Rotation 推理：环境变量配置说明

本文件汇总在 **vllm-qdq-plugin** 中使用两类 rotation + MXFP4 推理插件时需要配置的环境变量：

1. **SpinQuant / QuaRot（R1/R4 在线旋转）** → 量化配置名 `spinquant_mxfp4`
2. **Per-Linear / Random Hadamard（block-diagonal 32×32）** → 量化配置名 `hadamard_mxfp4`

> 两者针对**不同的 checkpoint**：前者匹配带 `spinquant_config` 的模型，后者匹配带
> `rotation_config.hadamard_type ∈ {hadamard, random_hadamard}` 的 auto-round 模型。
> 每个 checkpoint 会**自动识别**只属于自己的那个量化配置，互不干扰。

---

## 0. 前置条件（两类插件通用）

1. **安装插件**（注册为 vLLM general plugin entry point）：
   ```bash
   pip install -e /storage/lkk/rotation_vllm_plugin/vllm-qdq-plugin
   ```
2. **环境变量必须在 vLLM 进程启动前设置**（plugin 在 vLLM `load_general_plugins()` 时读取）。
   命令行前缀写法即可，例：`VLLM_SPINQUANT_MXFP4=1 python xxx.py`。
3. 这些 `VLLM_*` 变量已通过 `register_with_vllm_envs()` 注册进 vLLM env 表，
   **不会**再触发 "Unknown vLLM environment variable" 警告。

---

## 1. SpinQuant / QuaRot（`spinquant_mxfp4`）

| 环境变量 | 必需 | 取值 | 默认 | 说明 |
|---|---|---|---|---|
| `VLLM_SPINQUANT_MXFP4` | ✅ | `1`/`true`/`yes` 开启 | `0` | 注册 `spinquant_mxfp4` 量化配置，启用 R1/R4 在线旋转 + MXFP4 激活 QDQ |
| `VLLM_SPINQUANT_RUNTIME_BACKEND` | 可选 | `preunpack_bf16` / `preunpack_fp8` / `packed_fused` | 跟随 checkpoint | 权重运行时反量化/GEMM 后端，见下表 |
| `VLLM_SPINQUANT_MXFP4_QDQ_BACKEND` | 可选 | 字符串（前向兼容预留） | `""` | 激活 QDQ 后端覆盖；本仓库仅内置纯 PyTorch "even" 模式，留空即用它 |

**`VLLM_SPINQUANT_RUNTIME_BACKEND` 取值含义**：

| 取值 | GEMM 实现 | 备注 |
|---|---|---|
| `preunpack_bf16` | cuBLAS（`torch.nn.functional.linear`）| **默认**，最通用 |
| `preunpack_fp8` | CUTLASS FP8（`vllm.cutlass_scaled_mm`）| 需 FP8 算力支持 |
| `packed_fused` | Triton 融合核 | 保留打包权重，融合反量化 |

**取值优先级**（高 → 低）：
`VLLM_SPINQUANT_RUNTIME_BACKEND` 环境变量
→ 旧名 `AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND`
→ checkpoint 的 `spinquant_config.runtime_backend`
→ checkpoint 顶层 `runtime_backend`
→ 默认 `preunpack_bf16`。

**示例**：
```bash
CUDA_VISIBLE_DEVICES=7 \
VLLM_SPINQUANT_MXFP4=1 \
VLLM_SPINQUANT_RUNTIME_BACKEND=preunpack_bf16 \
python your_vllm_script.py --model /path/to/spinquant_mxfp4_ckpt
```

---

## 2. Per-Linear / Random Hadamard（`hadamard_mxfp4`）

| 环境变量 | 必需 | 取值 | 默认 | 说明 |
|---|---|---|---|---|
| `VLLM_HADAMARD_MXFP4` | ✅ | `1`/`true`/`yes` 开启 | `0` | 注册 `hadamard_mxfp4` 量化配置，启用 per-Linear block-Hadamard（32×32）在线逆旋转 + MXFP4 激活 QDQ |

**说明**：
- 自动从 checkpoint 的 `rotation_config.hadamard_type` 识别 `hadamard`（确定性 Sylvester）
  或 `random_hadamard`（随机），并校验 data_type 为 mxfp。
- **运行时后端固定**为 `preunpack_bf16`（一次性反量化为 bf16 dense，再走 cuBLAS GEMM），
  因此**没有** runtime-backend / qdq-backend 环境变量需要配置。
- 激活 QDQ 使用与 auto-round triton 融合核对齐的 `hadamard_mxfp4_act_qdq`
  op（scale ÷0.75 + midpoint round-to-nearest），与 SpinQuant 的 "even" 模式不同。

**示例**：
```bash
CUDA_VISIBLE_DEVICES=7 \
VLLM_HADAMARD_MXFP4=1 \
python your_vllm_script.py --model /path/to/hadamard_mxfp4_ckpt
```

---

## 3. 其它相关变量（非 rotation 必需）

| 环境变量 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `VLLM_QDQ` | `1`/... | `0` | 启用通用激活 QDQ patch（独立特性，rotation 推理**不需要**）|
| `VLLM_QDQ_TRACE` | `1`/... | `0` | QDQ 追踪日志 |

> SpinQuant / Hadamard 两个 rotation 插件**不依赖** `VLLM_QDQ`，单独开各自的
> `VLLM_SPINQUANT_MXFP4` / `VLLM_HADAMARD_MXFP4` 即可。

---

## 4. 常见组合速查

| 场景 | 需要设置 |
|---|---|
| 跑 SpinQuant/QuaRot MXFP4 模型（默认后端）| `VLLM_SPINQUANT_MXFP4=1` |
| SpinQuant 指定 FP8 GEMM | `VLLM_SPINQUANT_MXFP4=1 VLLM_SPINQUANT_RUNTIME_BACKEND=preunpack_fp8` |
| 跑 per-Linear / random Hadamard MXFP4 模型 | `VLLM_HADAMARD_MXFP4=1` |
| 多卡（TP）跑 Hadamard 模型 | `VLLM_HADAMARD_MXFP4=1` + `tensor_parallel_size=N`（详见分析文档 §7.6）|

---

## 5. 验证日志

成功启用后，引擎初始化日志会出现（`info` 级）：

- SpinQuant：
  `vllm-qdq-plugin: registered 'spinquant_mxfp4' quantization config (VLLM_SPINQUANT_MXFP4 enabled)`
  以及每个 Linear 的
  `... SpinQuant MXFP4 ... runtime_backend=preunpack_bf16 (cuBLAS ...)`
- Hadamard：
  `vllm-qdq-plugin: registered 'hadamard_mxfp4' quantization config (VLLM_HADAMARD_MXFP4 enabled)`
  以及
  `vllm-qdq-plugin: Hadamard MXFP4 rotation prepared (type=..., block_size=32, online input rotation=Hᵀ, per-partition=uniform(fast)|distinct xN(faithful), runtime=preunpack_bf16)`

日志里 `quantization=spinquant_mxfp4` 或 `quantization=hadamard_mxfp4` 出现在
`Initializing a V1 LLM engine ... with config:` 行，表示量化配置已被正确自动识别。

---

## 附：相关源码

- 环境变量定义：`vllm-qdq-plugin/src/vllm_qdq_plugin/envs.py`
- 插件注册入口：`vllm-qdq-plugin/src/vllm_qdq_plugin/__init__.py::register`
- SpinQuant 配置 / 方法：`rotation/config.py`、`rotation/linear_method.py`
- Hadamard 配置 / 方法：`rotation/perlinear_config.py`、`rotation/perlinear_linear_method.py`
- MXFP4 算子：`rotation/mxfp4.py`
