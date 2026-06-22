# AutoRound 中 QuaRot/SpinQuant Rotation + MXFP4 的 vLLM 插件分析

> 分析对象：`/storage/lkk/rotation_vllm_plugin/auto-round/`（当前分支 `vllm_selective_hadamard`）
> 以及配套的 `/storage/lkk/rotation_vllm_plugin/vllm-qdq-plugin/`。
>
> 本文从「为什么要 rotation」「量化期如何做 rotation+打包」「保存格式如何承载 rotation 元数据」「vLLM 推理期如何在线复原 rotation 并跑 MXFP4 GEMM」四条主线，梳理整套设计与代码实现。

---

## 0. 一句话总览

AutoRound 把 **QuaRot/SpinQuant 旋转（rotation）** 与 **MXFP4（microscaling FP4）量化** 拆成两段：

1. **量化期（offline，在 auto-round 内）**：对权重做 Hadamard/学习正交旋转，把 outlier 在通道维度上"摊平"，然后按 MXFP4（E2M1 + E8M0 group=32 scale）打包成 `uint8`，并把"哪些旋转需要在推理时在线补做"以 **buffer + config.json 元数据** 形式写进 checkpoint。
2. **推理期（online，在 vLLM 内）**：通过 **out-of-tree vLLM 插件**注册一个 `spinquant_mxfp4` 量化方法，加载时读取这些 buffer/元数据，重建旋转矩阵，在每个 Linear 的 forward 里对**激活**在线做旋转 + MXFP4 激活 QDQ，再跑权重反量化 GEMM。

整个链路的关键设计目标是：**旋转的数学不变性**（正交矩阵 `R @ Rᵀ = I`）使得"权重侧离线融合"和"激活侧在线旋转"可以拆开，从而能塞进 vLLM 标准的 `QuantizationConfig / LinearMethodBase` 扩展点，**零 vLLM 源码改动**。

---

## 1. 背景：为什么 rotation 对低比特量化有用

低比特量化（尤其 4-bit 及以下）最大的敌人是 **激活/权重里的离群值（outlier）**：少数通道幅值极大，导致 per-group scale 被拉大，其余数值的有效精度被吃掉。

**Hadamard / 正交旋转**的作用：用一个正交矩阵 `R` 对隐藏维做旋转 `x → x @ R`。因为旋转把能量在各通道间重新混合（类似随机投影），**重尾分布被"摊平"成接近高斯**，单通道 outlier 被分散，于是 per-group 量化误差显著下降。

代码里这个直觉被显式写进了 selective 决策注释（`rotation/selective.py:21-23`、`610-618`）：

> Hadamard rotation 帮助 heavy-tailed / expand / 有非线性缓冲的层，但**伤害** residual-sensitive / compress / 语义对齐的层（如 down_proj、o_proj、lm_head）。

### QuaRot vs SpinQuant 的区别

| | QuaRot | SpinQuant |
|---|---|---|
| 旋转矩阵 | **固定** Hadamard（确定性 Sylvester，或随机 ±1 sign-flip） | **可学习**正交矩阵，在 Stiefel 流形上用 Cayley/SGDG 优化训练 |
| 是否需训练 | 否（RTN 即可，production-ready） | 是（实验性，`trainable_rotation=True`） |
| 代码位置 | `spinquant/rotation_utils.py`、`known_hadamard.py` | `spinquant/cayley_optimizer.py`、`training_core.py`、`trainer.py` |
| 存储 type code | `0`(确定性) / `1`(随机) | `2`(trained) |

二者在 auto-round 里**共用同一套序列化与 vLLM 推理机制**，差别只在"旋转矩阵从哪来、怎么存"。

### R1/R2/R3/R4 四类旋转及其位置

SpinQuant 把 transformer 里能插旋转的位置分成 4 类（`spinquant/preprocessor.py`）：

| 旋转 | 作用维度 | 应用位置 | 推理期处理方式 |
|---|---|---|---|
| **R1** | hidden_size | q/k/v_proj、gate_proj、up_proj 的输入（残差流） | 可离线融权重，或**在线 hook/buffer** |
| **R2** | head_dim | attention 内，融进 v_proj 与 o_proj | **离线融进权重**，推理期无需处理 |
| **R3** | head_dim | RoPE 之后作用到 Q/K（monkeypatch `apply_rotary_pos_emb`） | **在线 hook** |
| **R4** | intermediate_size | MLP 里 down_proj 之前的激活 | 离线融，或**在线 buffer** |

> 数学不变性：对一个 Linear `y = x @ Wᵀ`，若令权重 `W' = W @ R`、并在输入侧做 `x' = x @ R`，则 `x' @ W'ᵀ = x@R@Rᵀ@Wᵀ = x@Wᵀ`，结果不变（`R` 正交）。R2 这类**两端都能融进权重**的就离线融掉；R1/R4 这类**输入侧需要在运行时旋转激活**的就保留为"在线"。

---

## 2. 量化期（auto-round 内）的 rotation 流水线

### 2.1 统一配置与后端分发

- **配置 schema 单一真源**：`rotation/config.py` 的 `RotationConfig`（pydantic）。关键字段：
  - `backend`：`"auto" | "inplace" | "transform"`
  - `hadamard_type`：`"hadamard" | "random_hadamard" | "quarot_hadamard"`
  - `block_size`：grouped Hadamard 的块大小（MXFP4 默认 32，NVFP4 默认 16，见 `normalize_rotation_config`）
  - `fuse_online_to_weight`、`allow_online_rotation`
  - **selective 字段**：`layer_selection`(`all|structural|auto`)、`include_layers`、`exclude_layers`、`kurtosis_threshold`、`ecr_threshold`、`score_threshold`

- **后端分发**：`rotation/dispatcher.py: resolve_hadamard_backend()`
  - `inplace`：QuaRot 风格的**残差流旋转**，任意 dtype 可用，可 `fuse_online_to_weight`。但"不支持真实导出，只能 fake 格式"（代码 warning）。
  - `transform`：**per-Linear 的权重+激活 Hadamard**，配 Triton fused kernel，**只支持 MXFP4/NVFP4，不能 fuse online**。
  - `auto`：要求 fuse → 选 inplace；data_type 是 mx/nv_fp → 选 transform；否则 inplace。

> **MXFP4 + vLLM 的主路径就是 `transform` 后端**：因为它把激活旋转保留成"在线"形式，正好对接 vLLM 插件的在线 R1。

### 2.2 transform 后端：per-Linear 旋转 + 量化期注入

入口：`rotation/apply.py: HadamardRotation.apply_to_model()`（backend=="transform" 分支，`apply.py:142-211`）：

1. 收集所有 `nn.Linear / QModuleBase` 模块。
2. 构造 `LayerSelector`（selective 决策，见 §2.3）；`auto` 模式先跑 activation profiling。
3. 逐层判断 `selector.should_rotate(name)`：
   - 跳过 `lm_head`。
   - 选中的层打标 `module._hadamard_rotate_enabled = True`，再调用 `_apply_to_module()`。
4. 通过 `_apply_weight_transform()`（`apply.py:324-360`）把 Hadamard 融进权重，并 **monkeypatch `WrapperLinear` / `WrapperWALayer`**（`rotation/patch.py`），使得 AutoRound 校准/调优时每次 forward 都对**被标记的层**应用旋转：
   - `_qdq_weight_patched`：旋转后的权重写回 `orig_layer.weight`（仅 `_hadamard_rotate_enabled` 的层，`patch.py:60-86`）。
   - `_qdq_act_patched`：激活先过 `inp_transform` 再量化（`patch.py:90-97`）。
   - 这些 patch 是**类级、幂等**（`_hadamard_patched` 守卫），但用**实例级 `_hadamard_rotate_enabled` 标志**实现 selective——这是 selective rotation 能精确到层的关键。
5. **打包**：`random_hadamard` 时 patch `QuantLinear.pack`（`patch.py:140-211`），在 MXFP4 打包同时把随机矩阵存成 buffer `hadamard_matrix`；MXFP4 走 `pack_fp4_to_uint8`，scale 走 E8M0（`(scales + 127).clamp(0,255).to(uint8)`）。

### 2.3 Selective Hadamard（本分支的核心创新）

`rotation/selective.py`（855 行）实现"**不是所有层都旋转**"：

- **structural 模式（零成本）**：基于层名 fnmatch 决策（`structural_decision`）：
  - 硬跳过 `STRUCTURAL_SKIP_PATTERNS`：`*lm_head*`、`*down_proj*`、`*o_proj*`、MoE router(`*mlp.gate`)、**MoE expert 的 gate/up/down_proj**（因为 vLLM 用 FusedMoE kernel，不支持在线旋转）、`*embed_tokens*`。
  - 强偏好 `STRUCTURAL_PREFER_PATTERNS`：`*up_proj* *gate_proj* *q_proj* *k_proj*`。
  - 条件 `*v_proj*`（因为 vLLM 把 q/k/v 合并成 qkv_proj，需要 load-time 补偿，见 §4.4）。
- **auto 模式（统计驱动）**：跑 ~32 条校准样本，统计每层输入激活的 **kurtosis（峰度→重尾）** 与 **ECR（energy concentration ratio→outlier 集中度）**，用 `compute_layer_score()` 打分：expand +1 / compress −1 / residual-consumer −1.5 / 非线性缓冲 +0.5 / 高峰度 +1 / 高 ECR +1 / structural-prefer +2；分数 ≥ `score_threshold`(默认 1.5) 才旋转。structural 硬跳过始终优先。
- 决策结果存 `model.rotation_decisions` 与 `model._rotated_layers`，供序列化时决定在哪些层注入 buffer。

### 2.4 桥接到 SpinQuant 序列化器（关键的"复用"设计）

`apply.py: _setup_serialize_bridge()`（`apply.py:219-250`）：Hadamard（QuaRot 风格的 per-linear 旋转）完成后，**合成一个 `SpinQuantConfig`**（`r1=True, online_r1_rotation=True, random_r1=(类型为 random), rotation_size=block_size`）挂到 `model._rotation_config`。

> 设计意图：让 QuaRot 的 per-linear Hadamard **复用 SpinQuant 已有的 buffer 注入 + config 序列化路径**，从而推理期可以用同一个 vLLM 插件加载。selective 模式还会置 `_selective_mode=True`，序列化时据 `_rotated_layers` 跳过未旋转层。

---

## 3. 保存格式：rotation 元数据如何写进 checkpoint

核心文件：`spinquant/serialize.py`。这是**量化期与推理期之间的契约**。

### 3.1 每个 QuantLinear 上注入的 buffer

前缀（`serialize.py:51-53`）：`_R1_PREFIX="spinquant_r1"`、`_R4_PREFIX="spinquant_r4"`。每层存：

- `{prefix}_type`：int32 标量 —— 旋转类型 code
  - `0` = 确定性 Hadamard（推理期由 size 重建，**不存矩阵**）
  - `1` = 随机 Hadamard（存 `int8` ±1 矩阵）
  - `2` = 训练得到的正交矩阵（存 `float32` 矩阵）
- `{prefix}_size`：int32 标量 —— 旋转块大小（`0` 表示该层**不旋转**）
- `{prefix}_matrix`：可选矩阵 buffer（仅 type 1/2 需要）

`_inject_rotation_buffers()`（`serialize.py:554-625`）：buffer 一律建在 **CPU**，确保即使参数在 `meta` 设备上 `save_pretrained()` 也能持久化；随机 Hadamard 存 `rotation_matrix.sign().to(int8)`，训练矩阵存完整 float32。

### 3.2 config.json 里的元数据

`save_spinquant_config()` 写入 `quantization_config.spinquant_config`，典型内容（见 `docs/selective_rotation_vllm_save.md`）：

```json
"spinquant_config": {
  "algorithm": "spinquant",
  "r1": true, "online_r1_rotation": true, "random_r1": false,
  "hidden_size": 896, "intermediate_size": 4864,
  "selective_rotation": true,
  "num_rotated_layers": 120,
  "rotated_layers": ["model.layers.0.mlp.gate_proj", "..."]
}
```

> **自然兼容性**：未旋转的层 `spinquant_r1_size` 默认为 0；推理期读到 size==0 直接跳过旋转，**无需 vLLM 插件做任何特判**。这就是 selective 模式能"自然兼容"的原因（`docs/selective_rotation_vllm_save.md` §原理）。

---

## 4. 推理期插件之一：auto-round 自带的 `spinquant_mxfp4` vLLM 插件

目录：`auto_round/vllm_plugin/`。这是**为 QuaRot/SpinQuant rotation + MXFP4 端到端推理而写的主插件**。

### 4.1 注册机制（zero vLLM 源码改动）

- `register.py: register()` 作为 vLLM general plugin 入口；或直接 `import auto_round.vllm_plugin`。
- `spinquant_mxfp4.py:71` 用 `@register_quantization_config("spinquant_mxfp4")` 把 `SpinQuantMXFP4Config` 注册进 vLLM 的量化方法表。
- `override_quantization_method()`（`:211-228`）**自动识别**：config.json 里有 `spinquant_config.online_r1_rotation=True` 且 `data_type` 含 `mxfp/mx_fp` → 自动启用 `spinquant_mxfp4`，用户无需显式指定 `quantization=`。
- `_weight_loading_patch.apply_weight_loading_patch()`：让 vLLM 的 `AutoWeightsLoader` 忽略 checkpoint 里那些**顶层、不对应任何 module 的旋转矩阵**（如 `spinquant_R2_head`）。

### 4.2 config 解析与三种 runtime backend

`from_config()`（`:136-209`）解析出 online_r1/r3/r4、各旋转 type、`rotation_size/hidden_size/head_dim/intermediate_size`，并读 `selective_rotation/rotated_layers`，推算 `_unrotated_suffixes`（哪些层类型整体未旋转，需在合并层里补偿）。

`runtime_backend`（由 env `AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND` 或 config 决定，默认 `preunpack_bf16`）：

| backend | 权重运行期形态 | GEMM 库 | 取舍 |
|---|---|---|---|
| `packed_fused` | 保持 MXFP4 packed | Triton fused dequant+GEMM | 省显存，但**无法对合并层做补偿** |
| `preunpack_bf16`（默认） | 加载时全量反量化成 BF16 dense | cuBLAS (`F.linear`) | 最稳，显存大 |
| `preunpack_fp8` | 加载时反量化成 FP8 | CUTLASS FP8 (`cutlass_scaled_mm`) | 折中 |

对应三个子类 `SpinQuantMXFP4{PackedFused,PreunpackBF16,PreunpackFP8}LinearMethod`，区别只在 `apply()` 的权重路径。

### 4.3 权重创建与加载后处理

`create_weights()`（`:253-390`）为每层注册：`weight_packed`(uint8 [N,K/2])、`weight_scale`(uint8 e8m0 [N,K/32])、`spinquant_r1/r4_type/size/matrix` buffer（**默认全 0**，对应未旋转），外加三套 backend 专用权重占位（`weight_dense_qdq / weight_unpacked_fp8 / weight_scale_bf16` 均先注册为 1 元素 dummy，避免 torch.compile 因缺 key 报错）。

`process_weights_after_loading()`（`:392-405`）：
1. `_process_rotation(layer,"r1")` 与 `("r4")` —— 把 buffer 变成运行期旋转状态。
2. `_prepare_runtime_weight_backend()` —— 按 backend 反量化权重并释放 packed 存储。
3. `_compensate_unrotated_partitions()` —— selective 合并层补偿。

`_process_rotation()`（`:407-498`）核心逻辑：
- 读 `type/size`；`size==0` → 释放矩阵、标记 `ROTATION_RUNTIME_NONE`（该层不旋转）。
- `type==0`(Hadamard)：用 `get_hadamard_K(rot_size)` 重建（支持非 2 的幂，如 3072=12×256 用已知小 Hadamard + butterfly `matmul_hadU`）。若 `rot_size==in_features` 走 butterfly runtime（不显式建矩阵）；否则按块建 block-diagonal Hadamard。
- `type==1/2`：从 `matrix` buffer 取出，必要时回退随机正交。
- 用完即 `delattr` 矩阵 buffer 省显存。

### 4.4 在线 forward：激活旋转 + MXFP4 激活 QDQ + 权重 GEMM

`_prepare_activations()`（`:654-666`）每次 forward 做三步：
1. **R1 在线旋转**（q/k/v/gate/up）：`_apply_rotation(layer,x,"r1")` —— butterfly(`matmul_hadU`) 或 `x @ R`，支持按 block reshape。
2. **R4 在线旋转**（down_proj）。
3. **MXFP4 激活 QDQ**：`torch.ops.auto_round.spinquant_mxfp4_act_qdq(x, group_size)` —— 模拟激活按 group=32 量化到 FP4 再反量化（与 Quark/vllm-ext 对齐），把"真实 MXFP4 GEMM 在激活侧的量化噪声"引入。

然后按 backend 跑权重 GEMM（packed→Triton fused / dense→`F.linear` / fp8→cutlass）。

> 这里 `spinquant_mxfp4_act_qdq` 与 `spinquant_mxfp4_linear` 都是用 `torch.library` 注册的**自定义 torch op**（文件底部 `_spinquant_mxfp4_*_impl/_fake`），带 fake 实现以兼容 torch.compile/CUDA graph。

### 4.5 Selective 合并层补偿（`_compensate_unrotated_partitions`，`:540-652`）

问题：vLLM 把 q/k/v 合并成一个 `qkv_proj`、gate/up 合并成 `gate_up_proj`。在线旋转对**整个合并输入**统一施加 `x@H`，但若 `v_proj` 在量化期**没被旋转**（structural 模式跳过 v_proj），它的权重就不该被旋转。

解法（load-time，零运行期开销）：对未旋转分区的权重**预先右乘 R 补偿**：

```
x_rot @ W_v_compᵀ = (x@H) @ (W_v@H)ᵀ = x@H@Hᵀ@W_vᵀ = x@W_vᵀ   （H 正交）
```

即把 `W_v` 替换成 `W_v @ H`，运行期被 `x@H` 一抵消，结果精确等于未旋转。代码按 `_output_partition_sizes`（QKV=[q,k,v]、gate_up=[gate,up]）定位需补偿的行区间，reshape 成 `(psize, -1, rot_size)` 后右乘 `R`。**注意**：`packed_fused` backend 因权重仍是 packed，无法补偿，会 warning 要求改用 `preunpack_bf16`。

---

## 5. 推理期插件之二：`vllm-qdq-plugin`（通用 QDQ 模拟）

目录：`vllm-qdq-plugin/`。这是一个**独立、通用**的 out-of-tree vLLM 插件，定位与 §4 不同：它**不引入 rotation**，只在已有量化 GEMM kernel 前对**激活做 QDQ 模拟**，用于研究"真实量化计算 vs 仅权重反量化"的精度影响。

- **注册**：`pyproject.toml` 的 `vllm.general_plugins` entry point `qdq = "vllm_qdq_plugin:register"`，vLLM 在主进程+所有 worker 自动加载。`register()` 检查 `VLLM_QDQ=1` 才打 patch。
- **patch 点**（`patch.py`）：monkeypatch `vllm._custom_ops.marlin_gemm` 与 `moe_wna16_marlin_gemm`，在调真 kernel 前对**输入激活**做 QDQ；按 `b_q_type` 路由（`float4_e2m1f`→MXFP4 QDQ，`float8_e4m3fn`→MXFP8 QDQ）。还 patch 了 `MLACommonImpl._compute_prefill_context` 修 packed 权重 dtype。
- **MXFP4 QDQ**（`qdq/mxfp4.py`）：pad K 到 32 的倍数 → reshape `[M,groups,32]` → per-group block_max → 取 2 的幂 E8M0 scale → 除以 scale → 模拟 round 到 E2M1 网格 `{0,0.5,1,1.5,2,3,4,6}` → 乘回。MXFP8 类似但量化到 E4M3FN（用纯算术模拟，不真转 fp8 dtype）。
- **env**：`VLLM_QDQ`（开关）、`VLLM_QDQ_TRACE`（打印 ≤200 行 trace）、`VLLM_MARLIN_MOE_QDQ_MODE=FORCE_MXFP4`（dtype 检测不足时强制）。
- 该仓库还附带 **sage3 / sparge attention** 后端（diffusion 注意力量化），与本主题关系较弱，仅作存在性说明。

> 两者关系：`spinquant_mxfp4`（§4）是**功能性插件**——真正承载 rotation + MXFP4 端到端推理；`vllm-qdq-plugin`（§5）是**实验/对齐工具**——给任意 Marlin MXFP4/MXFP8 模型在激活侧补上量化噪声以研究精度。二者都用 vLLM 的 general_plugins + monkeypatch 思路，"零源码改动"。

---

## 6. MXFP4 数据格式与 Triton kernel（共享底座）

### 6.1 MXFP4 格式

- **元素**：FP4 = E2M1，正值网格 `{0,0.5,1,1.5,2,3,4,6}`，4 bit（bit3=符号，bit0-2=幅值索引）。
- **打包**：每个 `uint8` 装 2 个 FP4（低 nibble=偶列，高 nibble=奇列），权重形状 `[N, K/2]`。
- **scale**：每 32 个元素一组，存 **E8M0**（8-bit 指数，`scale = 2^(e-127)`），形状 `[N, K/32]` uint8。

### 6.2 Triton kernel（`auto_round/triton_kernels/`）

- `mxfp4_dequant.py: triton_mxfp4_dequant()`：GPU 上把 packed 反量化（取 nibble → E2M1 查表 → 乘 E8M0 scale）。
- `mxfp4_gemm.py: triton_mxfp4_gemm()`：**fused dequant+GEMM**，`output = input @ dequant(W)ᵀ + bias`，分块 over M/N/K，每个 K-tile 内即时解包权重并 `tl.dot` 累加，**从不在显存里 materialize 完整反量化权重**。这是 `packed_fused` backend 的底层算子，也是 `rotated_linear.py` 融合 rotation 的基础。

### 6.3 `RotatedMXFP4Linear`（融合 rotation 的实验性 drop-in）

`triton_kernels/rotated_linear.py` 提供把"在线旋转 hook + MXFP4QuantLinear"融成单模块的优化：`forward` 里先 `input @ R` 再调 `triton_mxfp4_gemm`，消除 Python hook 回调开销与中间张量分配。还提供 `patch_mxfp4_forward_triton()` 这种"最轻量"路径——只把 dequant+`F.linear` 换成 triton kernel，保留原有旋转逻辑。这条线主要面向 auto-round 本地推理/profiling，与 vLLM 插件并行存在。

---

## 7. 端到端数据流（把一切串起来）

```
量化期 (auto-round):
  AutoRound(rotation_config={hadamard_type, block_size=32, layer_selection=...}, data_type="mx_fp")
   └─ apply_hadamard_rotation → backend=transform
       └─ HadamardRotation.apply_to_model
           ├─ LayerSelector 决定每层是否旋转 (structural/auto)
           ├─ 对选中层: 权重融 Hadamard + patch WrapperLinear(激活旋转)
           ├─ MXFP4 pack: weight_packed[N,K/2] + weight_scale[N,K/32](e8m0)
           └─ _setup_serialize_bridge → model._rotation_config = SpinQuantConfig(r1,online)
   └─ save_quantized:
       ├─ inject buffers: spinquant_r1_type/size(/matrix) (仅 rotated 层非零)
       └─ config.json: spinquant_config{online_r1_rotation, selective_rotation, rotated_layers,...}

推理期 (vLLM + auto_round.vllm_plugin):
  override_quantization_method → 自动选中 "spinquant_mxfp4"
   └─ SpinQuantMXFP4Config.from_config (解析 rotation + selective + runtime_backend)
   └─ create_weights (注册 packed 权重 + r1/r4 buffer 占位)
   └─ load checkpoint (只有 rotated 层 buffer 非零)
   └─ process_weights_after_loading:
       ├─ _process_rotation: buffer → 运行期旋转状态 (size==0 跳过)
       ├─ _prepare_runtime_weight_backend: 反量化到 bf16/fp8 或保持 packed
       └─ _compensate_unrotated_partitions: 合并层(qkv/gate_up)未旋转分区右乘 R 补偿
   └─ 每次 forward (apply):
       ├─ _prepare_activations: x@R1 (+x@R4) → spinquant_mxfp4_act_qdq(group=32)
       └─ 权重 GEMM: Triton fused / cuBLAS F.linear / CUTLASS FP8
```

---

## 8. 设计亮点与权衡总结

1. **正交不变性 → 关注点分离**：R2 这类两端可融的离线吃掉；R1/R4 这类需在线旋转激活的，用 buffer + 自定义 forward 在 vLLM 里复原。数学上精确无损。
2. **复用 SpinQuant 序列化承载 QuaRot**：QuaRot 的 per-linear Hadamard 通过合成 `SpinQuantConfig` 借道同一套 buffer/config 机制，推理端只需一个插件。
3. **Selective Hadamard**：用结构先验/激活统计避免对 down_proj、o_proj、lm_head、MoE expert 等"被旋转会掉点"的层施加旋转，既提精度又减少在线旋转开销。代价是引入"合并层补偿"复杂度。
4. **零 vLLM 源码改动**：靠 `register_quantization_config` + general_plugins + monkeypatch，主插件（功能）与 qdq 插件（实验）都遵循同一范式。
5. **三种 runtime backend 权衡显存/速度/精度**；`packed_fused` 最省显存但与 selective 补偿不兼容（需 `preunpack_bf16`）。
6. **torch.compile 友好**：所有 backend 参数名都预注册 dummy，自定义 op 带 fake 实现，避免编译期缺 key/动态图问题。

### 已知限制（来自代码注释/文档）
- `inplace` 后端"不支持真实导出，只能 fake 格式"。
- selective 的 `auto` 模式对 accelerate 多卡 dispatch 模型会跳过 profiling，退回 structural。
- MoE expert 层因 vLLM FusedMoE kernel 不支持在线旋转，被 structural 强制跳过。
- `random_hadamard + selective` 会在每个 rotated 层存完整 ±1 矩阵，模型略大；确定性 Hadamard 只存 type+size（运行期重建）。

---

## 附：关键文件索引

| 模块 | 文件 | 作用 |
|---|---|---|
| 旋转配置 | `auto_round/algorithms/transforms/rotation/config.py` | `RotationConfig` 单一真源 |
| 后端分发 | `.../rotation/dispatcher.py` | inplace/transform/auto 路由 |
| transform 应用 | `.../rotation/apply.py` | per-Linear 旋转 + serialize 桥接 |
| selective 决策 | `.../rotation/selective.py` | structural/auto 层选择 |
| 校准期 patch | `.../rotation/patch.py` | WrapperLinear/QuantLinear monkeypatch |
| SpinQuant 编排 | `.../spinquant/preprocessor.py` | R1-R4 初始化/融合/hook |
| 训练(SpinQuant) | `.../spinquant/cayley_optimizer.py`、`training_core.py` | Stiefel 流形优化 |
| **序列化契约** | `.../spinquant/serialize.py` | buffer 注入 + config 写入 + 在线重建 |
| **主 vLLM 插件** | `auto_round/vllm_plugin/spinquant_mxfp4.py` | rotation + MXFP4 端到端推理 |
| 插件注册 | `auto_round/vllm_plugin/register.py` / `__init__.py` | general_plugins 入口 |
| Triton kernel | `auto_round/triton_kernels/mxfp4_gemm.py`、`mxfp4_dequant.py` | fused dequant+GEMM |
| 融合旋转层 | `auto_round/triton_kernels/rotated_linear.py` | RotatedMXFP4Linear |
| 通用 QDQ 插件 | `vllm-qdq-plugin/src/vllm_qdq_plugin/patch.py`、`qdq/mxfp4.py` | Marlin 前激活 QDQ 模拟 |
| 设计文档(原仓库) | `auto-round/docs/selective_rotation_vllm_save.md` | selective 保存与 vLLM 推理说明 |
