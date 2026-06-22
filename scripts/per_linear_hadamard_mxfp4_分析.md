# auto-round Per-Linear / Random Hadamard + MXFP4 分析与 vLLM Plugin 可行性

> 范围：`auto_round/algorithms/transforms/quarot`（**QuaRot transform 后端**），
> 即 `rotation_config="default"`（deterministic hadamard）与
> `rotation_config="random_hadamard"`。
> 这与之前迁移到 `vllm-qdq-plugin` 的 **SpinQuant R1/R2/R3/R4** 是**两套不同机制**，
> 见文末对比。

测试脚本：
- `t_8B_hadamard_mxfp4_rtn.py` → `rotation_config="default"`
- `t_8B_random_hadamard_mxfp4_rtn.py` → `rotation_config="random_hadamard"`
- 二者都是 `AutoRound("Qwen/Qwen3-8B", scheme="MXFP4", iters=0, ...)`，RTN（无训练）量化后
  `format="auto_round"` 保存。

已落盘模型：
- `Qwen3-8_autoround_hadamard_rtn_mxfp4/Qwen3-8B-mxfp-w4g32/`
- `Qwen3-8_autoround_random_hadamard_rtn_mxfp4/Qwen3-8B-mxfp-w4g32/`

---

## 1. 核心机制：block-diagonal 32×32 Hadamard

per-Linear Hadamard **不是**对整个 hidden_size 做一个大旋转，而是把权重的输入维度按
`block_size = group_size = 32` 切块，每一块乘一个 **32×32** 的 Hadamard 矩阵
（等价于乘一个 block-diagonal 的大矩阵 `blockdiag(H, H, …, H)`）。

这样做的目的：MXFP4 是 per-group（group_size=32）量化，每 32 个元素共享一个 e8m0 scale。
把 Hadamard 限制在 32 元素的 group 内部，可以**在每个量化 group 内部打散 outlier**，
让组内分布更均匀、量化误差更小，同时保证旋转矩阵与量化分组完全对齐、无需跨组通信。

| 概念 | 取值 |
|---|---|
| `block_size` | 32（mx_fp 默认，`config.py::normalize_rotation_config`；nv_fp 为 16）|
| Hadamard 矩阵阶数 | 32×32，正交归一（`H @ Hᵀ = I`），元素 `±1/√32` |
| 作用范围 | **每个 Linear 的输入维度**，按 32 分块，block-diagonal |
| 融合时机 | 标定时把 `H` 融进权重；推理时对激活做在线逆旋转 |

### 后端选择（`quarot/dispatcher.py::resolve_hadamard_backend`）

QuaRot 有三个后端：
- `inplace`：经典 QuaRot 残差流旋转，任意 dtype，可把在线旋转 fuse 进权重；
- `transform`：**per-Linear 权重+激活 Hadamard**，triton 融合，**仅支持 MXFP4/NVFP4**，
  不能把在线旋转 fuse 掉；
- `auto`（默认）：若要求 fuse→`inplace`；否则若 `data_type` 是 mx_fp/nv_fp→`transform`；否则 `inplace`。

本场景 `scheme="MXFP4"` + `backend="auto"` → **解析为 `transform`**。
落盘 config 里 `rotation_config.backend` 仍写 `"auto"`，但实际走的是 transform 路径。

---

## 2. deterministic vs random：唯一区别是那块 32×32 矩阵

| | `hadamard`（default）| `random_hadamard` |
|---|---|---|
| 32×32 矩阵 | Sylvester 确定性 Hadamard（`deterministic_hadamard_matrix`）| 随机 ±1 对角 × Hadamard（`random_hadamard_matrix`）|
| 是否对称 | 对称（`Hᵀ = H`）| 一般不对称 |
| 是否存进 checkpoint | **否**（加载时重新生成）| **是**（每个 Linear 存一份 `hadamard_matrix` buffer）|
| 推理时来源 | 重新生成 Sylvester 矩阵 | 从 checkpoint 读 `hadamard_matrix` |

### 关键实测结论（用 `inspect_hadamard_model.py` 验证）

对 `random_hadamard` 模型：

```
rotation_config: {'algorithm':'hadamard','backend':'auto','block_size':32,
                  'hadamard_type':'random_hadamard', ...}
hadamard_matrix buffers count: 252        # = 36 层 × 7 个 Linear（q/k/v/o/gate/up/down）
shape: (32, 32)  ALL identical: True      # ← 252 份全部一模一样
orthonormal (H@H.T == I): True            # max off-diag ~6e-8
unique |values|: [0.17677...]  == 1/sqrt(32)
```

> **重要发现**：虽然代码每个 Linear 都 `register_buffer("hadamard_matrix", …)` 单独存了一份，
> 但 **252 份矩阵完全相同** —— 实际上是“**一个共享的随机 Hadamard**”。
> 原因：`RandomHadamardTransform` 在无 seed/generator 时用默认构造的 `torch.Generator()`，
> 其默认种子是固定的，于是每次 `build_hadamard_transform` 生成的随机符号串都一样。
> 这对 vLLM plugin 是个**极大简化**：可以只取一份 32×32 矩阵，不必管 per-Linear 差异。

对 `hadamard`（deterministic）模型：

```
rotation_config: {... 'hadamard_type':'hadamard', ...}
hadamard_matrix buffers count: 0          # ← 一份都不存，推理时重新生成
```

层内 key 对比：
```
random:        q_proj.{weight_packed, weight_scale, hadamard_matrix}
deterministic: q_proj.{weight_packed, weight_scale}
```

---

## 3. 数学等价性（为什么能消掉）

设原始计算 `y = x · Wᵀ`。标定时把 `H` 融进权重，推理时对激活做逆旋转：

- 权重融合（`apply.py::_apply_weight_transform`，location=`"weight"`）：
  `W_rot = blockdiag-matmul(W, Hᵀ)`（每 32 列块 `W_blk @ Hᵀ`）。
- 激活在线旋转（`_apply_input_transform` 注册的 forward pre-hook，location=`"input"`，inverse）：
  - random：用 `hadamard_matrix.T`（即 `Hᵀ`）→ `x_rot = blockdiag-matmul(x, Hᵀ)`
  - deterministic：用重新生成的对称 `H` → `x_rot = blockdiag-matmul(x, H)`

逐 block 验证（`H` 正交归一，`Hᵀ H = I`）：

```
y_rot = x_rot · W_rotᵀ
      = (x Hᵀ) (W Hᵀ)ᵀ
      = x Hᵀ H Wᵀ
      = x (Hᵀ H) Wᵀ = x Wᵀ   ✓  (random)
```
deterministic 同理（`H` 对称：`H H = I`）。

> 所以**误差只来自 MXFP4 量化本身**，旋转在数学上严格抵消。Hadamard 的作用是让
> 组内分布更平、量化更准。

block-diagonal 的实现见 `quarot/utils/matrix.py::multihead_matmul`：当
`A.shape[-1] = num_heads × B.shape[-2]` 时，把 `A` reshape 成 `(…, num_heads, 32)` 再 `@ B`，
等价于乘 `blockdiag(B,…,B)`。

---

## 4. 序列化与 HF 推理路径

### 落盘内容
- 权重：`weight_packed`（MXFP4 packed uint8，`[N, K//2]`）+ `weight_scale`（e8m0 uint8，`[N, K//32]`）。
  权重已是 **旋转后再量化** 的结果。
- random：每个 Linear 多一个 `hadamard_matrix`（32×32 fp32）。
- config.json `quantization_config.rotation_config` 记录 `hadamard_type / block_size / backend`。

### HF 推理（`eval_hf.py` → transformers → `inference/convert_model.py`）
1. `_replace_by_quant_layers` 把 Linear 换成 `MXFP4QuantLinear`（`experimental/qmodules/mx.py`）。
2. 检测到 `rotation_config` → 调 `apply_rotation_hooks_from_config`
   （`transforms/__init__.py`），对每个 Linear 注册 **forward pre-hook**，
   在线对输入激活做逆 Hadamard（location=`"input"`）。
3. `MXFP4QuantLinear.forward` 自己**不做** Hadamard，只做激活 qdq + 权重反量化 + `F.linear`；
   Hadamard 完全由 pre-hook 负责。

### HF smoke test 实测（`smoke_hf_hadamard.py`，GPU7，random 模型）
```
registered forward_pre_hooks (rotation hooks): 252   # = 每个量化 Linear 一个
OUTPUT: '<think>\nOkay, the user is asking for the capital of France ...'
```
→ **HF backend 推理正常**，旋转 hook 全部就位，输出连贯。

---

## 5. vLLM Plugin 可行性评估

**结论：可行，而且比 SpinQuant plugin 更简单**，几乎能复用现有 `vllm-qdq-plugin/rotation` 的全部基础设施。

### 为什么简单
1. **统一作用于每个 Linear 的输入**，旋转结构对所有 Linear 相同（block=32 的 block-diagonal）。
2. **矩阵只有一个**（random 实测 252 份全同；deterministic 直接重建）。
3. **合并层（常规情形）无需补偿**：qkv_proj / gate_up_proj 共享同一输入 hidden，当各分区
   Hadamard 相同时输入侧 block-Hadamard 只作用一次、与 output 分块无关——不像 SpinQuant
   选择性旋转需要对未旋转分区做权重补偿。
   （**真实 random** 各分区矩阵不同时，需按分区独立 rotate→qdq→GEMM→concat，见 §7.5。）
4. 现有 op 全部可复用：
   - 激活 MXFP4 qdq：`torch.ops.vllm_qdq_plugin.spinquant_mxfp4_act_qdq`
   - 权重反量化（preunpack_bf16 / fp8 / packed_fused）：`rotation/mxfp4.py`
   - block 旋转：现有 `_apply_rotation` 当 `rot_size != in_features` 时正是
     `x.reshape(…, -1, rot_size) @ R`，把 `rot_size=32`、`R=Hᵀ` 即可。

### 推理前向（每个 Linear）
```
x_rot = blockdiag-matmul(x, Hᵀ)          # rot_size=32 的 block 旋转
x_q   = spinquant_mxfp4_act_qdq(x_rot, 32)
y     = dequant(W_packed, W_scale) @ x_qᵀ # 复用现有 backend
```

### 需要新增的部分
1. **新 `QuantizationConfig`**（如 `hadamard_mxfp4`）：
   - `override_quantization_method`：检测 `quantization_config.rotation_config` 且
     `hadamard_type in {hadamard, random_hadamard}` 且 `data_type` 含 `mx_fp`。
   - 读 `block_size`（32）、`hadamard_type`。
2. **新 `LinearMethod`**（可继承现有 `SpinQuantMXFP4LinearMethod`）：
   - `create_weights`：除 `weight_packed/weight_scale` 外，注册 32×32 `hadamard_matrix` buffer。
   - `process_weights_after_loading`：
     - random：从 checkpoint 取一份 32×32（合并层 q/k/v 各有一份且相同，自定义 weight_loader
       忽略 shard、直接 copy 即可）。
     - deterministic：用 `get_hadamard_K(32)` 重建 Sylvester 矩阵（32=2⁵，纯 power-of-2），
       归一 `1/√32`。
     - 预计算 `R = Hᵀ`（random）或 `H`（deterministic，对称所以 `Hᵀ=H`）。
   - `_prepare_activations`：block-32 旋转 + act-qdq（复用现有逻辑，`rot_size=32`）。
3. **weight loading patch**：random 模型的 `hadamard_matrix` 是逐 Linear 的 buffer，
   vLLM 合并 q/k/v 时需要一个能忽略 shard、直接写同一 32×32 的 weight_loader（因为全同）。

### 与已迁移 SpinQuant plugin 的对比

| | SpinQuant R1/R4（已迁移）| per-Linear Hadamard（本文）|
|---|---|---|
| 旋转粒度 | 整个 hidden_size / intermediate_size 大旋转 | block-diagonal 32×32（=group_size）|
| 作用位置 | 仅特定点（残差流、down_proj 输入等）| **每个 Linear 的输入**|
| 矩阵数量/来源 | R1/R4 各一个，存 metadata + 可选矩阵 | 一个共享 32×32（random 存，det. 重建）|
| 合并层处理 | 选择性旋转需权重补偿 | **无需补偿**（输入侧统一旋转）|
| config key | `spinquant_config` | `rotation_config` |
| 复用程度 | — | 复用 ~90%（act-qdq / 反量化 / block 旋转）|

---

## 6. 待办 / 验证计划
- [x] 分析 per-linear / random hadamard 实现与数学等价性
- [x] 确认 random 模型确实存了 hadamard matrix（252 份，且全相同=共享随机 Hadamard）
- [x] 确认 HF backend 推理正常（252 个 forward pre-hook，输出连贯）
- [x] 实现 `hadamard_mxfp4` vLLM plugin（新 Config + LinearMethod + weight loader）
- [x] vLLM backend 推理跑通，并与 HF baseline 做 per-Linear 数值等价性验证（cos≈0.9999）

---

## 7. vLLM Plugin 实现结果（已完成）

新增文件（`vllm-qdq-plugin/src/vllm_qdq_plugin/`）：
- `rotation/perlinear_config.py` — `HadamardMXFP4Config`，注册名 `hadamard_mxfp4`，
  `override_quantization_method` 自动识别 `rotation_config.hadamard_type ∈ {hadamard, random_hadamard}`
  且 `data_type` 含 mxfp 的 checkpoint（并排除 `spinquant_config` 以免与 SpinQuant 插件冲突）。
- `rotation/perlinear_linear_method.py` — `HadamardMXFP4LinearMethod`：
  - `create_weights`：注册 `weight_packed/weight_scale` + 32×32 `hadamard_matrix` buffer
    （带自定义 `weight_loader`，合并层忽略 shard_id，因为各份相同）。
  - `process_weights_after_loading`：random 从 checkpoint 取 `H`，deterministic 用
    `deterministic_hadamard_matrix(32)` 重建；预计算 `R = Hᵀ`；MXFP4 权重预反量化为 bf16。
  - `apply`：`x' = blockdiag-matmul(x, Hᵀ)` → `hadamard_mxfp4_act_qdq` → `F.linear(x', W_bf16)`。
- `rotation/mxfp4.py` 新增 `mxfp4_act_qdq_hadamard` + 注册 op `hadamard_mxfp4_act_qdq`。
- `envs.py` 新增开关 `VLLM_HADAMARD_MXFP4`；`__init__.py` / `rotation/__init__.py` 接线
  `register_hadamard_mxfp4()`。

启用方式：`VLLM_HADAMARD_MXFP4=1`（自动识别 checkpoint，无需改 config）。

### 7.1 关键坑：激活 QDQ 必须对齐 **triton 融合核**，不是 `quant_mx`

排查等价性时发现：模型实际加载后 `pre_dequantized_input=True`，走的是
`quarot/utils/triton/mxfp4.py::mxfp4_forward_kernel`（融合了旋转+量化），
它与 `experimental/qmodules/mx.py` 里的 `_mx_qdq`→`quant_mx` **语义不同**：

| | `quant_mx`（非 triton 回退）| **triton 融合核**（实际推理路径）| Quark "even"（SpinQuant 用）|
|---|---|---|---|
| per-group scale | `2^(floor(log2(amax))−2)` | **`2^(floor(log2(amax))−2) / 0.75`** | round(amax→2^k) 后再取指数 |
| FP4 取整 | round-half-to-even | **显式中点阈值 round-to-nearest** | round-half-to-even |

三者都不同。逐元素比对实测：
- `quant_mx` / Quark-even 对齐 triton 核 → maxdiff ≈ 0.084（**错**，cos 仅 0.995）
- 复刻 triton 核（÷0.75 + 中点阈值）→ **maxdiff ≈ 6e-4，cos ≈ 0.9999995**（对）

所以 plugin 单独实现了 `mxfp4_act_qdq_hadamard`（带 `÷0.75` 和中点阈值），
**独立于** SpinQuant 路径的 `mxfp4_act_qdq`（Quark even），并在源码中加了等价性注释。

### 7.2 等价性验证（GPU7，`test_hadamard_equivalence.py`）

抓取 HF 真实 forward 的某个 Linear 原始输入与输出，用 plugin 的
`HadamardMXFP4LinearMethod.apply` 在同一输入上复算并比对：

| 模型 | cosine | rel mean err | 结果 |
|---|---|---|---|
| random_hadamard | 0.99991840 | 1.32e-2 | PASS |
| hadamard (det.) | 0.99993068 | 1.19e-2 | PASS |

> 残留 ~1% 的 rel-mean 来自：HF triton 核用 **bf16** `tl.dot` 做旋转，plugin 用 **fp32** 旋转
> （plugin 反而更准）；cosine 才是稳健指标。

### 7.3 端到端（`smoke_vllm_hadamard.py` / `compare_hf_vllm_hadamard.py`）

两个模型在 vLLM backend 都跑通（插件注册→自动识别→合并层加载→生成）。
HF 与 vLLM 贪心解码语义一致（Paris；red/blue/yellow；12×8=96），
token 级在 ~30 token 后分叉属正常（fp32 vs bf16 旋转精度在贪心解码上累积放大）。

### 7.4 本目录新增脚本
- `inspect_hadamard_model.py` — 检查 checkpoint 的 rotation_config / hadamard_matrix（落盘验证）
- `smoke_hf_hadamard.py` — HF backend 推理 smoke（验证 252 个旋转 hook）
- `test_hadamard_equivalence.py` — **per-Linear HF↔plugin 数值等价性 UT**
- `test_true_random_hadamard.py` — **真实 random（每个 Linear 不同矩阵）合并层等价性 UT**
- `smoke_vllm_hadamard.py` — vLLM backend 推理 smoke
- `compare_hf_vllm_hadamard.py` — HF vs vLLM 端到端生成对比

### 7.5 真实 random 场景：合并层的逐分区（per-partition）处理

**问题**：当前 auto-round 导出的 random 模型因 RNG 默认种子固定，252 份矩阵恰好全相同
（§2.60）。但**真实 random** 应允许每个 Linear（q/k/v/gate/up）拥有**不同**的随机
Hadamard。vLLM 会把 q/k/v 合并成 `qkv_proj`、gate/up 合并成 `gate_up_proj`。由于
Hadamard 作用在**输入（收缩）维**、且激活量化发生在旋转**之后**，"单一共享输入旋转"
只有在各分区矩阵相同时才正确；矩阵不同则数学上错误。

**对齐 auto-round HF 的事实**：HF 推理中 q/k/v 是**独立模块**，各有独立 pre-hook，做
各自的 `qdq(x @ Hᵢᵀ) @ Sᵢᵀ`。因此忠实实现就是按分区独立计算后拼接：

```
yᵢ = qdq(x @ Hᵢᵀ) @ Sᵢᵀ   →   y = concat([y_q, y_k, y_v], dim=-1)
```

**实现**（`perlinear_linear_method.py`）：
- `create_weights`：`hadamard_matrix` buffer 形状改为 `[num_partitions, bs, bs]`
  （`num_partitions = len(output_partition_sizes)`），每分区初始化为单位阵。
- 自定义 `_hadamard_weight_loader(param, w, loaded_shard_id)`：把 vLLM 的
  `loaded_shard_id`（QKV 为 `"q"/"k"/"v"`；MergedColumn 为 `0/1`；单层为 `None`）
  映射到分区下标，写入各自 slot —— **避免 last-shard-wins 覆盖**。
- `process_weights_after_loading`：逐分区求 `Rᵢ = Hᵢᵀ`，检测是否全相同：
  - **uniform 快路**（当前导出 / deterministic）：单旋转 + 单次合并 GEMM（原行为不变）。
  - **distinct 慢路**（真实 random）：按分区切 weight 行做 rotate→qdq→partial GEMM→concat。
- `apply`：按 `layer._rotation_uniform` 选择快/慢路。日志打印
  `per-partition=uniform(fast)` 或 `distinct xN(faithful)`。

**验证**（`test_true_random_hadamard.py`，GPU7，合成 MXFP4 权重 + QR 随机正交块）：

| 用例 | 分区 | distinct | uniform | cos | maxerr |
|---|---|---|---|---|---|
| qkv 真实 random | [128,64,64] | True | False | 0.99999994 | 0.000e+00 |
| gate_up distinct | [192,192] | True | False | 1.00000000 | 0.000e+00 |
| single | [160] | True | True | 1.0 | 0.000e+00 |
| qkv uniform 快路 | [128,64,64] | False | True | 1.0 | 0.000e+00 |

慢路输出与"3 个独立模块各自旋转后拼接"的参考**逐元素相等**（maxerr=0），证明
per-partition 路径精确复现 auto-round 分模块语义。两个真实 checkpoint 仍走 uniform
快路（§7.2 等价性 PASS、§7.3 端到端 PASS 不变）。

### 7.6 多卡支持（张量并行 TP / 流水并行 PP）

**结论：TP 与 PP 均支持。** 因为 block-Hadamard 是 32×32 块对角、作用在输入维，
天然与 vLLM 的权重切分兼容，无需额外跨卡通信。

**各层在 TP 下的正确性**：

| 层 | vLLM 切分 | rotation 是否成立 |
|---|---|---|
| `qkv_proj` / `gate_up_proj` | **Column**（切输出维），输入维完整 | ✅ 输入旋转作用在完整 hidden 上，各 rank 做同样旋转再 GEMM 自己的输出分片 |
| `o_proj` / `down_proj` | **Row**（切输入/收缩维），输入已分片 | ✅ 32 块对角旋转是**块内局部**的，每个 rank 只旋转自己那段完整的 32-blocks，无需跨卡通信 |

**关键实现点**：
- **旋转矩阵不分片**：`hadamard_matrix` 未设 `input_dim/output_dim`，只挂 `weight_loader`，
  因此每个 rank 都加载完整的同一份 `[num_partitions,32,32]`（复制而非切分）—— 正确。
- **权重正确分片**：`weight_packed`（`input_dim=1, output_dim=0`）按 rank 切，
  `weight_dense_qdq` 在 `process_weights_after_loading` 中由各 rank 自己的分片反量化。
- **per-partition 慢路（真实 random）**：`_partition_offsets` 由**每个 rank 的**
  `output_partition_sizes` 计算，切片偏移自动对齐到本 rank 的输出分片，distinct 路径
  在 TP 下同样正确。

**唯一约束**：RowParallel 层每卡输入维 `input_size_per_partition` 必须是 `block_size`(32)
的整数倍，否则 TP 切分会切断一个 32-block。
- 对 Qwen3-8B：`o_proj` 输入 4096、`down_proj` 输入 12288，对 2 的幂张量并行（TP≤128）
  都是 32 的倍数 → 始终满足。
- 一般地，只要 TP 是 2 的幂、且 hidden/intermediate 是 32 的常规倍数即恒成立。

**验证状态**：单卡（GPU7）等价性与端到端已 PASS；TP=2 端到端 smoke 待放开第二张卡后补测
（用 `smoke_vllm_hadamard.py` 加 `tensor_parallel_size=2` 即可）。

### 7.7 代码优化重构（SpinQuant + Hadamard 两个 plugin 共同）

在不改变行为的前提下做了以下重构/优化（重构后等价性与端到端测试指标与重构前**完全一致**）：

1. **抽取公共 helper** `rotation/_mxfp4_common.py`：`register_mxfp4_packed_weight`（打包权重/scale
   参数注册）、`dense_linear_stable_dtype`（dtype 稳定的 dense GEMM）、`clear_packed_storage`。
   `linear_method.py` 与 `perlinear_linear_method.py` 不再各写一份，消除漂移风险。
2. **抽取公共 config 基类** `rotation/_config_base.py::MXFP4RotationConfigBase`：收口
   `get_supported_act_dtypes / get_min_capability / get_config_filenames` 与 `data_type_is_mxfp`
   检测；`SpinQuantMXFP4Config`、`HadamardMXFP4Config` 共同继承。
3. **旋转矩阵 dtype 统一**：旋转矩阵改为按 `params_dtype`（激活 dtype）存储，热路径
   `R.to(x.dtype)` 在常见情形是 no-op（不再每 forward 复制）；SpinQuant 原先存 fp16、Hadamard
   存 fp32 的不一致一并消除。
4. **Hadamard 慢路（真实 random）预切片**：在 `process_weights_after_loading` 一次性把 dense weight
   切成 per-partition 连续视图（行切片零额外显存）并预存 bias 行偏移，`apply` 不再每 forward
   重复切片。
5. **`VLLM_SPINQUANT_MXFP4_QDQ_BACKEND` 接入生效**：原为死配置，现按值选择 `even`（默认，SpinQuant
   语义）/ `triton`（triton-match，高级覆盖）；默认行为不变。

回归验证：`test_true_random_hadamard.py`（4 例 maxerr=0）、`test_hadamard_equivalence.py`
（random cos 0.99991840、deterministic cos 0.99993068，与重构前一致）、
`tests/test_rotation_qdq_equivalence.py`（5 passed）、vLLM 端到端 smoke、SpinQuant 合成前向
（hadamard-runtime + matrix-runtime 均 finite）全部通过。

---

## 附：相关文件索引
- 配置/后端：`quarot/config.py`、`quarot/dispatcher.py`
- 变换/矩阵：`quarot/transforms.py`、`quarot/utils/math.py`、`quarot/utils/matrix.py`
- 标定融合 + 推理 hook：`quarot/apply.py`、`quarot/patch.py`、`algorithms/transforms/__init__.py`
- 推理 qmodule：`experimental/qmodules/mx.py`
- HF 入口：`inference/convert_model.py`
- 本目录脚本：`inspect_hadamard_model.py`（落盘检查）、`smoke_hf_hadamard.py`（HF 推理 smoke）
