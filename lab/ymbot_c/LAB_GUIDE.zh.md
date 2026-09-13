# YMBOT-C 实验指南

**系统：** NULLXES MINAKANUSHI（简称 MINA）  
**交付包名：** YMBOT-C  
**实验轨：** L0 闭环 · L1 小型残差 · L2 仅 6.8B 推理  
**已发布 6.8B 权重状态：** research checkpoint，**未验收**

本目录就是给合作实验室拷走的内容。它不是聊天模型，不是 VLA，也不是 NULLXES 完整仓库。

语言只是一种模态。Token 不是认知单位。执行侧只接收 **ActionIntent**。禁止 PWM。

---

## 1. 你在测什么

```text
Observation → MinaUnit → world belief → uncertainty → futures
→ strategies → Constraint Kernel → ActionIntent → SyntheticWorld
```

必须复现：

| 门禁 | 通过条件 |
|---|---|
| 持续性 | 被遮挡的运动体仍留在 world belief，uncertainty 上升 |
| 时间顺序 | 先 A 后 B ≠ 先 B 后 A（速度符号相反） |
| 反事实 | WAIT 与 MOVE_TO 的终端状态不同 |
| 硬约束 | 高分但进入禁区的 MOVE_TO 被拒绝；策略不能选它 |
| 闭环 | Intent 改变下一次观测（或诚实 HOLD 原地不动） |
| 权限 | `policy_enabled=false` → SAFE_HOLD，世界信念仍更新 |
| 无 PWM | Intent JSON 中 `pwm: false`，没有电机占空比 |

训练 loss 下降 **不算** 通过。

---

## 2. 安装

Python **3.11+**。在本目录：

```text
python -m pip install -r requirements.txt
python run_lab.py l0 --selftest
python -m pytest tests -q
```

预期：selftest `"passed": true`，pytest 全绿。

```text
python run_lab.py l0 --steps 24 --seed 11 --report artifacts/l0.json
```

研究种子是 **11**。改种子后不要和 NULLXES 的数字比。

---

## 3. 三条实验轨

### L0 — 单文件，CPU

`mina_loop.py` 就是封好的闭环。笔记本即可，分钟级。

```text
python run_lab.py l0 --selftest
python run_lab.py l0 --policy-off --steps 16
```

`--policy-off`：认知开，自主选择关。Intent 必须是 `SAFE_HOLD`。

### L1 — 实验室 GPU 仪器（不是 6.8B）

在同一 SyntheticWorld 上训练极小的下一位置残差头。约束核仍是非学习的。**不会构建 6.8B。**

```text
python run_lab.py l1 --device cpu --steps 40 --report artifacts/l1.json
python run_lab.py l1 --device cuda --steps 40
```

通过：`ade_trained < ade_zero`，梯度有限，`constructs_6_8b: false`，参数量 ≪ 10k。

这是主仓库里 `gpu_train_v01`（6.2M 仪器）这一档，压缩到本目录能独立训练的程度。

### L2 — 仅 6.8B 推理

三件都要有：

1. NULLXES 提供的 partner wheel（可 `import minakanushi`）；
2. [MINAKANUSHI-6.8B](https://huggingface.co/MagistrTheOne/MINAKANUSHI-6.8B) 上的 `minakanushi_stage0_step1128.mina`；
3. **≥ 80 GB** 显存（见矩阵）。

```text
python run_lab.py l2 \
  --checkpoint /data/minakanushi_stage0_step1128.mina \
  --arch /opt/minakanushi/configs/architecture/minakanushi_6_8b.yaml \
  --report artifacts/l2.json
```

规范运行时是 `*.mina`。Hub 上的 safetensors 只是权重镜像。禁止当成 `AutoModelForCausalLM` 加载。

官方 heldout 包（不是这次冒烟加载）：[mina-6.8b-v03](https://huggingface.co/datasets/MagistrTheOne/mina-6.8b-v03)。

在 CPU 或 24 GB 卡上跑 L2，本工具会 **exit 2**。这是正确行为。

**不要用本工具训练 6.8B。** 训练拓扑是 2× H200 141 GB 或 1× B300。单张 H100 80 GB 会在 Adam 上 OOM。训练仍由 NULLXES 执行。

---

## 4. 显卡矩阵

| 卡 | 显存 | L0 | L1 | L2 推理 | 6.8B 训练 |
|---|---:|---|---|---|---|
| CPU / 笔记本 | — | 可以 | 可以 | **否** | **否** |
| RTX 4080 | 16 | 可以 | 勉强 | **否** | **否** |
| RTX 4090 | 24 | 可以 | 可以 | **否** | **否** |
| A100 40 GB | 40 | 可以 | 可以 | **否**（合同档） | **否** |
| A100 80 GB | 80 | 可以 | 可以 | 可以 | **否** |
| H100 80 GB | 80 | 可以 | 可以 | 可以 | **否**（优化器 OOM） |
| H800 80 GB | 80 | 可以 | 可以 | 可以 | **否** |
| RTX PRO 6000 BW | 96 | 可以 | 可以 | 可以 | **否** |
| H20 96 GB | 96 | 可以 | 可以 | CUDA 栈匹配则可以 | **否** |
| 2× H200 141 / 1× B300 | 141 / ~288 | — | — | — | 仅 NULLXES，不在本包 |
| 昇腾 910B / 其他 NPU | — | 仅 L0 | **合同外** | **否** | **否** |

对本指南而言，H800 与 H100 同档：可以推理，不能做 6.8B 训练。

本次评估不要把 Dynamic World Core 移植到 CANN。NPU 若不能走已发布的 PyTorch CUDA 路径，停在 L0/L1 并如实记录。不要为了迁卡改架构。

GPU 轨软件锁定：

```text
Python 3.11+
PyTorch 2.8 + CUDA 12.8   （2.3+ 可跑 L0/L1）
L2 使用 bf16
seed 11
```

---

## 5. 身份与许可

YMBOT-C 是交付包名。架构名是 **NULLXES MINAKANUSHI**。

合作方可按 `LICENSE_EVALUATION.md` 在实验室测试中使用 MINA。不得改名、不得包成聊天模型、不得把权重复发成别的机构的 checkpoint。

SelfModel 是护照状态（身份、具身、权限），不是 Transformer，也不是 “I AM MINA” 文本目标。

`policy_enabled=false`：大脑开着，自主选择关掉。

---

## 6. 回传给 NULLXES

```text
artifacts/l0.json          （selftest + 闭环）
artifacts/l1.json          （设备，零预测 vs 训练后 ADE）
artifacts/l2.json          （仅当 L2 真正加载成功）
nvidia-smi.txt             （GPU 名称 + 驱动）
torch_cuda.txt             （torch.__version__，torch.version.cuda）
```

不要回传模型的聊天记录。这里没有聊天。

---

## 7. 本次评估禁止

```text
不得改 latent_dim / core_depth / world_slots
不得替换 DWC
不得加 language head
不得把 authority 当成神经网络目标来训
不得调用 AutoModelForCausalLM
不得在 4090 / 6000 / 单卡 H100 上训 6.8B
ActionIntent ≠ PWM
```
