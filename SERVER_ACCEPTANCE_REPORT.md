# 服务器端 TinyLM 启动验收报告

日期：2026-09-09

## 1. 验收结论

服务器端 TinyLM 启动验收和第一次正式 GPU 训练均已完成。GPU 环境、PyTorch CUDA、项目测试、Python 编译检查、Shakespeare 数据、短训练冒烟测试和 5000 步正式训练均已实际验证。

正式训练使用 SLURM 在 `gpu01` 上运行并成功完成。后续训练前仍需注意工作区存在此前已有的未提交修改。

## 2. 集群与代码状态

- 集群管理节点：`admin.cluster-bicmr.com`
- GPU 计算节点：`gpu01`
- GPU 分区：`gpu`
- 本次验证方式：通过 SLURM 申请短时 GPU 任务
- 项目目录：`/bicmr/home/crispyxt/LLM&Agent Project`
- Git 分支：`main`
- Git commit：`a9e1c55de27da2a08dbb790dc2c13a220a6102b5`
- `HEAD` 与 `origin/main`：一致
- 长时间训练：已完成第一次 5000 步训练

## 3. 工作区状态

工作区存在未提交修改，未执行 `git pull` 或覆盖操作。当前状态如下：

```text
 M .gitignore
 M README.md
 M SERVER_WORK_PROMPT.md
 M environment.yml
 M pyproject.toml
 M requirements.txt
 M tests/README.md
 M tiny_lm/__init__.py
```

这些修改已在启动过程中报告并保留。验证过程中生成的 `local_llm_agent.egg-info/` 已清理。

## 4. GPU 与 Python 环境

GPU 计算节点上的实际验证结果：

```text
GPU: NVIDIA A100 80GB PCIe
GPU memory: 81920 MiB
PyTorch: 2.5.1+cu121
CUDA runtime: 12.1
cuda_available: True
device_count: 1
```

使用的已有 Conda 环境：

```text
Environment: mcpg_cuda12
Python: 3.12.12
Python path: /bicmr/home/crispyxt/.conda/envs/mcpg_cuda12/bin/python
```

本次未新建 `local-llm-agent` 环境，因为现有 `mcpg_cuda12` 环境已经具备可用的 GPU 版 PyTorch，并成功完成项目验证。

加载的集群模块：

```text
cuda/11.7
anaconda/3
```

PyTorch 自带 CUDA runtime 为 `12.1`，在集群驱动环境下实际识别并使用了 A100 GPU。

## 5. 项目安装、测试与编译

项目已在 `mcpg_cuda12` 环境中以 editable 方式安装：

```bash
python -m pip install -e .
```

测试依赖 `pytest` 已安装。测试命令：

```bash
python -m pytest -q
```

实际结果：

```text
15 passed in 15.90s
```

编译检查命令：

```bash
python -m compileall -q tiny_lm tests
```

实际结果：通过。

## 6. Shakespeare 数据核对

数据文件：

```text
data/raw/tinyshakespeare.txt
```

实际核对结果：

```text
文件大小：1,115,394 bytes
SHA-256：86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
```

与项目要求的 SHA-256 一致：

```text
86C4E6AA9DB7C042EC79F339DCB96D42B0075E16B8FC2E86BF0CA57E2DC565ED
```

## 7. GPU 短训练冒烟测试

为避免覆盖仓库中已有实验产物，本次使用了临时配置和临时输出目录。

训练配置：

```yaml
block_size: 128
n_layer: 2
n_head: 2
n_embd: 64
batch_size: 8
learning_rate: 0.0003
max_steps: 5
eval_interval: 2
eval_batches: 2
device: cuda
```

训练命令：

```bash
python -m tiny_lm.train --config <临时 GPU 配置>
```

训练实际结果：

| 项目 | 初始值 | 最终值 |
|---|---:|---:|
| Train Loss | 4.212871 | 3.989967 |
| Validation Loss | 4.192527 | 4.044074 |

- 训练步数：`5`
- 训练耗时：约 `6.33 秒`
- GPU：`NVIDIA A100 80GB PCIe`
- 训练状态：成功

生成文件：

```text
/tmp/tinylm-smoke.4i6oR2/tinyshakespeare_gpu.pt
/tmp/tinylm-smoke.4i6oR2/tinyshakespeare_gpu.csv
/tmp/tinylm-smoke.4i6oR2/tinyshakespeare_gpu_loss.svg
```

Checkpoint 已确认包含：

```text
config
cuda_random_state
metadata
model_state_dict
numpy_random_state
optimizer_state_dict
random_state
step
tokenizer
torch_random_state
```

## 8. 正式 GPU 训练

正式训练使用配置文件：

```text
configs/tiny_lm_gpu.yaml
```

关键配置：

```yaml
n_layer: 2
n_head: 2
n_embd: 128
block_size: 128
batch_size: 32
learning_rate: 0.0003
max_steps: 5000
eval_interval: 250
eval_batches: 20
device: cuda
```

SLURM 作业：

```text
Job ID: 1482635
Partition: gpu
Node: gpu01
GPU: 1
CPU: 8
Memory: 16G
Status: COMPLETED
Exit code: 0:0
Elapsed: 00:00:59
```

训练结果：

| 项目 | 初始值 | 最终值 |
|---|---:|---:|
| Train Loss | 4.171730 | 1.373604 |
| Validation Loss | 3.964478 | 1.393005 |

CSV 记录的最后几步如下：

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 4000 | 1.457341 | 1.459833 |
| 4250 | 1.445217 | 1.425229 |
| 4500 | 1.433453 | 1.411957 |
| 4750 | 1.409022 | 1.382069 |
| 5000 | 1.373604 | 1.393005 |

训练输出：

```text
checkpoints/tinyshakespeare_gpu.pt
logs/tinyshakespeare_gpu.csv
logs/tinyshakespeare_gpu_loss.svg
```

文件 SHA-256：

```text
c2f1b97d06a003d23de6f6f1e50e2be2b84a6bc2b1215e2059540c5d8f36d5f6  checkpoints/tinyshakespeare_gpu.pt
d4f8d039e1b8c34db5ac44ac660e431b75f7ecf31f8aaa1b50ba96e1c7415249  logs/tinyshakespeare_gpu.csv
8a8111e6663cd9c2ff6347e717eb9e988dfc402909717fdc16ee4b6dcf  logs/tinyshakespeare_gpu_loss.svg
```

Checkpoint 的 `step` 为 `4750`，这是因为当前训练实现只在验证 Loss 创新低时保存最佳模型。第 `4750` 步的 Validation Loss 为 `1.382069`，优于第 `5000` 步的 `1.393005`，因此 Checkpoint 保存的是最佳模型，而不是最后一步模型。

## 9. Checkpoint 生成结果

使用最佳 Checkpoint：

```text
checkpoints/tinyshakespeare_gpu.pt
```

生成参数：

```text
max_new_tokens: 220
随机种子：20260910、20260911、20260912
```

### 示例一

Prompt：`ROMEO:`
Temperature：`0.7`

```text
ROMEO:
That's the duke.

JOHN OF GAUNT:
I cannot contrary to your brother, your daughter:
He hands and he spoor ender and home Most my foe
The liban-saye want that my live,
And therefore the famedity a prayers
To fortune to an
```

### 示例二

Prompt：`HAMLET:`
Temperature：`0.8`

```text
HAMLET:
Madam, father, stay thee comes devil,
Take him yet? we content now. Hark: he bein mild
in hand, and the worlding ribunes sufference;
And me not thing scape storn; sirreturness,
That I not him to with a gone, and England
```

### 示例三

Prompt：`KING:`
Temperature：`0.9`

```text
KING:
I lord, they and with triumphan my brother,
And break for one any I will my sorry;
Here but I come; for my son, he worst commend
are it shall be much a traitor. Let's name,
Be shall be honest, the shall care, the sensho
```

### 生成质量初步评价

- 模型已经学习到 Shakespeare 语料的部分形式特征，包括角色名、冒号、戏剧对白分行和古英语词汇。
- 输出仍存在明显的句法不稳定、词语组合不连贯和长文本语义一致性不足。
- 部分词形是字符级模型根据局部上下文生成的近似组合。
- 当前结果说明训练和生成流程已经闭环，但模型容量与训练规模仍不足以生成高质量连贯文本。

## 10. 当前验收清单

```text
[x] GPU 计算节点可通过 SLURM 申请
[x] PyTorch 可以导入
[x] torch.cuda.is_available() 为 True
[x] GPU 型号可以读取
[x] 15 项现有测试全部通过
[x] Python 编译检查通过
[x] Shakespeare 文件存在
[x] SHA-256 核对一致
[x] 短训练成功
[x] Checkpoint 已生成
[x] CSV 已生成
[x] SVG Loss 曲线已生成
[x] 验收结果已记录
[ ] 工作区无未报告修改
[x] 第一次 5000 步 GPU 训练成功
```

## 11. 下一步

等待下一步指令。后续可以：

1. 进一步分析 `logs/tinyshakespeare_gpu_loss.svg`；
2. 根据 Loss 和生成结果决定是否调整学习率、模型规模或训练步数；
3. 在确认工作区修改处理方式后，再进行新的实验。

在收到下一步指令前，不自行启动新的长时间训练，也不开始更大模型、Agent、RAG 或微调实验。
