# 服务器端 TinyLM 启动验收报告

日期：2026-09-09

## 1. 验收结论

服务器端 TinyLM 启动验收已完成。GPU 环境、PyTorch CUDA、项目测试、Python 编译检查、Shakespeare 数据和短训练冒烟测试均已实际验证通过。

当前暂不进入长时间训练，原因是工作区存在此前已有的未提交修改，需要先确认这些修改的处理方式。

## 2. 集群与代码状态

- 集群管理节点：`admin.cluster-bicmr.com`
- GPU 计算节点：`gpu01`
- GPU 分区：`gpu`
- 本次验证方式：通过 SLURM 申请短时 GPU 任务
- 项目目录：`/bicmr/home/crispyxt/LLM&Agent Project`
- Git 分支：`main`
- Git commit：`d253c69ac48f4eaebe0367c6b45eef3a4edd8e75`
- `HEAD` 与 `origin/main`：一致
- 长时间训练：未启动

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
?? SERVER_START_PROMPT.md
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

## 8. 当前验收清单

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
[ ] 进入长时间 GPU 训练
```

## 9. 下一步

等待本地确认工作区未提交修改的处理方式。确认后再决定是否：

1. 保留这些修改并建立服务器端正式 GPU 配置；
2. 创建独立的 `local-llm-agent` 环境；
3. 使用正式配置运行更长的 TinyLM 训练；
4. 记录正式实验结果。

在收到下一步指令前，不启动 5000 步或更长时间训练，也不开始更大模型、Agent、RAG 或微调实验。
