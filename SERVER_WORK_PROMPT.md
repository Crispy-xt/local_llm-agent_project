# 服务器工作提示词

你正在服务器上协助开发和运行 `Local LLM & Agent Learning Project`。服务器具备 GPU，主要负责较长时间的 TinyLM 训练、现实语料实验、Checkpoint 保存、性能记录和后续本地 LLM/Agent 实验；本地电脑主要负责代码编写、阅读、单元测试和小规模调试。

## 当前项目目标

项目按以下顺序推进：

```text
字符级 TinyLM
→ Shakespeare 现实语料
→ BPE / SentencePiece
→ 本地现成 LLM 推理
→ 最小 Agent
→ 代码仓库 Agent
→ 标准评测和微调实验
```

不要跳过 TinyLM 的基础验证，也不要在当前阶段直接构建复杂多 Agent 系统。

## 获取项目

服务器不要复制整个本地目录，也不要上传 `.git` 之外的临时环境。推荐使用远程 Git 仓库：

```bash
git clone <repository-url>
cd "LLM&Agent Project"
```

如果仓库已经存在：

```bash
git pull --ff-only
```

如果当前还没有远程仓库，应先在本地创建初始 commit 并配置 remote，再在服务器 clone。不要把模型权重、日志、缓存和虚拟环境提交到 Git。

## 哪些内容同步

### 通过 Git 同步

```text
README.md
SERVER_WORK_PROMPT.md
tiny_lm/
tests/
configs/
pyproject.toml
requirements.txt
environment.yml
data/README.md
experiments/run_cpu_experiments.py
```

### 不通过 Git 同步

```text
.venv/
checkpoints/
logs/
data/raw/
data/processed/
experiments/*.json
*.pt
*.pth
*.bin
*.safetensors
```

这些文件要么在服务器重新生成，要么通过对象存储、`scp` 或 `rsync` 单独传输，并记录校验和。

## 创建 GPU 环境

先检查服务器基础环境：

```bash
hostname
uname -a
nvidia-smi
python --version
conda --version
```

服务器不要直接使用本地 CPU 配置中的 `cpuonly`。建议复制 `environment.yml` 为服务器专用配置，删除 `cpuonly`，再根据服务器驱动和 CUDA 兼容情况安装 GPU 版 PyTorch。安装完成后必须验证：

```bash
conda run -n local-llm-agent python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

不要仅根据 `nvidia-smi` 判断 PyTorch 能使用 GPU；必须检查 `torch.cuda.is_available()`。

## 服务器首次验收

完成环境创建后，先运行：

```bash
conda run -n local-llm-agent pytest -q
conda run -n local-llm-agent python -m compileall -q tiny_lm tests
```

在测试通过前，不开始长时间训练。

## 获取语料

当前 Shakespeare 原始语料被 Git 忽略，服务器应单独下载或从可信存储复制。下载后必须：

1. 保存到 `data/raw/tinyshakespeare.txt`；
2. 使用 UTF-8 读取；
3. 计算 SHA-256；
4. 与 `data/README.md` 中记录的值核对；
5. 记录下载日期、来源和复现命令。

不要把未经确认来源的大规模语料直接用于训练。

## 第一轮 GPU 训练

先使用较小规模做冒烟测试：

```bash
conda run -n local-llm-agent python -m tiny_lm.train --config configs/tiny_lm.yaml
```

确认 Checkpoint、CSV 和 Loss 曲线都生成后，再创建服务器专用配置，例如：

```yaml
device: cuda
max_steps: 5000
checkpoint_path: checkpoints/tinyshakespeare_gpu.pt
log_path: logs/tinyshakespeare_gpu.csv
curve_path: logs/tinyshakespeare_gpu_loss.svg
```

训练过程中记录 GPU 型号和显存、PyTorch 版本、CUDA 可用性、模型配置、语料文件 SHA-256、训练步数、训练/验证 Loss、训练耗时、tokens/second 和 Checkpoint 路径。

## 训练纪律

每次实验只改变少数变量，并保留 baseline。不要同时改变数据、Tokenizer、模型规模、学习率和训练步数，否则无法解释结果。

推荐实验顺序：

```text
固定模型，增加训练步数
→ 固定训练步数，比较学习率
→ 比较模型规模
→ 比较 block_size
→ 引入 BPE / SentencePiece
```

训练 Loss 和验证 Loss 都下降，才说明当前实验有积极信号。若训练 Loss 下降而验证 Loss 上升，应检查过拟合、数据切分和训练配置。

## Checkpoint 和结果回传

训练完成后必须汇报：Git commit、环境和 GPU 信息、数据集来源与 SHA-256、完整配置、初始/最终/最佳训练和验证 Loss、训练时间和吞吐量、Checkpoint 文件大小和校验和、生成样例、已知问题和下一步建议。

建议只将源码、配置、CSV、SVG 和实验摘要同步回本地。大型 Checkpoint 单独传输，不要提交到 Git。

## 安全和资源边界

服务器上的 Agent 或实验代码必须遵守：

```text
不执行未经确认的删除命令
不覆盖未知目录
不把密钥写入日志
不默认开放任意网络访问
不使用超过分配额度的 GPU 资源
不启动未授权的长时间任务
```

长时间任务使用 `tmux`、作业调度器或服务器规定的任务系统，不要依赖 SSH 会话一直保持。

## 汇报格式

每次任务完成后按以下格式汇报：

```text
任务：
Git commit：
环境：
GPU：
数据：
配置：
运行命令：
结果：
生成样例：
产物：
测试：
问题：
下一步：
```

## 当前服务器第一项任务

服务器上的第一项工作不是直接进行大规模训练，而是：

```text
1. 获取最新 Git 代码
2. 创建 GPU conda 环境
3. 验证 CUDA 和 PyTorch
4. 运行 15 项单元测试
5. 核对 Shakespeare 数据 SHA-256
6. 用当前配置做短训练冒烟测试
7. 汇报环境、测试和产物
```

只有这项任务全部通过后，才开始 5000 步或更长时间的 GPU 训练。
