# 服务器端启动工作提示词

你现在负责启动服务器端的 `Local LLM & Agent Learning Project`。当前日期为 2026-09-09。服务器具备 GPU，但服务器端工作尚未开始。你的第一阶段任务不是立即进行大规模训练，而是完成环境、代码、数据和短训练的可验证验收。

## 项目目标

当前项目处于 TinyLM 学习阶段，工作顺序必须保持为：

```text
代码同步
→ GPU 环境确认
→ PyTorch/CUDA 验证
→ 单元测试
→ Shakespeare 数据核对
→ 短训练冒烟测试
→ 记录实验结果
→ 再进行长时间训练
```

不要跳过测试，不要直接开始 5000 步或更长时间的训练。

## 重要边界

- 服务器不是本地电脑的替代品，而是负责 GPU 训练和长时间实验。
- 不修改项目总体架构，不提前开发复杂 Agent、多 Agent、RAG 或微调功能。
- 不执行删除整个目录、覆盖未知目录或修改系统配置的命令。
- 不把密码、Token、SSH 私钥、API Key 或其他敏感信息写入日志或提交到 Git。
- 不把 Checkpoint、原始语料、缓存和运行日志提交到 Git。
- 所有结论必须基于实际命令输出，不得假设 GPU、CUDA 或测试已经可用。

## 第一步：确认服务器位置

先输出并记录：

```bash
pwd
hostname
uname -a
whoami
git --version
conda --version
python --version
nvidia-smi
```

确认当前目录是项目目录。如果不是，请进入服务器上的项目目录。不要在不确定的目录执行批量创建、移动或删除操作。

## 第二步：获取最新代码

如果项目尚未下载：

```bash
git clone git@github.com:Crispy-xt/local_llm-agent_project.git
cd local_llm-agent_project
```

如果项目已经存在：

```bash
git status --short --branch
git fetch origin
git pull --ff-only origin main
```

确认当前代码来自远程 `main`，并记录：

```bash
git rev-parse HEAD
git log -1 --oneline
```

如果工作区存在未提交修改，不要覆盖它们；先报告状态并停止危险操作。

## 第三步：创建服务器 GPU 环境

本地的 `environment.yml` 是 CPU 配置，其中包含 `cpuonly`，不能直接用于服务器 GPU 训练。

先检查驱动和 CUDA 信息。然后根据服务器的驱动、CUDA 兼容性和管理员要求创建 GPU conda 环境。可以参考项目配置创建服务器版本，但必须删除 `cpuonly`，并安装与服务器兼容的 GPU 版 PyTorch。

环境名称建议保持：

```text
local-llm-agent
```

环境创建完成后执行：

```bash
conda run -n local-llm-agent python -c "import sys, torch; print('python:', sys.version); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_version:', torch.version.cuda); print('device_count:', torch.cuda.device_count()); print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

只有当 `cuda_available` 为 `True` 时，才可以声称服务器已经准备好 GPU 训练。

## 第四步：安装和验证项目

在项目根目录执行：

```bash
conda run -n local-llm-agent python -m pip install -e .
conda run -n local-llm-agent pytest -q
conda run -n local-llm-agent python -m compileall -q tiny_lm tests
```

当前基线应至少通过 15 项测试。如果测试数量或结果不同，必须报告实际结果，不得写成“测试通过”。

如果依赖安装失败：

1. 记录完整错误的关键部分；
2. 记录 Python、PyTorch、CUDA 和操作系统信息；
3. 不随意升级或降级一批依赖；
4. 不修改项目代码来掩盖环境问题。

## 第五步：获取并核对 Shakespeare 数据

原始语料不通过 Git 同步，需要单独放置到：

```text
data/raw/tinyshakespeare.txt
```

如果服务器允许联网，可以从项目 `data/README.md` 记录的来源重新下载；如果不能联网，则从可信存储复制。下载或复制后执行：

```bash
sha256sum data/raw/tinyshakespeare.txt
wc -c data/raw/tinyshakespeare.txt
```

Windows 环境可使用：

```powershell
Get-FileHash data/raw/tinyshakespeare.txt -Algorithm SHA256
(Get-Item data/raw/tinyshakespeare.txt).Length
```

必须核对以下 SHA-256：

```text
86C4E6AA9DB7C042EC79F339DCB96D42B0075E16B8FC2E86BF0CA57E2DC565ED
```

如果哈希不一致：

- 不开始训练；
- 检查下载是否完整；
- 检查编码和文件是否被修改；
- 报告实际哈希值。

## 第六步：短训练冒烟测试

先使用当前配置运行短训练：

```bash
conda run -n local-llm-agent python -m tiny_lm.train --config configs/tiny_lm.yaml
```

确认以下文件已经生成：

```text
checkpoints/tinyshakespeare_cpu.pt 或服务器配置指定的 Checkpoint
logs/*.csv
logs/*.svg
```

服务器上应使用 GPU 配置，不能把 `device: auto` 在 CUDA 不可用时误认为 GPU 训练。建议在正式训练前明确使用：

```yaml
device: cuda
```

第一次只运行短实验，确认：

- 程序能够识别 GPU；
- Loss 能够正常计算；
- 验证 Loss 能够生成；
- Checkpoint 能够保存；
- CSV 和 Loss 曲线能够生成；
- GPU 显存使用符合服务器限制。

## 第七步：汇报结果

完成启动验收后，必须按以下格式汇报：

```text
任务：服务器端 TinyLM 启动验收
日期：2026-09-09
服务器：
项目目录：
Git commit：
Python：
Conda：
PyTorch：
CUDA runtime：
GPU：
GPU 显存：
CUDA available：
数据文件：
数据 SHA-256：
测试命令：
测试结果：
编译检查：
训练配置：
训练命令：
训练步数：
初始 Train Loss：
最终 Train Loss：
初始 Validation Loss：
最终 Validation Loss：
训练耗时：
Checkpoint：
日志：
Loss 曲线：
问题：
下一步：
```

## 启动验收通过条件

只有以下条件全部满足，才可以进入长时间 GPU 训练：

```text
[ ] 代码来自远程 main 的明确 commit
[ ] 工作区没有未报告的修改
[ ] conda 环境创建成功
[ ] PyTorch 可以导入
[ ] torch.cuda.is_available() 为 True
[ ] GPU 型号可以读取
[ ] 所有现有测试通过
[ ] Python 编译检查通过
[ ] Shakespeare 文件存在
[ ] SHA-256 核对一致
[ ] 短训练成功
[ ] Checkpoint、CSV 和 SVG 已生成
[ ] 结果已按格式记录
```

完成以上验收后，暂停并等待下一步指令，不要自行启动更大模型、更大语料、微调或 Agent 实验。
