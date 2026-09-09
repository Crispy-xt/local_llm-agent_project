# Local LLM & Agent Learning Project

本项目用于系统学习本地大语言模型（LLM）和 Agent 的实现。项目分为两条相互衔接、但职责独立的路线：

1. 从零实现一个可训练、可保存、可生成文本的教学型小语言模型 TinyLM。
2. 使用本地已有 LLM 构建具备工具调用、状态管理、权限控制、日志和评测能力的 Agent。

核心原则：**先理解模型，再构建 Agent；先实现最小闭环，再逐步增加复杂度。**

## 1. LLM 与 Agent 的边界

### LLM 训练

LLM 训练解决模型内部机制：

```text
文本数据 → Tokenizer → Dataset → Transformer → Loss
→ 反向传播 → 参数更新 → Checkpoint → 文本生成
```

重点包括 Tokenizer、语言建模目标、Attention、因果掩码、训练循环、优化器、Checkpoint 和生成策略。

### Agent 构建

Agent 构建解决模型如何完成外部任务：

```text
用户任务 → LLM 决策 → 工具调用 → Runtime 执行
→ 结果返回 → 状态更新 → 继续执行或结束
```

重点包括 Prompt、工具协议、Agent Loop、状态、记忆、上下文、权限、沙箱、超时、日志和评测。

> Agent 不等于重新训练一个大语言模型。第一版 Agent 应使用已经训练好的本地模型，通过工具和运行时完成任务。

## 2. 推荐学习顺序

```text
字符级 TinyLM
  → BPE / SentencePiece Tokenizer
  → 本地现成模型推理
  → 单工具 Agent
  → 多工具 Agent
  → 代码仓库 Agent
  → 上下文与长期记忆
  → 自动化评测
  → SFT / LoRA / QLoRA 微调实验
```

不要在 TinyLM 尚未完成基本测试前进入复杂的多 Agent、RAG 或微调工作。

## 3. 推荐项目结构

```text
local-llm-agent/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/                 # 模型和 Agent 配置
├── data/raw/                # 原始数据，不删除
├── data/processed/          # 处理后数据
├── checkpoints/             # 模型 Checkpoint
├── logs/                    # 完整运行日志
├── experiments/             # 训练和推理实验记录
├── tiny_lm/                 # 模型、训练、生成
├── inference/               # 本地现成模型推理封装
├── agent/                   # Agent Loop、状态、上下文、记忆
├── tools/                   # 工具实现与注册
├── harness/                 # Runtime、沙箱、权限、限制、追踪
├── evals/                   # 任务集、评测运行器、指标
└── tests/                   # 自动化测试
```

职责边界：

| 目录 | 职责 |
|---|---|
| `tiny_lm/` | 模型、训练、生成和 Checkpoint |
| `inference/` | 加载和调用本地现成模型 |
| `agent/` | 决策循环、状态、上下文和记忆 |
| `tools/` | 对外提供能力并校验参数 |
| `harness/` | 隔离、权限、超时、资源限制和日志 |
| `evals/` | 通过客观指标判断任务是否完成 |
| `tests/` | 自动化测试 |

## 4. 环境准备

建议使用 Anaconda/Miniconda 管理环境，项目提供 environment.yml 作为可复现配置。当前本地采用 Python 3.11 和 CPU 版 PyTorch；服务器环境应另行创建 GPU 版 PyTorch 环境。

``powershell
conda env create -f environment.yml
conda activate local-llm-agent
``

基础依赖包括：

``text
PyTorch, pytest, PyYAML, numpy
``

进入本地模型和 Agent 阶段后，再按需安装 	ransformers、ccelerate、safetensors、itsandbytes、llama-cpp-python 或其他框架。
没有 GPU 时仍可训练 TinyLM，但应降低 `batch_size`、`block_size`、`n_layer` 和 `n_embd`，不要直接尝试训练十亿级以上模型。

## 5. 阶段一：实现字符级 TinyLM

### 5.1 字符级 Tokenizer

实现以下接口：

```python
encode(text: str) -> list[int]
decode(ids: list[int]) -> str
```

至少支持建立字符表、字符与 ID 双向映射、未知字符处理、词表保存和加载。重点是理解文本如何转成 token ID，以及 token ID 如何还原成文本。

### 5.2 Dataset

采用自回归语言建模样本：

```python
x = sequence[i : i + block_size]
y = sequence[i + 1 : i + block_size + 1]
```

预期形状：

```text
x:      [batch_size, sequence_length]
y:      [batch_size, sequence_length]
logits: [batch_size, sequence_length, vocab_size]
```

### 5.3 初始配置

```yaml
vocab_size: auto
block_size: 128
n_layer: 4
n_head: 4
n_embd: 128
dropout: 0.0
learning_rate: 0.0003
batch_size: 32
max_steps: 5000
```

### 5.4 模型组件

至少实现 Token Embedding、Position Embedding 或 RoPE、Causal Self-Attention、Multi-Head Attention、Feed Forward Network、Residual Connection、LayerNorm 或 RMSNorm，以及 Language Model Head。

必须保证当前位置不能读取未来 token。

### 5.5 训练和 Checkpoint

训练循环应包含：

```text
读取 batch → 前向传播 → Cross Entropy Loss
→ 清空梯度 → 反向传播 → optimizer.step()
→ 记录 loss → 定期保存 Checkpoint
```

Checkpoint 至少保存：

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "step": ...,
    "config": ...,
    "tokenizer": ...,
}
```

推荐使用以下文本进行过拟合验证：

```text
abcabcabcabcabcabcabcabc
```

如果 loss 不下降，优先检查输入与目标是否错位、causal mask、loss 维度、优化器参数、训练模式、学习率和 token ID 范围。

## 6. 阶段一验收标准

进入下一阶段前必须满足：

1. Tokenizer 能完成 `encode/decode`；
2. Dataset 能生成正确的 `x/y` 样本；
3. 模型输出 shape 正确；
4. Causal mask 能阻止未来信息泄漏；
5. 小数据集上的 loss 明显下降；
6. 模型能过拟合重复文本；
7. Checkpoint 可以保存、加载并继续训练；
8. 模型可以生成文本；
9. 相关单元测试全部通过。

至少覆盖 Tokenizer、未知字符、Attention shape、causal mask、模型 forward、loss 和 Checkpoint 恢复测试。

## 7. 阶段二：更接近实际的 Tokenizer

字符级模型通过后，再尝试 BPE 或 SentencePiece，并记录文本长度、token 数量、平均 token/字符比例和不同 Tokenizer 的切分结果。

注意：token 不一定等于一个字或一个词，不同模型的 Tokenizer 通常不能混用。Tokenizer 变更后必须重新训练，或使用与模型匹配的词表。

## 8. 阶段三：运行本地现成 LLM

TinyLM 用于理解内部机制；实际 Agent 应优先使用已经训练好的本地模型。可封装 Transformers、Ollama、llama.cpp 或 vLLM，第一版优先考虑 Transformers 或 Ollama。

统一推理接口，避免将模型加载细节泄漏到 Agent Loop：

```python
class LocalLLM:
    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        tools: list[dict] | None = None,
    ) -> str:
        ...
```

每次推理记录模型名称、输入和输出 token 数、首 token 延迟、总延迟、tokens/second、显存、生成参数和是否截断。

## 9. 阶段四：实现最小 Agent

第一版只接入 `calculator` 工具。模型响应应能区分工具调用和最终回答：

```json
{
  "type": "tool_call",
  "tool": "calculator",
  "arguments": {"expression": "12 * 8"}
}
```

```json
{
  "type": "final",
  "content": "计算结果是 96。"
}
```

基本循环：

```python
for step in range(max_steps):
    response = llm.generate(messages, tools=tool_schemas)
    decision = parse_response(response)

    if decision.type == "final":
        return decision.content

    if decision.type == "tool_call":
        validate_tool(decision)
        result = execute_tool(decision)
        append_tool_result(messages, decision, result)

return "任务未在限定步骤内完成"
```

必须具备最大步数、工具名称校验、参数 schema 校验、异常捕获、超时、未知响应处理、重复调用检测、日志和明确终止条件。

状态不要只依赖消息列表，建议独立维护：

```python
class AgentState:
    task: str
    messages: list
    summary: dict
    facts: list
    tool_calls: list
    step: int
    status: str
    errors: list
```

## 10. 阶段五：代码仓库工具

建议依次增加：

```text
list_files → search_files → read_file → run_tests → git_diff → apply_patch
```

文件操作必须经过：

```text
规范化路径 → 验证位于 workspace 内
→ 检查敏感路径和符号链接 → 执行操作
```

应防止 `../` 越权、绝对路径访问敏感目录、符号链接越权、读取密钥文件和环境变量。

命令工具必须限制允许的命令、工作目录、执行时间、输出大小、环境变量和网络权限，不要默认开放任意 shell。修改文件时保存修改前后 diff，并以实际工具结果而不是 Agent 的文字说明判断是否成功。

## 11. 上下文、记忆与原始日志

上下文采用分层设计：

```text
最近消息 + 当前任务状态 + 关键约束 + 相关检索结果
+ 必要工具输出 + 完整原始日志引用
```

长期保存用户约束、项目配置、技术决策、已确认事实、失败测试、修改文件和待办事项。可以压缩重复工具输出、已解决讨论、冗长解释、无关日志和已归档消息。

压缩摘要至少包含：

```json
{
  "goal": "...",
  "constraints": [],
  "confirmed_facts": [],
  "decisions": [],
  "completed_steps": [],
  "failed_steps": [],
  "open_questions": [],
  "next_action": "..."
}
```

摘要不能替代原始日志。完整日志应保存在 `logs/`，重要数字、路径、错误信息和限制条件必须优先保留。

## 12. Agent 训练的正确理解

“Agent 训练”分为三个层次：

### 第一层：Prompt 与工具协议

通过 Prompt、工具 schema 和 Agent Loop，让现成模型学会何时调用工具、如何生成参数、如何读取结果以及何时结束。这属于系统编排，不是参数训练。

### 第二层：任务数据与评测

收集带有预期工具和成功条件的固定任务，评估工具选择正确率、参数正确率、任务完成率、测试通过率、平均调用次数、执行时间和越权次数。

### 第三层：模型微调

只有 Prompt、工具协议、任务数据和 baseline 评测稳定后，才考虑 SFT、LoRA、QLoRA 或 DPO。微调不能替代权限控制、参数校验、沙箱、超时和评测。

## 13. 安全要求

任何可以读写文件、执行命令或访问网络的 Agent，都必须实现：

```text
workspace 隔离
路径校验
命令白名单
超时限制
输出长度限制
网络权限限制
敏感文件过滤
操作日志
人工确认机制
```

高风险操作必须人工确认，包括删除文件、大范围修改、数据库写入、外部请求、安装依赖、发布代码和修改系统配置。

> 安全规则不能只放在系统 Prompt 中，必须在 Tool 和 Runtime 层强制执行。

## 14. 测试与评测

### TinyLM

记录训练步数、训练/验证 loss、参数量、batch size、序列长度、训练耗时、GPU 信息和生成样例。

### Agent

第一批任务应简单、客观、可自动检查：

```text
计算表达式
列出目录文件
搜索关键词
读取文件
运行测试
查看 git diff
修复一个小型测试失败
```

每个任务必须定义输入、允许工具、预期行为、成功条件、最大步数和最大执行时间。

## 15. 推荐里程碑

| 里程碑 | 完成内容 |
|---|---|
| M1 | 字符级 Tokenizer、Dataset、Transformer、训练、Checkpoint、生成 |
| M2 | causal mask、shape、过拟合、loss 和恢复测试 |
| M3 | 本地现成模型加载、统一推理接口和性能记录 |
| M4 | calculator、工具 schema、Agent Loop、异常处理和日志 |
| M5 | 文件搜索、读取、测试、diff、补丁和 workspace 隔离 |
| M6 | 结构化状态、上下文截断、摘要、检索和固定评测 |
| M7 | 在已有 baseline 上进行 SFT/LoRA/QLoRA 对比实验 |

## 16. 每次开发任务规范

```text
1. 检查 git status
2. 阅读相关源码和测试
3. 明确输入、输出和错误行为
4. 做最小范围修改
5. 添加或更新测试
6. 运行相关测试和完整测试
7. 查看 git diff
8. 记录实验结果并更新文档
```

不要直接重写整个项目，不要使用 Prompt 代替权限控制，不要默认开放 shell，不要无限追加历史，也不要只根据最终文字判断任务成功。

## 17. 项目级编码 Agent 提示词

```text
你正在协助开发一个本地 LLM 和 Agent 学习项目。

项目目标：
1. 实现可训练、可保存、可生成文本的教学型 TinyLM。
2. 使用本地现成 LLM 构建工具调用 Agent。
3. 实现状态管理、上下文管理、权限控制、日志和自动化评测。

工作要求：
1. 修改前读取仓库结构、相关源码、测试和 README，并检查 git status。
2. 遵循现有命名和架构，不做无关重构。
3. 保持 tiny_lm、inference、agent、tools、harness 和 evals 的职责分离。
4. 为张量、工具参数和状态结构使用明确类型或 schema。
5. 权限必须由 Tool 和 Runtime 强制执行，不能依赖 Prompt。
6. Agent 必须具备最大步数、超时、异常处理和终止条件。
7. 保留完整原始日志，摘要只用于上下文优化。
8. 训练记录配置、随机种子、loss 和 Checkpoint；推理记录 token 数、耗时和错误。
9. 每次工具调用记录工具名、参数、结果和耗时。
10. 新增行为必须有对应测试，先运行相关测试，再运行完整测试。
11. 测试无法运行时必须说明原因，不得假设测试通过。

完成后汇报：修改的文件及原因、实现的行为、运行的命令、测试结果和已知限制。
```

## 18. 第一个实际任务

项目启动后的第一项任务限定为：

> 实现 `tiny_lm/` 中的字符级 Tokenizer、Dataset、Causal Self-Attention、TinyGPT、训练循环、Checkpoint 和生成脚本，并为 tokenizer、causal mask、模型 shape、loss 下降和 Checkpoint 恢复添加测试。

验收条件：字符词表可建立、`encode/decode` 可逆、Dataset 可生成 `x/y`、Transformer shape 正确、causal mask 阻止未来信息、loss 下降、模型可过拟合重复文本、Checkpoint 可保存和加载、模型可生成文本、相关测试全部通过。

完成这一阶段前，不进入复杂 Agent 开发。

## 19. 实验记录

每次训练或推理实验至少保存：

```text
实验时间
代码版本
模型配置
Tokenizer
数据集
训练步数
训练 loss / 验证 loss
GPU 与显存
训练或推理耗时
生成参数与结果
已知问题
复现命令
```

建议命名为：

```text
experiments/YYYY-MM-DD-tinylm.json
experiments/YYYY-MM-DD-local-agent.json
```

每次实验都应能够回答：改了什么、为什么改、指标是否改善、是否引入新问题、如何复现。

## 20. 最终目标架构

```text
用户
  ↓
Agent API
  ↓
Agent Loop
  ├── Context Manager
  ├── Memory Manager
  ├── Tool Registry
  ├── Policy Checker
  ├── Local LLM
  └── Evaluator
          ↓
      Harness / Runtime
          ↓
      文件、测试、数据库、网络等工具
```

模块职责：

- `Local LLM`：理解任务并生成决策或回答；
- `Agent Loop`：控制多轮决策流程；
- `Context Manager`：控制发送给模型的上下文；
- `Memory Manager`：保存、检索和更新长期信息；
- `Tool Registry`：管理工具与参数 schema；
- `Policy Checker`：检查工具是否有权限执行；
- `Harness`：提供隔离、超时、日志和资源限制；
- `Evaluator`：通过客观指标判断任务是否真正完成。

开发时始终围绕以下闭环检查：

```text
模型产生了什么？
系统解析了什么？
工具实际执行了什么？
状态如何变化？
为什么继续或停止？
任务是否客观完成？
```

## 21. 配置驱动训练

当前 TinyLM 已支持通过 YAML 配置运行正式训练流程：

```powershell
conda activate local-llm-agent
python -m tiny_lm.train --config configs/tiny_lm.yaml
```

训练配置支持：

- `data_path`：UTF-8 文本语料路径；
- `block_size`、`batch_size`：序列长度和批次大小；
- `n_layer`、`n_head`、`n_embd`：模型规模；
- `learning_rate`、`max_steps`：优化设置；
- `validation_fraction`：验证集比例；
- `eval_interval`、`eval_batches`：验证频率和验证批次数；
- `device: auto`：有 CUDA 时使用 GPU，否则使用 CPU；
- `checkpoint_path`、`log_path`、`curve_path`：输出文件路径。

训练完成后生成：

```text
checkpoints/tiny_lm.pt   # 模型、优化器、配置、Tokenizer 和实验元数据
logs/tiny_lm.csv         # step、训练 Loss、验证 Loss、学习率和耗时
logs/tiny_lm_loss.svg    # 可直接用浏览器打开的 Loss 曲线
```

当前默认配置使用 `data/raw/tinyshakespeare.txt` 作为第一个现实语料实验。`data/demo.txt` 仍可用于快速冒烟测试。进入更正式的训练阶段时，应继续记录语料来源、哈希和复现命令，并在 GPU 服务器上运行长时间训练。