# 数据目录说明

- `raw/`：保存原始数据，默认不提交到 Git。
- `processed/`：保存处理后的数据，默认不提交到 Git。
- 数据处理脚本应记录来源、版本、处理时间和复现命令。

## 当前数据

`raw/tinyshakespeare.txt` 是 Tiny Shakespeare 英文语料，用于第一个现实语料实验。

- 来源：Karpathy `char-rnn` 仓库中的 `data/tinyshakespeare/input.txt`
- 下载日期：2026-09-08
- 文件大小：1,115,394 bytes
- SHA-256：`86C4E6AA9DB7C042EC79F339DCB96D42B0075E16B8FC2E86BF0CA57E2DC565ED`
- 复现方式：从原始仓库下载对应 `input.txt`，并核对 SHA-256