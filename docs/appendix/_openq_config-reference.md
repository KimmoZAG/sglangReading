# config-reference 开放问题（Open Questions）

本文件记录撰写 `config-reference.md` 时未完全确认、但已在正文以 `> **[OPEN]**` 标注的问题。请勿直接修改 `open-questions.md`。

### --config 文件与命令行参数的合并优先级

- **描述**：`prepare_server_args`（`python/sglang/srt/server_args.py:9658-L9694`）在 `argv` 含 `--config` 时调用 `ConfigArgumentMerger(parser).merge_config_with_args(argv)`（`server_args.py:9673-L9679`）。文档中称"CLI 通常覆盖文件"，但并未展开 `ConfigArgumentMerger.merge_config_with_args` 的实现来确认最终优先级方向（是文件作为默认值、CLI 覆盖文件；还是 CLI 作为默认值、文件覆盖 CLI）。
- **可能的方向**：
  1. `ConfigArgumentMerger` 先将文件中的键值作为 argparse 的 defaults 注入，再让命令行 `argv` 覆盖（最常见模式，即"CLI 优先"）。
  2. 反向：命令行未显式给出的字段回退到文件值（本质同 1，但布尔 `store_true` 旗标的"未出现"判定需特殊处理，这也是源码注释提到"提取 boolean actions from the parser to handle them correctly"的原因）。
- **如何验证**：阅读 `python/sglang/srt/server_args_config_parser.py` 中 `ConfigArgumentMerger.merge_config_with_args` 的实现，确认文件值写入 `parser.set_defaults` 还是直接拼接到 `argv` 前。
