# Open Questions — open-questions（整合任务自身）

> 本文件由 open-questions.md 整合任务（最后一个子任务）自身产生的开放问题。
> 请勿直接修改 open-questions.md（避免并发冲突），新问题追加到此文件即可。

### 源 `_openq_*.md` 锚点的路径/行号是否需要统一复核

在整合各模块 `_openq_*.md` 时，经 `Read` / `Grep` 在 SSOT（commit `e1c4db96`）实测复核锚点，发现至少一处路径漂移：

- `_openq_observability.md` 将 `metrics_collector.py` 归于 `python/sglang/srt/managers/` 之下，但真实路径为 `python/sglang/srt/observability/metrics_collector.py`（行号 `77-L78` 一致）。open-questions.md 已采用修正后的真实路径。

**问题描述**：各 `_openq_*.md` 由不同子任务独立撰写，其锚点是否全部经统一复核、是否个别引用了早期 commit 或不同分支的文件/行号，目前无完整证据。直接引用这些锚点时，应以本仓库（对齐 commit）中 `Read` 实测结果为准。

**可能的验证方向**：
1. 对全部 27 个 `_openq_*.md` 中出现的"证据锚点"做一次脚本化批量复核（grep 锚点中的 file:line 是否真实存在于 SSOT 对应 commit），输出漂移清单。
2. 特别关注因目录重命名（如 `managers/` ↔ `observability/`、`srt/lang` 模块的实际包路径）导致的路径类漂移，以及因行号随 commit 推进而偏移的数值类漂移。
3. 确认 `python/sglang/lang/` 与 `python/sglang/srt/lang/` 是否为同一模块的不同引用（本任务实测 `fork` / `concatenate_and_append` 逻辑位于 `python/sglang/lang/interpreter.py`，无 `srt/` 前缀）。
