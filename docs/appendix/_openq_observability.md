### `SchedulerStats.token_usage` 命名误导的后续兼容性

`metrics_collector.py:78` 的源码注释明确标注 "FIXME: misleadingly named 'token_usage'; rename requires API deprecation"。该字段实际含义是 `max(full, swa, mamba)` 的瓶颈 KV 使用率，并非仅 full-attention 使用率。

可能的方向：
- 未来版本可能新增 `bottleneck_token_usage` 之类的准确命名，并将 `token_usage` 标记为 deprecated 后删除。
- 在重命名落地前，外部看板/告警若直接按字面理解 `token_usage`，可能误判 KV 池水位。
- 本文档基于 commit `e1c4db96` 仅记录现状，无法预判该重命名是否会进入公开指标 API，也无法确认是否会给用户带来破坏性变更。
