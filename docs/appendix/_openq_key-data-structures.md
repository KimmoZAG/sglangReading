# Open Questions: key-data-structures

### ForwardBatch 是否真的拥有 attn_backend_data / req_to_token_pool 字段？
任务书写作重点列出 `ForwardBatch` 含 `attn_backend_data`、`req_to_token_pool` 等字段。实读 `python/sglang/srt/model_executor/forward_batch_info.py:L411-L638` 的 `ForwardBatch` 定义，**未找到**这两个字段：
- `req_to_token_pool`：`ForwardBatch` 不持有内存池引用；它经 `model_runner` 间接访问（init_new 的第二个参数 `model_runner` 提供了池的入口）。可能历史上存在、或在某些分支中存在，需在 `ModelRunner` 中确认是否会把池以其它名字（如通过 `attn_backend` 或 `model_runner`）暴露给前向代码。
- `attn_backend_data`：注意力后端元数据是在 `ModelRunner._forward_raw` 内调用 `attn_backend.init_forward_metadata(fb)` 时由后端临时构造并挂到 attention backend 自身（或写到 `fb._attn_output` 等运行时字段），并非 `ForwardBatch` 的持久 dataclass 字段。需确认不同 attn backend（FlashInfer / Triton / Aiter）各自的元数据载体命名，以决定文档是否应改为"运行时由后端注入，非 FB 成员"。

可能的方向：以源码为准，文档正文已按"不存在这两个字段"描述；若后续发现命名差异，应补一节说明后端元数据生命周期。
