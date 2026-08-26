# OPEN Questions: add-a-model

### 权重加载 v1/v2 双路径的迁移状态
文档仅确认分流点在 `LlamaForCausalLM.load_weights`（`python/sglang/srt/models/llama.py:663-668`），由环境变量 `SGLANG_ENABLE_WEIGHT_LOADER_V2` 控制。
可能的方向：进一步阅读 `python/sglang/srt/models/utils.py` 与 `python/sglang/srt/model_loader/auto_loader.py`，
梳理 `AutoWeightsLoader`、`RemapRegistry`、`register_weight_remap` / `get_weight_remap` 的注册机制，以及 PR1 协议（issue #31051 / RFC #24703）
中 `load_weights(..., run_post_load=True)` 与 `post_load_weights(...)` 的拆分边界。需确认在哪些模型上 v2 已成为默认、v1 何时被移除。
