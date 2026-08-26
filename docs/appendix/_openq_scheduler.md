# Open Questions — scheduler 文档

> 由 deep-dive/scheduler.md 撰写过程中标注的未决/需进一步追源码的问题。请勿直接修改 open-questions.md（避免并发冲突），本文件独立维护。

### DP attention + spec decoding 同时开启时 prefill/decode 混合的隔离细节

文档正文在 `python/sglang/srt/managers/scheduler.py:3108` 附近指出：`maybe_prepare_mlp_sync_batch` 通过 `need_mlp_sync` 确保 prefill 与 decode 批次不被混合（源码注释原文 "make sure prefill and decode batches will not be mixed when spec and dp-attn is enabled"）。但具体判定逻辑、跨 DP rank 的同步触发条件以及 `dp_attn_adapter.py` / `scheduler_pp_mixin.py` 内部实现尚未完整追到 worker 级别。需进一步阅读 `python/sglang/srt/managers/scheduler_components/dp_attn.py` 的 `maybe_prepare_mlp_sync_batch` 与 PP mixin 确认其确切行为，才能把"混合隔离"一节写扎实。

### retract_decode 的驱逐顺序与 radix 缓存联合优化

`python/sglang/srt/managers/schedule_batch.py:2867` 源码自带 `TODO(lsyin): improve retraction policy for radix cache`。当前 `_get_decode_retraction_order`（`schedule_batch.py:2856`）默认按 `(len(output_ids), -len(origin_input_ids))` 逆序保留"输出长、输入短"的请求，未联合考虑哪些请求在 radix 树中贡献了被他人共享的前缀（即驱逐它会导致更多重算）。在缓存命中率高的场景，该顺序可能次优。可作为后续优化方向记录，但当前文档仅描述现状、未做推断。
