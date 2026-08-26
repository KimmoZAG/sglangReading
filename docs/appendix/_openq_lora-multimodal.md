# Open Questions: lora-multimodal

追踪 `lora-multimodal.md` 中尚未完全确证、待进一步源码验证的问题。不要直接修改 `open-questions.md`，避免并发冲突。

### RadixCache 对图像占位 token（词表外 hash）的确切处理

**描述**：多模态占位 token 的 id 由 `_compute_pad_value(hash)` 得到，落在模型词表之外（`schedule_batch.py:1983` 注释）。前缀树（RadixCache）按 token id 字面匹配，因此不同图像的占位 id 不同 → 天然前缀隔离；相同图像可共享 KV。但 KV 之外的视觉嵌入并不在 Radix 缓存中，`encoder_cached` 用 `len(req.prefix_indices) >= im.num_image_tokens` 来判定是否可跳过编码器（`schedule_batch.py:2252-2255`）。

**可能的方向**：
- 在 `mem_cache/radix_cache.py` 与 `scheduler.py` 的 `match_prefix` / `extend` 路径上确认：是否存在对 multimodal / image token 的特殊保护，确保图像 token 不会被 chunked-prefill 的前缀边界切断。
- 验证 `encoder_cached` 的 `prefix_indices >= num_image_tokens` 判定是否为“前缀命中且含完整图像”的唯一防线，是否存在其它兜底（如图像 token 始终置于 prompt 前缀、不可位于可切断区间）。
- 检查 `prepare_encoder_info_extend` 调用时机与 chunked prefill 分片策略之间的契约，确认跨分片时图像嵌入不丢失。
