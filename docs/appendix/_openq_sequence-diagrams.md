### 被抢占（retract）请求的精确恢复语义

在 `ScheduleBatch.retract_decode` 中，超限请求通过 `release_req` 释放 KV（见 `python/sglang/srt/managers/schedule_batch.py:2897`，最终调用 `python/sglang/srt/utils/common.py` 的 `release_req`）。代码注释明确写 "release memory and don't insert into the tree because we need the space instantly"（见 `schedule_batch.py:2825`），即抢占时**不会**把 KV 写回 radix 树。

**可能的方向（待核实）**：
- `release_req` 是否对 `req.last_node` 执行 `tree_cache.dec_lock_ref`？若执行，则该请求在树中独有的后续节点被移除，恢复时只能复用与**其他仍在树中的请求共享**的公共前缀（通过下一轮 `init_next_round_input → match_prefix` 命中），其被抢占掉的独有 continuation 必须重新 prefill。
- 若抢占释放的 KV 段与树中某节点完全对应（因为 prefix 命中后该请求持有 `last_node` 锁），`dec_lock_ref` 可能触发节点删除，也可能仅减少引用计数。两种情形下"恢复时能否命中更长前缀"的行为不同。
- SWA（滑动窗口）场景下 `swa_reprefill_tail_tokens` 还会强制 re-prefill 尾部窗口（见 `schedule_batch.py:1336`、`radix_cache` 的 `swa_reprefill_tail_tokens`），这会在恢复时增大需要重新计算的长度。

建议在 `python/sglang/srt/utils/common.py` 中精读 `release_req` 与 `evict_from_tree_cache` 的具体实现后，补充恢复路径的确定性结论。
