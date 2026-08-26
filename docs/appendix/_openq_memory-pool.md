# Open Questions — memory-pool

> 本文件由 memory-pool.md 文档化过程中遗留的未决问题汇总而成。请勿直接修改 `open-questions.md`（避免并发冲突），如需补充请追加到此文件。

### DCP 与 paged 同时启用时 free_pages 容量如何协调

`PagedTokenToKVPoolAllocator` 在非 DCP 路径下的 `page_size` 来自 `get_schedule().page_size`；但 `dcp_enabled` 时 `allocator.page_size` 会被放大为 `page_size * dcp_size`（见 `python/sglang/srt/mem_cache/kv_cache_builder.py` 中 `params.page_size` 的处理逻辑：`page_size if not dcp_enabled else token_to_kv_pool_allocator.page_size`）。两段代码在 DCP + paged 同时启用时如何协调 `free_pages` 的真实容量（`num_pages = size // page_size` 还是 `// (page_size * dcp_size)`），本文档尚未逐行验证。可能的方向：查阅 `MultiEndedAllocator` / `UnifiedMambaTokenToKVPoolAllocator` 等复合分配器，确认 DCP 维度的 token 编号空间是否已在 `size` 中预先乘过 `dcp_size`（构造 `PagedTokenToKVPoolAllocator` 时传入的 `size` 已经是 `sizes.max_total_num_tokens * get_parallel().attn_dcp_size`，见 `kv_cache_configurator.py` 约 L1718-L1720），若是，则 `num_pages = size // page_size` 中的 `page_size` 实际已是 `page_size * dcp_size`，二者自洽。

### evict_from_tree_cache 的驱逐策略与 alloc_extend 仍返回 None 的边界

`mem_cache/allocation.py` 的 `alloc_extend` 调用前会先 `evict_from_tree_cache(tree_cache, num_tokens)`（高估预算：`extend_num_tokens + len(seq_lens_cpu) * page_size`），但文档仅在调用层面确认，未深入 `evict_from_tree_cache` 内部实现。需明确：驱逐遵循的具体策略（LRU / 按 `radix_eviction_policy` server_args 选择）、是否存在“可驱逐额度上限”、以及为何在驱逐之后 `alloc_extend` 仍可能返回 `None`（例如前缀锁 `lock`、session cache、或正在被其它 batch 引用的节点不可驱逐）。建议下一步阅读 `mem_cache/radix_cache.py` / `unified_radix_cache.py` 中的 `evict` / `evictable_size` 相关实现。
