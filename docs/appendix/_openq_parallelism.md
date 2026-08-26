# OPEN QUESTIONS: parallelism

### EPLB 重平衡与 radix 缓存失效的耦合点
在 `python/sglang/srt/eplb/eplb_manager.py` 的 `EPLBManager.rebalance` /
`update_expert_location_with_recovery` 中，专家布局（`physical_to_logical_map`）变更后会把
缺失权重回灌到对应 rank（见 L302-L353）。但本次阅读未确认：当专家物理位置因 EPLB 改变时，
是否存在显式的 radix 缓存（前缀缓存）失效调用？可能的方向：
- 在 `ExpertLocationUpdater.update` 的调用方（scheduler / model_runner）中是否触发了
  `RadixCache` 的 `cache_finished_req` / 全局失效；
- 是否依赖「EPLB 仅在空闲窗口 / prefill 阶段触发」来规避缓存一致性问题；
- `ep_dispatch_algorithm="lp"` 的 LPLB 路径与 radix 命中之间的交互。
需要追踪 `ExpertLocationUpdater` 的调用链及 scheduler 的缓存失效逻辑以给出确定结论。
