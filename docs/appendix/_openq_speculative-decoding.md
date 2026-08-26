# OPEN Questions: speculative-decoding

> 与 `docs/deep-dive/speculative-decoding.md` 配套。本文件记录源码阅读中无法在 SSOT 内确认、或实现分叉点较多的开放问题。

### 接受准则概率比的具体数学实现

接受/拒绝的精确判定（概率比 `min(1, q(x)/p(x))`、阈值 `speculative_accept_threshold_single/_acc` 在单步/累计模式下的具体作用）实现在 `sgl_kernel` 的 `verify_tree_greedy` / `tree_speculative_sampling_target_only` / `chain_speculative_sampling_triton` 等 CUDA/Triton kernel 内，这些源码不在本地 SSOT（`/home/kimmo/develop/sglang` Python 层）中。Python 侧仅能确认其输入契约与输出语义。建议结合 `sgl_kernel` 仓库逐行核对 kernel 内部公式。

### FROZEN_KV_MTP 在 scheduler 全流程的完成度

`SpeculativeAlgorithm.is_eagle()`（`python/sglang/srt/speculative/spec_info.py:97-104`）当前仍把 `FROZEN_KV_MTP` 包含在内，源码标注 `FIXME(kpham_sgl)` 待 scheduler 支持确立后移除。worker 创建时 FROZEN_KV_MTP 复用 EAGLE 路径，但是否在 scheduler 的草稿缓存分配、radix 协同等全流程完全支持尚未确认。建议结合 scheduler 侧 `_draft_extend_for_*` 与 `frozen_kv_mtp_worker_v2.py` 进一步核实。

### EAGLE3 aux hidden state 宽度推导的覆盖面

`get_draft_input_from_target_hidden_dim`（`python/sglang/srt/speculative/eagle_utils.py:442-481`）依赖 `hf_config.eagle_config` 的 `use_aux_hidden_state` / `num_aux_hidden_states` / `eagle_aux_hidden_state_layer_ids`。这些字段取值来自具体模型 config，SSOT 内无样例权重，宽度推导分支的覆盖面建议结合实际 EAGLE3 模型配置复核。
