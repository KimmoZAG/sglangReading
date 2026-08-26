### 并行采样（parallel sampling, n>1）与每请求单一 FSM 的交互行为

约束对象 `BaseGrammarObject` 在 `python/sglang/srt/model_executor/forward_batch_info.py:L777-L780` 中按**请求**展开（`batch.sampling_info.grammars = [req.grammar for req in batch.reqs]`），而 `update_regex_vocab_mask`（`python/sglang/srt/sampling/sampling_batch_info.py:L239-L264`）的 vocab_mask 是按 `batch_size`（采样行数）分配的。

可能的方向：
- 若并行采样 `n>1` 被允许且与 grammar 共存，同一请求的多行会共享同一 `BaseGrammarObject`；`accept_token` 只能接受最终被选中的那一个分支 token，其余并行分支的 FSM 状态无法独立维护，可能导致后续 mask 基于错误状态计算。
- 另一种可能是 SGLang 在更上层（请求校验 / 调度）已隐式禁止或限定了 grammar + parallel sampling 的组合，但本 commit 的 `SamplingParams.verify`（`sampling_params.py:L201-L210`）只校验四类约束互斥，并未对 `n>1` 与 grammar 同现做显式校验。

需要进一步确认：搜索 parallel_sample_num / n 采样路径是否对 `has_grammar` 请求有特殊处理，或实测构造 `json_schema + n=3` 请求观察行为。
