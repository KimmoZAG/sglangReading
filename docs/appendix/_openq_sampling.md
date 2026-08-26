# OPEN QUESTIONS: sampling

> 本文件记录 `deep-dive/sampling.md` 写作过程中未能从源码确证、或存在多种解释的问题。正文对应处已用 `**[OPEN]**` 标注。请勿直接编辑 `open-questions.md`（并发冲突）。

### SamplingParams.n 是否真正生效？

`SamplingParams` 定义了 `n: int = 1`（`sampling_params.py:70`），语义上应为「单请求产出 n 条序列」。但在 `sampling` 模块（`SamplingBatchInfo`、`Sampler`、各 penalizer）内均未看到按 `n` 沿 batch 维展开的逻辑——`from_schedule_batch` 直接 `len(batch.reqs)` 个请求。需确证：`n>1` 是在 scheduler 层（如 `Req` 克隆 / `ScheduleBatch` 展开）处理，还是当前引擎根本不支持。若由 scheduler 展开，则在采样侧它表现为「多个独立请求」，penalty 状态彼此不共享，这与 vLLM 的 `n` 语义可能不同。

### overlap 模式下 penalty 张量的使用边界

`apply_logits_bias`（`sampling_batch_info.py:283-L300`）同时使用了 overlap 模式专用缓冲 `acc_additive_penalties` / `acc_scaling_penalties`（由 `update_penalties()` 填充，`sampling_batch_info.py:266-L281`）与非 overlap 模式的 `penalizer_orchestrator.apply`。源码注释区分了 "Used in the overlap mode" 与 "Used in the non-overlap mode"，但并未在同一函数内给出「两者互斥 / 择一」的显式判断。需确认：`ModelRunner` 在 overlap（chunked overlap / double-batch）调度下是否仅依赖 `copy_for_forward()` 预计算的 `acc_*` 张量、而把 `penalizer_orchestrator` 置空（见 `copy_for_forward` 的 `dataclasses.replace(self, penalizer_orchestrator=None)`，`sampling_batch_info.py:453-L456`）。若如此，则 overlap 路径下 `penalizer_orchestrator.apply` 分支因 orchestrator 为 None 而被跳过——这一点需要在 `model_runner.py` 的 overlap 调度路径中二次确认。

### custom_logit_processor 与 grammar 的优先级

`apply_logits_bias` 中顺序是：additive penalty → scaling penalty → `penalizer_orchestrator.apply` → `grammar_mask.apply` → `logit_bias.add_`。但自定义 logit processor 是在更早的 `_preprocess_logits`（`sampler.py:88-L95`）中、于 `apply_logits_bias` 之前施加。若某自定义 processor（如 `ThinkingBudgetLogitProcessor`）把某 token 抬到 0 而 grammar 同时将其置 `-inf`，最终 `-inf` 胜出。需确证：当自定义 processor 与 grammar 约束冲突时，引擎的「约束优先」是否有意设计，还是仅由执行顺序（processor 先、grammar 后）隐式决定。
