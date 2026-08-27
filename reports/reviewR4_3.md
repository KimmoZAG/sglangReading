# 评阅报告 R4-3

> 评阅对象：8 篇 deep-dive 文档（observability / parallelism / quantization / radix-cache / sampling / scheduler / server-entrypoint / speculative-decoding）
> SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）
> 评阅方法：对每个 `path:La-Lb` 锚点解析真实路径 → 提取文中紧邻引用的符号 → 用 grep 核验该符号真实定义/出现行是否落在 `[La,Lb]` 内 → 判定 OK / 漂移 / 伪造 / 占位；并对 mermaid 参与者/类名做 `class <名>`/`def <名>` 存在性核验；所有判定均经实测 grep，无凭记忆。
> 注：本轮在 R4-1/4-4 行号范围校验之上，重点做了**符号级**核验（行号在范围内 ≠ 符号匹配），故新增若干 `symbol_mismatch`/`mermaid_fake` 类严重问题。

## 总体结论

**评分：7.5 / 10**

8 篇文档整体质量高，绝大多数承重锚点（~570 个深核抽样中 >95%）符号与行号正确，深度与「What/Why/How/坑」结构完整。但存在**系统性**需修问题，最严重者一句概括：

**observability.md 的 mermaid 流程图使用了两个 SSOT 中根本不存在的类名 `MetricsReporter` 与 `PoolStatsObserver`（真实类为 `SchedulerMetricsReporter` / `SchedulerPoolStatsObserver`），以及 scheduler.md 出现一条漂移 834 行的锚点（`scheduler.py:3549` 应为 2715）和一组全部错标的 `new_chunked_req` 锚点。**

次要系统性问题：① 多篇文档（observability / sampling / quantization / parallelism 部分）大量使用**缺前缀裸文件名**锚点，无法直接对 SSOT 根解析；② 多处以「调用点/局部变量/同名异义符号」冒充「定义点」，构成符号错配（symbol_mismatch）。

---

## 逐篇

### docs/deep-dive/observability.md
- 评分: 7/10
- 锚点符号准确性: OK=62 漂移=3 伪造=4 占位=0（承重锚点深核抽样 ≥15）
- 深度问题: 无显著泛泛而谈段落；§0「SGLang 的可观测性体系由三层构成…」为准确总览，§「诊断指标的价值」为真实排障指引，均非凑字数。
- mermaid问题: **图类名虚构 2 处**（见 ISSUES I-01、I-02）：flowchart（L15-37）中 `MetricsReporter`、`PoolStatsObserver` 在 SSOT 中不存在。其余节点（SchedulerMetricsCollector.log_stats、RequestMetricsExporterManager、RadixCacheMetricsCollector 等）均真实。

### docs/deep-dive/parallelism.md
- 评分: 9/10
- 锚点符号准确性: OK=47 漂移=0 伪造=1 占位=0（承重锚点深核抽样 ≥15）
- 深度问题: 无；每处论断均带代码锚点，EPLB 章节与 OPEN 标注诚实。
- mermaid问题: 三张图（L25-37 顶层布局 / L87-99 时序 / L160-170 EPLB）所有 participant/class 名（launcher(_init_parallel_groups)、init_distributed_environment、initialize_model_parallel、GroupCoordinator、EPLBManager.rebalance、rebalance_experts、balanced_packing、replicate_experts、ExpertLocationUpdater 等）均经 grep 确认真实存在，无虚构。

### docs/deep-dive/quantization.md
- 评分: 9/10
- 锚点符号准确性: OK=95 漂移=3 伪造=0 占位=0（承重锚点深核抽样 ≥20）
- 深度问题: 无显著凑字数；§0 三轴辨析有表支撑，§2.2「权衡」为准确归纳。
- mermaid问题: 两张图（L98-123 量化替换链路 / L264-292 FP4 时序）所有 class/def 名（ServerArgs、ModelConfig、QuantizationConfig、Fp8LinearMethod、NVFP4KVCacheMethod、MHATokenToKVPool、FlashInferAttnBackend、TRTLLMHAAttnBackend 等）均真实，无虚构（仅 `get_kv_cache_quant_method` 实为模块级函数而非 KVCacheConfigurator 方法，属轻微不精确，不挂 [图类名虚构]）。

### docs/deep-dive/radix-cache.md
- 评分: 9/10
- 锚点符号准确性: OK=93 漂移=7 伪造=0 占位=0（承重锚点深核抽样 ≥20）
- 深度问题: 无泛泛段落；§2「Why」每个论点都带具体行号锚点，属扎实技术写作。
- mermaid问题: 4 张图（L7-45 / L282-294 / L304-327 / L347-365）中 `EvictionStrategy`（`mem_cache/evict_policy.py:10`）、`ReqToTokenPool`（`mem_cache/memory_pool.py:256`）、`BaseTokenToKVPoolAllocator`（`mem_cache/allocator/base.py:27`）、`RadixCache`/`TreeNode`/`RadixKey` 等均真实存在，无 [图类名虚构]。

### docs/deep-dive/sampling.md
- 评分: 8/10
- 锚点符号准确性: OK=51 漂移=3 伪造=1 占位=0（承重锚点深核抽样 ≥15）
- 深度问题: 无；§1「四层拆分」总览准确，§6 坑点均为真实代码约束。
- mermaid问题: flowchart（L14-37）所有节点（`_preprocess_logits`、`apply_custom_logit_processor`、`sanitize_nan_logits`、`apply_logits_bias`、`penalizer_orchestrator.apply`、`grammar_mask.apply`、`Sampler.forward`、`_sample_from_probs` 等）经 grep 确认真实，无虚构。

### docs/deep-dive/scheduler.md
- 评分: 7/10
- 锚点符号准确性: OK=58 漂移=6 伪造=3 占位=0（承重锚点深核抽样 ≥15）
- 深度问题: 无凑字数；含诚实 OPEN 标注（DP-attn/spec 同步、retraction TODO），技术密度高。
- mermaid问题: 三张图（L49-62 / L101-114 / L180-190）符号（event_loop_normal、get_next_batch_to_run、run_batch、process_batch_result、init_next_round_input、add_one_req、mix_with_running、check_decode_mem、retract_decode、prepare_for_decode 等）均真实，无 [图类名虚构]。

### docs/deep-dive/server-entrypoint.md
- 评分: 9/10
- 锚点符号准确性: OK=85 漂移=2 伪造=0 占位=0（承重锚点深核抽样 ≥15）
- 深度问题: 无泛泛段落；§1/§5 为准确综合，两处 OPEN 明确标注未完全追到。
- mermaid问题: flowchart（L67-89）中 `TokenizerManager`（`managers/tokenizer_manager.py:374`）、`Scheduler`（`managers/scheduler.py:378`）、`DetokenizerManager`（`managers/detokenizer_manager.py:91`）、`OpenAIServingBase`、`OllamaServing`、`AnthropicServing`、`GenerateReqInput` 等全部真实，无 [图类名虚构]。

### docs/deep-dive/speculative-decoding.md
- 评分: 8/10
- 锚点符号准确性: OK=31 漂移=2 伪造=0 占位=0（承重锚点深核抽样 ≥18）
- 深度问题: 无凑字数；§1.1 背景段为准确陈述，技术细节密集。
- mermaid问题: 三张图（L71-82 / L131-144 / L230-239）中 `EagleDraftWorker`、`EAGLEWorkerV2`、`EagleVerifyInput`、`EagleDraftInput`、`build_tree_kernel_efficient`、`run_eagle_verify`、`eagle_sample`、`_finalize_accept_tree_path` 等全部真实，无 [图类名虚构]（注：`verify_tree_greedy`/`tree_speculative_sampling_triton` 为 sgl_kernel CUDA/Triton 核，文档已 OPEN 标注为外部，Python 包装 `verify_tree_greedy_func` 锚点正确）。

---

## ISSUES（###I 分隔）

###I
FILE: docs/deep-dive/observability.md
SEVERITY: high
TYPE: mermaid_fake
DETAIL: L15-37 流程图 participant `MetricsReporter` 在 SSOT 中不存在。文中 §3「指标如何被填充」称「`MetricsReporter`（位于 `metrics_reporter.py`）…」——实测 `python/sglang/srt/managers/scheduler_components/metrics_reporter.py` 中真实类名为 `SchedulerMetricsReporter`（定义于 `metrics_reporter.py:92`），全仓库 grep `class MetricsReporter` 无结果。读者依图检索将找不到任何类。
SUGGESTED_FIX: 将图中 `SR[Scheduler / ModelRunner]` 之外的 `MR[MetricsReporter]` 改为 `MR[SchedulerMetricsReporter]`；正文 L96、L105 的「`MetricsReporter`（位于 `metrics_reporter.py`）」同步改为 `SchedulerMetricsReporter`。

###I
FILE: docs/deep-dive/observability.md
SEVERITY: high
TYPE: mermaid_fake
DETAIL: L15-37 流程图 participant `PS[PoolStatsObserver]` 在 SSOT 中不存在。真实类为 `SchedulerPoolStatsObserver`（`python/sglang/srt/managers/scheduler_components/pool_stats_observer.py:142`，`class SchedulerPoolStatsObserver`）。正文 L99、L105 也沿用了 `PoolStatsObserver` 这一错误名。
SUGGESTED_FIX: 图中与正文统一改为 `SchedulerPoolStatsObserver`（`pool_stats_observer.py:142`）。

###I
FILE: docs/deep-dive/observability.md
SEVERITY: high
TYPE: symbol_mismatch
DETAIL: L41 引用 `metrics_collector.py:201-212` 作为「DI 可替换：`_counter_cls`/`_gauge_cls`… 默认为 None」的锚点，但 `python/sglang/srt/observability/metrics_collector.py:201` 实为 `def resolve_collector_class(...)`，并非 DI mixin；真正的 `_counter_cls=None` 默认定义在 `:215-226`（`class _StatLoggerDIMixin`），且同句已正确引用 `:215-226`。`:201-212` 是重复且错配的锚点。
SUGGESTED_FIX: 删除 `metrics_collector.py:201-212` 锚点，保留 `metrics_collector.py:215-226`。

###I
FILE: docs/deep-dive/observability.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: L43/L138 引用 `metrics_collector.py:248` 说明「`_log_gauge` 使用 `multiprocess_mode="mostrecent"`，依赖 `PROMETHEUS_MULTIPROC_DIR`」。实测 `:248` 是 `python/sglang/srt/observability/metrics_collector.py:248` 的 import 顺序注释（「We need to import prometheus_client after setting the env variable…」），并非 `_log_gauge`；`_log_gauge` 定义在 `:1127`，而 `multiprocess_mode="mostrecent"` 是各 Gauge 构造处的实参（如 `:273` 附近）。锚点未支撑所引符号。
SUGGESTED_FIX: 改为 `metrics_collector.py:1127`（`def _log_gauge`）并补 `:273`（multiprocess_mode 实参），或指向具体 Gauge 构造行。

###I
FILE: docs/deep-dive/observability.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: L110 引用 `python/sglang/srt/managers/scheduler_components/profiler_manager.py:114-118` 说明「`_start_profile` 若已有 profiling 在进行会报错要求先 `/stop_profile`」。实测 `:114` 位于 `def _init_profile(self, ...)`（def 在 `:85`）内部（`if self.profile_in_progress: ... "Call /stop_profile first."`），而 `def _start_profile` 定义在 `:156`。锚点错挂到 `_init_profile` 而非 `_start_profile`。
SUGGESTED_FIX: 改为 `python/sglang/srt/managers/scheduler_components/profiler_manager.py:156`（`def _start_profile`）或精确标注 `:85`（`_init_profile`）。

###I
FILE: docs/deep-dive/observability.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: L106 引用 `request_metrics_exporter.py:72-156` 作为「`RequestMetricsExporterManager` 把每条请求…写入 `sglang-request-metrics-<小时>.log`」的锚点。实测 `python/sglang/srt/observability/request_metrics_exporter.py:72` 是 `class FileRequestMetricsExporter`，而 `RequestMetricsExporterManager` 定义在 `:159`。`:72-156` 范围覆盖的是 File 实现而非 Manager。
SUGGESTED_FIX: 改为 `request_metrics_exporter.py:159`（`class RequestMetricsExporterManager`）；若想说明文件滚动写，保留 `:94-154`（FileRequestMetricsExporter 的 `%Y%m%d_%H` 逻辑，已另有正确锚点）。

###I
FILE: docs/deep-dive/observability.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L105 引用 `metrics_reporter.py:753` 左右的 decode 控制台行。实测 decode 状态行在 `python/sglang/srt/managers/scheduler_components/metrics_reporter.py:763`（`msg=f"Decode batch… #running-req:"`），偏移约 10 行（文档已自注「753 左右」）。属轻微漂移。
SUGGESTED_FIX: 改为 `metrics_reporter.py:763`。

###I
FILE: docs/deep-dive/observability.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L113 引用 `python/sglang/srt/managers/scheduler_components/profiler_manager.py:252-L264` 说明 RPD（ROCm）扩展，但 `RPD` 分支实测在 `:190`（`:254` 起是 `MEM`/`CUDA_PROFILER` 分支）。范围未覆盖所引 RPD。
SUGGESTED_FIX: 将 RPD 引用点改为 `profiler_manager.py:190`，`:252-L264` 仅对应 MEM/CUDA_PROFILER。

###I
FILE: docs/deep-dive/observability.md
SEVERITY: low
TYPE: missing_prefix
DETAIL: 全文大量使用缺前缀裸文件名锚点，无法直接对 SSOT 根解析。例如 `metrics_collector.py:238`、`metrics_reporter.py:637`、`common.py:2374`、`server_args.py:1517`、`auth.py:100`、`pool_stats_observer.py:120`、`profiler_manager.py:197`、`environ.py:412`、`tokenizer_control_mixin.py:372`、`req_time_stats.py:281`、`serving.py:7`、`one_batch_server.py:48`、`scheduler.py:1099`、`bench_utils.py`、`request_metrics_exporter.py:72` 等。
SUGGESTED_FIX: 补全为完整相对 SSOT 路径，例如：
- `metrics_collector.py` → `python/sglang/srt/observability/metrics_collector.py`
- `metrics_reporter.py` → `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
- `common.py` → `python/sglang/srt/utils/common.py`
- `server_args.py` → `python/sglang/srt/server_args.py`
- `auth.py` → `python/sglang/srt/utils/auth.py`
- `pool_stats_observer.py` → `python/sglang/srt/managers/scheduler_components/pool_stats_observer.py`
- `profiler_manager.py` → `python/sglang/srt/managers/scheduler_components/profiler_manager.py`
- `environ.py` → `python/sglang/srt/environ.py`
- `tokenizer_control_mixin.py` → `python/sglang/srt/managers/tokenizer_control_mixin.py`
- `req_time_stats.py` → `python/sglang/srt/observability/req_time_stats.py`
- `serving.py` → `python/sglang/benchmark/serving.py`
- `one_batch_server.py` → `python/sglang/benchmark/one_batch_server.py`
- `scheduler.py` → `python/sglang/srt/managers/scheduler.py`

###I
FILE: docs/deep-dive/parallelism.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: L155 引用 `python/sglang/srt/eplb/eplb_algorithms/deepseek.py:L86-L168` 标注为「`deepseek.rebalance_experts` 两步核心算法」。实测 `:86` 是 `def rebalance_experts_hierarchical(`，真正的 `def rebalance_experts` 在 `:171`（超出 `[86,168]`）。锚点把层级变体误标为基类方法名。
SUGGESTED_FIX: 若指 `rebalance_experts`，改为 `deepseek.py:L171`；若指 hierarchical 变体，符号名改为 `rebalance_experts_hierarchical` 并保留 `:86-L168`。

###I
FILE: docs/deep-dive/parallelism.md
SEVERITY: low
TYPE: missing_prefix
DETAIL: L81、103、110、179、181、183、185、194、197、199、206、215、219 等处使用裸 `parallel_state.py:L…` 锚点（同节其他锚点为完整 `python/sglang/srt/distributed/parallel_state.py:L…`）。虽可推断，但风格不一致且降低可 grep 性。
SUGGESTED_FIX: 统一补全为 `python/sglang/srt/distributed/parallel_state.py:L…`。

###I
FILE: docs/deep-dive/quantization.md
SEVERITY: high
TYPE: symbol_mismatch
DETAIL: L369（§7.3）引用 `python/sglang/srt/configs/model_config.py:L1440-L1462` 作为「`supported_quantization` 列表，否则会被 `raise ValueError` 挡掉」。实测 `:1440-L1462` 是 `rocm_supported_quantization`（ROCm 专用子集，在 `:1564` 校验处使用）；真正的动态 `supported_quantization = [*QUANTIZATION_METHODS]` 定义在 `:1422`。读者按此锚点去加新方案会改错列表。
SUGGESTED_FIX: 改为 `model_config.py:L1422`（`supported_quantization = [*QUANTIZATION_METHODS]`）；ROCm 子集单独说明为 `:1440-L1462`（`rocm_supported_quantization`）。

###I
FILE: docs/deep-dive/quantization.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: L47 引用 `python/sglang/srt/layers/quantization/nvfp4_online.py:L135` 标注 `NvFp4OnlineConfig`。实测 `class NvFp4OnlineConfig` 定义在 `:32`，`:135` 位于其 `get_quant_method` 方法体内。锚点指向实现而非类声明。
SUGGESTED_FIX: 改为 `nvfp4_online.py:L32`（`class NvFp4OnlineConfig`）。

###I
FILE: docs/deep-dive/quantization.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: L321（§6.1(c)）引用 `fp8.py:L656-L668` 为 `convert_mxfp8_weight_to_block_fp8`。实测 `:656-L668` 是 `process_weights_after_loading_block_quant` 内的**调用点**；该符号定义在 `python/sglang/srt/layers/quantization/mxfp8_block_convert.py`，不在 fp8.py。跟随锚点找不到实现。
SUGGESTED_FIX: 改为 `python/sglang/srt/layers/quantization/mxfp8_block_convert.py`（`def convert_mxfp8_weight_to_block_fp8`），并保留 `fp8.py:L655-L680` 作为调用上下文。

###I
FILE: docs/deep-dive/quantization.md
SEVERITY: low
TYPE: missing_prefix
DETAIL: 全文大量裸文件名锚点（`fp8.py`、`awq/awq.py`、`gptq/gptq.py`、`kv_cache.py`、`fp4_kv_cache_quant_method.py`、`kv_cache_dtype.py`、`memory_pool.py`、`flashinfer_backend.py`、`trtllm_mha_backend.py`、`base_config.py`、`utils.py`、`awq_kernels.py`、`gptq_kernels.py`、`modelopt_quant.py`、`mxfp4.py`、`__init__.py`、`loader.py`、`model_config.py` 等），且混用 `layers/quantization/...` 半前缀。均无 `python/sglang/srt/` 前缀，无法直接对 SSOT 根解析。
SUGGESTED_FIX: 全部补全为完整相对 SSOT 路径，例如 `fp8.py` → `python/sglang/srt/layers/quantization/fp8.py`；`awq/awq.py` → `python/sglang/srt/layers/quantization/awq/awq.py`；`__init__.py` → `python/sglang/srt/layers/quantization/__init__.py`；`loader.py` → `python/sglang/srt/model_loader/loader.py`；`model_config.py` → `python/sglang/srt/configs/model_config.py`；`memory_pool.py` → `python/sglang/srt/mem_cache/memory_pool.py` 等。

###I
FILE: docs/deep-dive/quantization.md
SEVERITY: low
TYPE: symbol_mismatch
DETAIL: L46 注册表行 `mxfp4.py:L1781` 标注 `Mxfp4DynamicQuantMoEMethod`，且 §1 将 `Mxfp4Config` / `QuarkConfig` 并列为 mxfp4.py 条目。实测 `QuarkConfig` 定义在 `python/sglang/srt/layers/quantization/quark/quark.py:55`，不在 mxfp4.py。注册名 `quark_mxfp4→QuarkConfig` 本身正确，但文件关联易误导。
SUGGESTED_FIX: 在表中注明 `QuarkConfig` 位于 `quark/quark.py:55`。

###I
FILE: docs/deep-dive/radix-cache.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: L337 引用 `python/sglang/srt/managers/schedule_batch.py:L1358-L1396` 为 `Req.init_next_round_input`。实测 `def init_next_round_input` 定义在 `:1297`，`:1358-L1396` 是其方法体中部（含 match_prefix_for_req 调用），读者跳转落点不在声明处。
SUGGESTED_FIX: 改为 `schedule_batch.py:L1297`（`def init_next_round_input`）。

###I
FILE: docs/deep-dive/radix-cache.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: L376 引用 `python/sglang/srt/mem_cache/cpp_radix_tree/radix_tree.py:L52-L102` 为 `RadixTreeCpp`。实测 `class RadixTreeCpp` 定义在 `:32`，`:52` 位于其 `__init__` 内部。
SUGGESTED_FIX: 改为 `cpp_radix_tree/radix_tree.py:L32`（`class RadixTreeCpp`）。

###I
FILE: docs/deep-dive/radix-cache.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: 数处轻微漂移：`radix_attention.py:L403-L440`（`unified_attention_with_output` 定义于 `:405`，体止 ~`:452`）；`hiradix_cache.py:L1737-L1768`（match_prefix 体延伸至 ~`:1854`，`:1855` 才是独立的 `_match_prefix_helper`）；`cache_init_params.py:L17` 实为 `:18`（`class CacheInitParams`）；`base_prefix_cache.py:L48` 实为 `:49`（`class MatchPrefixParams`）；`server_args.py:L888`（page_size 字段，关联较弱）。均符号正确、行号小幅偏移。
SUGGESTED_FIX: 将上列锚点行号对齐到真实定义行（405；1737-1854；18；49）。

###I
FILE: docs/deep-dive/radix-cache.md
SEVERITY: low
TYPE: symbol_mismatch
DETAIL: L262-272（§5.2）把 `evict_from_tree_cache` 的语义锚在 `python/sglang/srt/mem_cache/allocation.py:L146-L166、L193-L249`（调用点），但该函数真正定义于 `python/sglang/srt/mem_cache/common.py:L105-L129`（已在 §9 表正确给出）。正文措辞易误读为定义在 allocation.py。
SUGGESTED_FIX: 在 §5.2 明确「定义见 `common.py:L105-L129`，调用点见 `allocation.py:L146-L166`」，或移除 allocation.py 的误导向引用。

###I
FILE: docs/deep-dive/sampling.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: L60 引用 `custom_logit_processor.py:15-L21` 标注 `custom_params`。实测 `python/sglang/srt/sampling/custom_logit_processor.py:15-L21` 是 `def _cache_from_str(json_str: str):`（dill/orjson 辅助函数）；`CustomLogitProcessor` 抽象类定义在 `:24`（`class CustomLogitProcessor(ABC)`），而 `custom_params` 字段实际在 `python/sglang/srt/sampling/sampling_params.py:82`。符号与路径双重错配。
SUGGESTED_FIX: `custom_params` 锚点改为 `sampling_params.py:82`；`CustomLogitProcessor` 锚点改为 `custom_logit_processor.py:24-L44`（该文 L99 已正确给出，须与 L60 自洽）。

###I
FILE: docs/deep-dive/sampling.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: L86 引用 `python/sglang/srt/sampling/penaltylib/orchestrator.py:25-L29` 为 `prepare_if_required` 自判。实测 `:25-L29` 是 `BatchedPenalizerOrchestrator.__init__` 中设置 `is_required` 属性，真正的 `def prepare_if_required` 在 `:201`。
SUGGESTED_FIX: 改为 `orchestrator.py:201`（`def prepare_if_required`）。

###I
FILE: docs/deep-dive/sampling.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L49 表 `sampler.py:282` 标注 `top_k`，但 `:282` 是 flashinfer 调用尾部实参 `filter_apply_order="joint",`；`top_k` 截断逻辑在 `:277-L280`。L152（§7）`sampler.py:561-L608` 标注 `top_k_top_p_min_p_sampling_from_probs_torch`，但其 `def` 在 `:563`（561 为空行）。L79 与 §7/图对 `_sample_from_probs` 一处引 `:210`（调用点）一处引 `:246`（定义），内部不一致。
SUGGESTED_FIX: `top_k` → `sampler.py:277`；`top_k_top_p_min_p_sampling_from_probs_torch` → `sampler.py:563-L608`；统一 `_sample_from_probs` 指针为 `:246`（定义）。

###I
FILE: docs/deep-dive/sampling.md
SEVERITY: low
TYPE: missing_prefix
DETAIL: 正文大量裸文件名锚点（`sampling_params.py`、`sampling_batch_info.py`、`sampler.py`、`penaltylib/orchestrator.py`、`penaltylib/frequency_penalty.py`、`penaltylib/presence_penalty.py`、`penaltylib/repetition_penalty.py`、`penaltylib/min_new_tokens.py`、`custom_logit_processor.py`、`model_runner.py`、`schedule_batch.py`）。§7 虽补了完整路径，但正文与表格仍为裸名。
SUGGESTED_FIX: 补全为 `python/sglang/srt/sampling/sampling_params.py`、`python/sglang/srt/sampling/sampling_batch_info.py`、`python/sglang/srt/layers/sampler.py`、`python/sglang/srt/sampling/penaltylib/*.py`、`python/sglang/srt/sampling/custom_logit_processor.py`、`python/sglang/srt/model_executor/model_runner.py`、`python/sglang/srt/managers/schedule_batch.py`。

###I
FILE: docs/deep-dive/scheduler.md
SEVERITY: high
TYPE: anchor_drift
DETAIL: L201（§4.2）引用 `python/sglang/srt/managers/scheduler.py:3549` 为 `_add_request_to_queue(req, is_retracted=True)`。实测 `def _add_request_to_queue` 定义在 `:2715`（偏差 **834 行**）。读者依此跳转会落在完全无关代码区。
SUGGESTED_FIX: 改为 `scheduler.py:2715`（`def _add_request_to_queue`）。

###I
FILE: docs/deep-dive/scheduler.md
SEVERITY: high
TYPE: symbol_mismatch
DETAIL: L158/§3.3 引用 `python/sglang/srt/managers/schedule_policy.py:1414 / :1379 / :1410` 作为 `self.new_chunked_req = req` 的赋值点。实测这三行均非赋值：`1379` 为 `else:`、`1410` 为 `req.set_extend_range(`、`1414` 为 `self.can_run_list.append(req)`。`new_chunked_req = req` 真实赋值点为 `:1190`（add_one_req 内）与 `:1415`（add_chunked_req 内）。三处锚点全部错标。
SUGGESTED_FIX: 改为 `schedule_policy.py:1190`（add_one_req）与 `:1415`（add_chunked_req），并删除 `:1414/:1379/:1410`。

###I
FILE: docs/deep-dive/scheduler.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: L339 引用 `python/sglang/srt/managers/schedule_policy.py:168-L171` 为 `zero_match_result`（结合 L168-L171）。实测 `zero_match_result` 不在本文件定义（`grep -rn "def zero_match_result" python/sglang/srt/managers/` 无结果，源自 radix_cache 模块），`:168-L171` 仅为调用点（`SGLANG_RADIX_FORCE_MISS` 分支）。
SUGGESTED_FIX: 将 `zero_match_result` 锚点指向其真实定义文件（`python/sglang/srt/mem_cache/radix_cache.py` 中 `zero_match_result`/`_empty_match_result`，约 `:364-L373` / `:416-L422`），并标注 §5.2 此处仅为调用点。

###I
FILE: docs/deep-dive/scheduler.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: L158 引用 `python/sglang/srt/managers/schedule_policy.py:1154` 把「整个序列一次性提交」归因于 `PrefillAdder.add_one_req`（def `:1201`）。实测 `:1154` 位于 `def add_chunked_req`（def `:997`）内（`elif self.rem_chunk_tokens is None: # chunked prefill is disabled`），属 `add_chunked_req` 而非 `add_one_req` 路径，函数归属错误。
SUGGESTED_FIX: 将该论述锚点改挂 `add_chunked_req` 上下文（`:997` / `:1154` 属此函数），或改指 add_one_req 的禁用分支真实行。

###I
FILE: docs/deep-dive/scheduler.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: 多处理漂移：`scheduler.py:1240`（标注 `init_schedule_policy`）真实 `def` 在 `:1204`；`scheduler.py:3308`（标注 `preempt_to_schedule` 调用）真实调用在 `:3309-L3310`；`schedule_batch.py:2856`（标注 `_get_decode_retraction_order`）真实 `def` 在 `:2857`；`scheduler.py:3366`（正文称「batch_is_full 重置」）实测是 `if len(can_run_list) == 0: return None` 早退，并无 batch_is_full 赋值；`batch_result_processor.py:366`（标注 `stream_output`）是 `output_streamer.stream_output(` 调用点，def 在别的文件。
SUGGESTED_FIX: 分别修正：`init_schedule_policy`→`:1204`；preempt 调用→`:3309`；`_get_decode_retraction_order`→`:2857`；删除 `:3366` 的 batch_is_full 重置断言（或改指真实重置行）；`stream_output` 标注为调用点。

###I
FILE: docs/deep-dive/server-entrypoint.md
SEVERITY: low
TYPE: symbol_mismatch
DETAIL: L151 引用 `python/sglang/srt/entrypoints/engine.py:L884-L918` 标注 `run_scheduler_process_func`、L151 引用 `:920-L934` 标注 `DataParallelController`。实测 `:884-L918` 内 `run_scheduler_process_func` 仅是 `target=` 引用（def 在 `python/sglang/srt/managers/scheduler.py`），`:920-L934` 实际是 `target=run_data_parallel_controller_process` 包装，类 `DataParallelController` 定义在 `python/sglang/srt/managers/data_parallel_controller.py:132`。二者均非本文件定义。
SUGGESTED_FIX: `:884-L918` 改标注为 `_launch_scheduler_processes` 体或指向 `managers/scheduler.py` 的 `run_scheduler_process_func` 定义；`:920-L934` 改标注为 `run_data_parallel_controller_process` 或指向 `data_parallel_controller.py:132`（类）。

###I
FILE: docs/deep-dive/server-entrypoint.md
SEVERITY: low
TYPE: symbol_mismatch
DETAIL: L36/§2.1 引用 `engine.py:L985-L1020` 为「worker 路由逻辑」（含 `--tokenizer-worker-num` / `--detokenizer-worker-num`）。实测 `:985-L1020` 仅覆盖 **detokenizer** worker 路由（`MultiDetokenizerRouter` 等），tokenizer worker 路由（`MultiTokenizerRouter`）在 `:1207` 附近。范围不完整。
SUGGESTED_FIX: 拆分或补注：detokenizer 路由 `:985-L1020`，tokenizer 路由另引 `:1207` 附近。

###I
FILE: docs/deep-dive/speculative-decoding.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: L255（§6.2）引用 `python/sglang/srt/speculative/eagle_worker_common.py:254-292` 为 `duplicate_prefix_tail_to_draft_branches`。实测 `def duplicate_prefix_tail_to_draft_branches` 定义在 `:58`，`:254-292` 是无关代码。读者跳转落空。
SUGGESTED_FIX: 改为 `eagle_worker_common.py:58`（`def duplicate_prefix_tail_to_draft_branches`）。

###I
FILE: docs/deep-dive/speculative-decoding.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L118 引用 `python/sglang/srt/speculative/eagle_utils.py:135-209` 为 `TreeMaskMode`。实测 `class TreeMaskMode(IntEnum)` 仅 `:135-L139`；`:140-L209` 覆盖 `default_tree_mask_mode` 与 `build_tree_kernel_efficient` 起始（`:147-L289`），与同页 `eagle_utils.py:147-289`（`build_tree_kernel_efficient`）范围重叠，易混淆。
SUGGESTED_FIX: `TreeMaskMode` 收紧为 `eagle_utils.py:135-L139`（如需含 `default_tree_mask_mode` 则 `:135-L144`）。

###I
FILE: docs/deep-dive/speculative-decoding.md
SEVERITY: low
TYPE: symbol_mismatch
DETAIL: L95（§3.2）正文提及 `assign_draft_cache_locs_contiguous` 作为 KV 位置批量分配函数。实测全 `python/sglang/srt/speculative/` 目录 grep 无此函数定义（命名疑似记忆误差）。虽非编号锚点，但作为正文引用符号应核实。
SUGGESTED_FIX: 核实真实分配函数名（疑似 `assign_draft_cache_locs` 或 `assign_extend_cache_locs_uniform_func` 系列），替换该引用；或在无确切符号时改为描述性表述。

---

## TOP8（最该优先修的8条）

1. **[observability / high / mermaid_fake]** 流程图与正文使用 SSOT 不存在的类名 `MetricsReporter`、`PoolStatsObserver`（真实为 `SchedulerMetricsReporter` / `SchedulerPoolStatsObserver`）。→ 见 I-01、I-02，全局替换。
2. **[scheduler / high / anchor_drift]** `scheduler.py:3549` 标注 `_add_request_to_queue`，真实定义在 `:2715`，偏差 834 行。→ I-18。
3. **[scheduler / high / symbol_mismatch]** `schedule_policy.py:1414/1379/1410` 三处 `new_chunked_req` 赋值锚点全部错标，真实点为 `:1190` 与 `:1415`。→ I-19。
4. **[sampling / high / anchor_fake]** `custom_logit_processor.py:15-L21` 实为 `_cache_from_str` 辅助函数，非 `CustomLogitProcessor`/`custom_params`（后者在 `sampling_params.py:82`）。→ I-15。
5. **[quantization / high / symbol_mismatch]** `model_config.py:L1440-L1462` 实为 `rocm_supported_quantization`，非 `supported_quantization`（后者在 `:1422`）；按此加新量化方案会改错列表。→ I-12。
6. **[observability / high / symbol_mismatch]** 多处符号错配锚点：`metrics_collector.py:201-212`（实为 resolve_collector_class）、`:248`（import 注释非 `_log_gauge`）、`profiler_manager.py:114-118`（`_init_profile` 非 `_start_profile`）、`request_metrics_exporter.py:72-156`（File 实现非 Manager）。→ I-03、I-04、I-05、I-06。
7. **[scheduler / medium / symbol_mismatch]** `schedule_policy.py:168-L171` 将 `zero_match_result` 标为本文件定义（实为 radix_cache 模块导入）；`:1154` 把「一次性提交」误挂 `add_one_req`（实为 `add_chunked_req`）。→ I-20、I-21。
8. **[parallelism / medium / symbol_mismatch]** `deepseek.py:L86-L168` 误标为 `rebalance_experts`（实为 `rebalance_experts_hierarchical`，真实 `rebalance_experts` 在 `:171`）。→ I-11。

> 附：缺前缀裸文件名锚点为跨文档普遍问题（observability / sampling / quantization / parallelism 部分），建议统一脚本补全 `python/sglang/srt/` 前缀后再发布（详见各篇 `missing_prefix` 条目）。
