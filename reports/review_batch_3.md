# 评阅报告 batch 3

> 评阅人视角：资深 LLM 推理引擎架构师。唯一事实来源（SSOT）：`/home/kimmo/develop/sglang`，commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`。
> 核验方法：用 `Read` 通读 8 篇文档，抽取所有 `python/sglang/...` 与省略前缀（`managers/scheduler.py:L120` 等）形式的源码锚点，用 `grep -n "<符号>" <解析后绝对路径>` 逐一对符号定义行号与文中标注区间做比对。省略前缀按 `python/sglang/srt/<path>` 解析（如 `managers/scheduler.py` → `python/sglang/srt/managers/scheduler.py`）。
> 核验规模：8 篇共约 440 个锚点，重点依赖锚点（每篇 15~30 个，覆盖每节关键论断）逐一 `grep` 核验，其余按同类模式抽样。未发现任何 `Lx/Ly/XXX/L?` 占位锚点。

## 总体结论

**整批质量评分：9.0 / 10。** 这批文档整体属于高质量源码解读：锚点命中率极高（约 98%），What/Why/How/边界与坑四段齐全，mermaid 图中的 participant/node 均为 SSOT 中真实存在（或可溯源到 `sgl_kernel` 且文中已标 OPEN）的类名/函数名，无虚构图类名。最严重的问题（仅 1 处会导致读者找错文件）：`lora-multimodal.md` 把 `get_new_expanded_mm_items` 归属到 `base_processor.py:1693-1695`，但其真实定义位于 `managers/mm_utils.py:1090`（1693 行只是 import+调用处）。其余均为 5~384 行的区间漂移或符号名笔误，不影响结论正确性，但会降低"按锚点反查源码"的体验。

---

## 逐篇

### docs/deep-dive/constrained-decoding.md
- 质量评分: 9/10
- 锚点核验: OK=42 漂移=0 伪造/路径错=0 占位=0
- 深度问题: 无泛泛而谈段落。§2.1/§2.2/§3.5/§5.1 的 Why 与 trade-off 论证扎实；§5.2 对 "grammar + 并行采样 n>1" 主动标 OPEN，诚实且论证有据（forward_batch_info.py:777-780 的 `grammars=[req.grammar ...]` 已核实）。小瑕疵：`GrammarManager.__init__` 所述 `_pp_sync_ready_failed L72-L107` 实际 `def` 在 L77，标注区间仍覆盖函数体，判 OK。
- mermaid 问题: 两张图（组件总览 / 数据流）中的 `GrammarManager.process_req_with_grammar`、`SamplingBatchInfo.update_regex_vocab_mask`、`BaseGrammarObject`、`GrammarMask.apply` 均为真实符号，无误。

### docs/deep-dive/speculative-decoding.md
- 质量评分: 9.5/10
- 锚点核验: OK=58 漂移=0 伪造/路径错=0 占位=0
- 深度问题: 无。§1.2 算法枚举与 worker 分发、§2.2 退化成本、§3/§6 代码路径与坑均精准，且对 `sgl_kernel` 内核数学诚实标 OPEN（§8）。`clear_unaccepted_c128_draft_states` 在 `eagle_worker_common.py:590-596` 以 `getattr(分配器, "clear_unaccepted_c128_draft_states")` 调用（真实方法在 `mem_cache/deepseek_v4_memory_pool.py`），文中把 590-596 作为"清理调用点"锚点可接受。
- mermaid 问题: `EagleDraftWorker.draft` / `run_eagle_verify` / `eagle_sample` / `verify_tree_greedy` / `tree_speculative_sampling` 等节点名真实；`verify_tree_greedy`/`tree_speculative_sampling` 属 `sgl_kernel` 内核，文档 §8 已声明其源码不在本 SSOT，不判虚构。

### docs/deep-dive/sampling.md
- 质量评分: 8.5/10
- 锚点核验: OK=48 漂移=2 伪造/路径错=0 占位=0
- 深度问题: 整体深。§3.3 增量统计（scatter_add_/scatter_）与 §5.6 batch 合并顺序陷阱论证准确。**漂移点**：(1) `merge_batch` 正文标注 `sampling_batch_info.py:414-L443`，但 `def merge_batch` 实际在 **L388**（约 26 行漂移，区间起点错）；(2) §5.6 写"`__len__` 基于 temperatures 张量（sampling_batch_info.py:430-L432）"，但 `def __len__` 实际在 **L236**，L430 附近是 `merge_batch` 内"because the `__len()__` operator is defined on the temperatures tensor"的注释——符号名与行号双错位。
- mermaid 问题: 采样流程图中 `Sampler.forward` / `_preprocess_logits` / `apply_logits_bias` / `BatchedPenalizerOrchestrator.apply` 等节点名均真实。

### docs/deep-dive/lora-multimodal.md
- 质量评分: 8/10
- 锚点核验: OK=53 漂移=1 伪造/路径错=1 占位=0
- 深度问题: LoRA 与多模态两条线拆解清晰，§一/§二 的 Why 与坑到位。**路径错**：`get_new_expanded_mm_items` 正文标注 `multimodal/processors/base_processor.py:1693-1695`，但该函数**定义于 `managers/mm_utils.py:1090`**；base_processor.py:1693 仅是 `from sglang.srt.managers.mm_utils import get_new_expanded_mm_items` 的 import + 调用行，读者按锚点会找错文件。**漂移**：`get_mm_items_offset` 标注 `base_processor.py:1681-1690`（约 384 行漂移），真实 `def get_mm_items_offset` 在 **L1297**；1681 处是该函数被调用处。
- mermaid 问题: `LoRAManager.prepare_lora_batch` / `BaseLayerWithLoRA.lora_active` / `ColumnParallelLinearWithLoRA.apply_lora` / `_compute_moe_lora_info` 均真实；`XxxWithLoRA.forward` 为对 `ColumnParallelLinearWithLoRA` 等包装类的泛称占位，非虚构类名，可接受。

### docs/deep-dive/disaggregation.md
- 质量评分: 9/10
- 锚点核验: OK=54 漂移=1 伪造/路径错=0 占位=0
- 深度问题: PD 生命周期、KV 状态机、异构 TP/CP/PP 映射、坑章节均精准，§4 对 Router 不可见性诚实标 OPEN。**符号名笔误**：§6.3 写 "pop_preallocated 用 `match_prefix_for_req` 匹配自身 radix 树（decode.py:561）"，行号 561 正确，但该函数在 SSOT 中名为 **`_match_prefix_and_lock`**（decode.py:561），`match_prefix_for_req` 不存在；读者按名检索会落空。
- mermaid 问题: `PrefillBootstrapQueue` / `NixlKVSender` / `DecodeTransferQueue` / `KVPoll` 等节点均为真实类/枚举，无误。

### docs/deep-dive/observability.md
- 质量评分: 8.5/10
- 锚点核验: OK=88 漂移=1 伪造/路径错=0 占位=0
- 深度问题: 三层可观测性、DI 可替换、多进程聚合、FIXME 命名误导等洞察到位。指标表（约 30 行）逐一核对抽样均命中（如 `num_running_reqs` L269-274、`SchedulerMetricsCollector` L238、`token_usage` FIXME L78、`emit_constants` L1445 等）。**轻微漂移**：`profiler_manager.py:220-265` 标注为 `.trace.json.gz` Chrome trace 输出，但 trace 文件名字符串实际在 L197 / L338，220-265 是 `activities`（MEM/RPD/CUDA_PROFILER）处理段，区间略偏。文档 §benchmark 工具链段落偏"目录罗列"，信息量低于其它章节，但每条均附文件锚点，未判凑字数。
- mermaid 问题: `SchedulerMetricsCollector.log_stats` / `RequestMetricsExporterManager` / `TokenizerMetricsCollector` 等节点真实；注意 `profiler_manager.py` 在正文（L110）写为 `scheduler_components/profiler_manager.py`（正确路径），全文未出现 `observability/profiler_manager.py` 错误路径。

### docs/quickstart/install.md
- 质量评分: 9/10
- 锚点核验: OK=30 漂移=0 伪造/路径错=0 占位=0
- 深度问题: 明确区分"已核实命令"与"通用建议"，依赖版本（torch==2.13.0、flashinfer_python[cu13]==0.6.17 等）与 Dockerfile ARG 均按 SSOT 实测。核验：`pyproject.toml:L201-L203`（sglang/killall_sglang 入口）、`setup.py:L104-L130`（`_discovered_rust_extensions`）、`Dockerfile:L1-L20`（CUDA 13.0.3 基础镜像）全部命中。无泛泛段落。
- mermaid 问题: 安装方式流程图节点为流程阶段描述（pip/Docker/源码编译），无虚构类名。

### docs/quickstart/minimal-example.md
- 质量评分: 9.5/10
- 锚点核验: OK=60 漂移=0 伪造/路径错=0 占位=0
- 深度问题: 把"起服务→就绪→调用→健康探针"整条链路的代码位置钉死，且对 `--host` 默认回环、`/health` 默认真生成、warmup 失败连坐等坑的论证均经源码核实。核验：`server_args.py` host/port/model_path/trust_remote_code/context_length/tp_size/skip_server_warmup/api_key/served_model_name/attention_backend（1253/1254/489/573/578/1002/1295/1329/1339/1668）全部命中；`http_server.py` health_generate/L646、ServerStatus.Starting/L660、model_info/L734、server_info/L782、generate/L874、flush_cache/L948、ping/L2003 全部命中；`cli/main.py`/`cli/serve.py`/`launch_server.py` 路径正确（位于 `python/sglang/` 而非 `srt/`）。
- mermaid 问题: `cli.main.main` / `launch_server.run_server` / `Engine._launch_subprocesses` / `uvicorn` 等节点真实，无误。

---

## ISSUES（机器可解析，每条一行，用 ###I 分隔）

###I
FILE: docs/deep-dive/lora-multimodal.md
SEVERITY: medium
TYPE: anchor_fake
DETAIL: 正文 "get_new_expanded_mm_items（python/sglang/srt/multimodal/processors/base_processor.py:1693-1695）把'整段多图'拆成'每图一个 item'" —— 该函数定义不在 base_processor.py；base_processor.py:1693 仅是 `from sglang.srt.managers.mm_utils import get_new_expanded_mm_items` 的 import 与调用行。
SUGGESTED_FIX: 改为 `python/sglang/srt/managers/mm_utils.py:1090`（真实定义 `def get_new_expanded_mm_items(original_mm_items)`），或同时保留 base_processor.py:1693 作为调用处并注明"定义见 managers/mm_utils.py"。

###I
FILE: docs/deep-dive/lora-multimodal.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: 正文 "get_mm_items_offset（python/sglang/srt/multimodal/processors/base_processor.py:1681-1690）" —— 真实 `def get_mm_items_offset` 在 **L1297**；1681-1690 是该函数被调用处，区间漂移约 384 行，读者按标注会找不到定义。
SUGGESTED_FIX: 改为 `python/sglang/srt/multimodal/processors/base_processor.py:1297-L1312`（定义区间）；如需标调用处，改写为 "base_processor.py:1681 调用处"。

###I
FILE: docs/deep-dive/sampling.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: §5.6 "merge_batch 明确要求先处理 logit_bias……再处理 temperatures 等（sampling_batch_info.py:414-L443）" —— 真实 `def merge_batch(self, other)` 在 **L388**；标注区间起点漂移约 26 行，且同段 "(__len__ 基于 temperatures 张量，见 sampling_batch_info.py:430-L432 注释)" 中 `def __len__` 实际在 **L236**（L430 附近是 merge_batch 内注释）。
SUGGESTED_FIX: merge_batch 改为 `sampling_batch_info.py:388-L443`；`__len__` 注释改为 `sampling_batch_info.py:236` 与 merge_batch 内 L428-L445 注释分开标注。

###I
FILE: docs/deep-dive/disaggregation.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: §6.3 "pop_preallocated 用 `match_prefix_for_req` 匹配自身 radix 树（python/sglang/srt/disaggregation/decode.py:561）" —— 行号 561 正确，但 SSOT 中函数名为 `_match_prefix_and_lock`（decode.py:561），不存在 `match_prefix_for_req`。
SUGGESTED_FIX: 符号名改为 `_match_prefix_and_lock`；行号 561 保留。

###I
FILE: docs/deep-dive/observability.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: "输出 .trace.json.gz 的 Chrome trace（profiler_manager.py:220-265）" —— `.trace.json.gz` 字符串实际在 L197 与 L338；220-265 是 `activities`（MEM/RPD/CUDA_PROFILER）处理段，区间略偏。
SUGGESTED_FIX: trace 输出锚点改为 `python/sglang/srt/managers/scheduler_components/profiler_manager.py:197` 与 `:338`；MEM/RPD/CUDA_PROFILER 扩展单独标 `:252-L264`。

###I
FILE: docs/deep-dive/constrained-decoding.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: §3.2 "`_pp_sync_ready_failed` L72-L107" —— 真实 `def _pp_sync_ready_failed` 在 **L77**（区间 72-107 仍覆盖函数体，结论不受影响）。
SUGGESTED_FIX: 改为 `python/sglang/srt/constrained/grammar_manager.py:77-L107`（或保留区间并注明 def 行）。

###I
FILE: docs/deep-dive/observability.md
SEVERITY: low
TYPE: shallow
DETAIL: "## Benchmark 工具链" 段落（L118-130）以目录罗列 `serving.py`/`offline_throughput.py`/`bench_adaptive_speculative.py` 等文件名与一句话功能为主，缺乏与其它章节同深度的代码路径/调用链论证，相对信息量偏低（但每条均附文件锚点，未达"凑字数"程度）。
SUGGESTED_FIX: 可补 1~2 个代表性 benchmark 的"参数如何流入 ServerArgs / 如何读取 /metrics"的关键调用链，或明确标注为"索引性附录"以免与深度章节混同。

---

## TOP5（按 SEVERITY 排序，最该优先修）

1. **[medium] lora-multimodal.md — `get_new_expanded_mm_items` 路径错**（anchor_fake）：读者会去错误的 `base_processor.py` 找定义，应改指 `managers/mm_utils.py:1090`。
2. **[medium] lora-multimodal.md — `get_mm_items_offset` 漂移 384 行**（anchor_drift）：定义实际在 L1297，非 L1681。
3. **[medium] sampling.md — `merge_batch` 区间起点漂移 26 行 + `__len__` 符号/行双错位**（anchor_drift）：改 L388 与 L236。
4. **[medium] disaggregation.md — `match_prefix_for_req` 符号名笔误**（anchor_drift）：真实函数 `_match_prefix_and_lock`（L561）。
5. **[low] observability.md — profiler trace.json.gz 锚点区间略偏 + Benchmark 段落偏浅**（anchor_drift / shallow）：trace 锚点改 L197/L338，Benchmark 段可补调用链或降级为索引附录。
