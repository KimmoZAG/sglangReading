# 评阅报告 batch 4

> 评阅人角色：资深 LLM 推理引擎架构师
> 评阅对象：8 篇 SGLang 中文源码文档（见下）
> 唯一事实来源：/home/kimmo/develop/sglang（commit e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7）
> 核验方法：Read 全文 + grep 命中符号 + sed 比对行号区间

## 总体结论

**整体质量评分：9 / 10。**

本批次文档整体工程素养很高：锚点绝大多数可追溯到 SSOT 且符号命中区间准确，尤其是 `config-reference.md`（server_args 字段行号逐个精确）、`open-questions.md`、`dev-setup.md`（环境变量行号精确）几乎零误差。文档普遍覆盖了 What / Why / How / 边界与坑，且有诚实的 `[OPEN]` 标记。

**最严重问题一句话概括：** `reading-guide.md` 存在两处显著行号漂移，其中 `launch_batch_sample_if_needed` 被标到 `scheduler.py:1814`，实际定义于 `scheduler.py:3881`（偏差 ~2067 行），会直接把读者引到错误代码位置。

其余问题均为轻微级（off-by-1、off-by-数十行、跨文件归属或裸文件名路径），未见伪造锚点、占位锚点（`Lx`/`XXX`/`L?`）与图类名虚构。

---

## 逐篇

### docs/quickstart/e2e-observation.md
- 质量评分: 9/10
- 锚点核验: OK≈40 漂移=4 伪造/路径错=0 占位=0
- 深度问题: 无显著"凑字数"段落；What/Why/How/边界齐全，日志样例与指标名均对得上真实 Prometheus 指标。唯一可挑剔处：`5) metrics 端点查看示例`（line 135-164）偏操作手册式，但信息量充足，不构成浅层。
- mermaid 问题: 无。sequenceDiagram 中 `request_logger.log_received_request()` / `ModelRunner.forward()` / `set_prefill_finished_time()` / `collect_metrics()` / `log_time_stats()` 等参与者与方法名均在 SSOT 中真实存在（`tokenizer_manager.py:755` `generate_request`、`schedule_batch.py:1787` `log_time_stats`、`req_time_stats.py:805` 等已验证）。
- 漂移明细：
  - `request_logger.py:200-L234` 标 `_compute_metadata`，实际定义 `request_logger.py:193`（偏差约 7 行）。
  - `metrics_collector.py:1698 / :1708 / :1715` 标三个直方图指标名，实际为 `:1699 / :1709 / :1716`（均 off-by-1）。

### docs/hacking/dev-setup.md
- 质量评分: 9/10
- 锚点核验: OK≈55 漂移=1（路径类） 伪造/路径错=0 占位=0
- 深度问题: 无浅层段落；环境变量清单由 `grep` 实测而来，默认值与锚点精确（environ.py 各 `EnvBool/EnvInt` 行号全中）。`2. Why`（line 31-35）略偏"常识性"，但服务于新手，可接受。
- mermaid 问题: 无。三张图（graph TD / graph LR / flowchart / sequenceDiagram）节点为描述性文字或真实角色名，无虚构类名。
- 路径问题：`contribution_guide.mdx` 被以裸文件名引用（如 line 34、62、79、197、199、201），但 SSOT 中真实路径为 `docs/docs/developer_guide/contribution_guide.mdx`，在仓库根无法解析。建议统一补全目录前缀。其余跨引用 `install.mdx`、`test/README.md`、`.pre-commit-config.yaml`、`python/pyproject.toml` 均真实存在。

### docs/hacking/add-a-model.md
- 质量评分: 9.5/10
- 锚点核验: OK≈30 漂移=0 伪造/路径错=0 占位=0
- 深度问题: 无。How（六步手把手）与边界（6 条坑）信息密度高，且 `EntryClass` 必须精确匹配、`load_weights` v1/v2 分流、`tie_word_embeddings` 等陷阱均给出真实行号证据，`[OPEN]` 对 v2 迁移状态做了诚实标注。
- mermaid 问题: 无。flowchart 中 `get_model_architecture` / `resolve_model_cls` / `TransformersForCausalLM` / `get_model` / `load_model` 均为真实符号（`registry.py:80`、`model_loader/utils.py:197`、`model_loader/__init__.py:23` 已验证）。

### docs/hacking/add-a-kernel-backend.md
- 质量评分: 9/10
- 锚点核验: OK≈35 漂移=1 伪造/路径错=0 占位=0
- 深度问题: 无浅层。基类接口契约、CUDA Graph out/in_graph 边界、forward_mode 分派、HybridAttnBackend 等讲得扎实，`[OPEN]` 对"注册表/choices 两份名单一致性缺启动自检"的洞察准确。
- mermaid 问题: 无。flowchart/sequenceDiagram 的参与者（`ServerArgs`、`ModelRunner`、`resolve_attention_backend_strs`、`build_attention_backends`、`_build_full_attention_backend_from_str`、`ATTENTION_BACKENDS`、`AttentionBackend`）与工厂函数名全部真实（`attention_backend_setup.py:155/67/249`、`attention_registry.py:31`、`base_attn_backend.py:33` 已验证）。
- 漂移明细：
  - `attention_backend_setup.py:L191-L222` 标 `_build_resolved_backend`，实际定义 `attention_backend_setup.py:179`（函数体延续至约 222）。
  - 备注：`flashinfer_backend.py:L719-L721` 标 `_prepare_cuda_graph_metadata`，该处是**调用点**（line 719 调用），真实定义位于 `:1202`。作为"调用处"引用可接受，仅提示读者这是调用点而非定义。

### docs/hacking/reading-guide.md
- 质量评分: 8/10
- 锚点核验: OK≈30 漂移=2（其中 1 处严重） 伪造/路径错=0 占位=0
- 深度问题: 无浅层；四文件闭环路线与断点建议实用。但两处关键锚点漂移降低了可信度（见下），且作为"入口级"文档，错误行号对初学者误导最大。
- mermaid 问题: 无。flowchart 各节点（`run_server`、`http_server.launch_server`、`run_scheduler_process`、`Scheduler.__init__`、`dispatch_event_loop`、`event_loop_overlap/normal`、`get_next_batch_to_run`、`run_batch`、`ModelRunner.forward`、`_forward_raw`、`RadixCache.match_prefix/insert/evict`）均为真实符号（已抽样验证 `launch_server.py:15`、`scheduler.py:4894/1714/1749/3012/3623`、`model_runner.py:1510/1654`、`radix_cache.py:376/436/592`）。
- 漂移明细（重点）：
  - **`launch_batch_sample_if_needed` 标 `scheduler.py:1814`，实际定义在 `scheduler.py:3881`**（偏差 ~2067 行）；原文 line 73、133 以 `:1814(-1815)` 标注"overlap 版本多了 `launch_batch_sample_if_needed`（:1814）"，但 :1814 处为 `if self.is_generation:`。严重影响定位。
  - `init_model_worker` 标 `scheduler.py:901`，实际定义在 `scheduler.py:986`（偏差 ~85 行，原文 line 141）。

### docs/appendix/glossary.md
- 质量评分: 9/10
- 锚点核验: OK≈50 漂移=0 伪造/路径错=0 占位=0
- 深度问题: 无浅层；高频术语表 + 分主题深读 + 边界与坑组织良好，RadixAttention/RadixCache、TP/PP/DP/EP 细分、EAGLE 等解释与源码一致。
- mermaid 问题: 无。各图节点（`match_prefix`、`cache_finished_req`、`evict`、`waiting_queue`、`Scheduler`、`running_batch`、`ModelRunner`、`PrefillAdder`、`merge_batch`、`mix_with_running`）均为真实符号/真实对象。
- 轻微归属提示：`mix_with_running` 与 `merge_batch` 实为 `ScheduleBatch` 的方法，文中以 `scheduler.py:3440`、`:3085-L3099` 标注（3440 仅为调用点），建议补 `schedule_batch.py` 中的真实定义行，避免读者在 scheduler.py 内找不到方法定义。

### docs/appendix/config-reference.md
- 质量评分: 10/10
- 锚点核验: OK≈60 漂移=0 伪造/路径错=0 占位=0
- 深度问题: 无。ServerArgs 字段行号逐条精确（抽样 40+ 行全部命中，如 `model_path:489`、`tp_size:1002`、`ep_size:2314`、`log_level:1467`、`enable_metrics:1517`、`watchdog_timeout:1222`、`disable_cuda_graph:1884` 等），字段关联/互斥约束与 `__post_init__` 后处理行号均准确；Why 与边界章节信息量大。
- mermaid 问题: 无。`prepare_server_args`、`ServerArgs.add_cli_args`、`ConfigArgumentMerger.merge_config_with_args`（位于 `server_args_config_parser.py:52`）、`parser.parse_args`、`_apply_fuseep_mode_env_compat`（`server_args.py:9636`）、`from_cli_args`、`run_post_process_pass`（`server_args.py` 内真实函数）节点均真实存在。

### docs/appendix/open-questions.md
- 质量评分: 9.5/10
- 锚点核验: OK≈40 漂移=0 伪造/路径错=0 占位=0
- 深度问题: 无。作为"待验证问题索引"定位准确，9 大模块 32 条问题均附真实证据锚点，并诚实标注"跨模块才能闭环""SSOT 之外实现（sgl_kernel / 外部 Router）"。文中已主动纠正一处路径漂移（`managers/metrics_collector.py` → `observability/metrics_collector.py`）。
- mermaid 问题: 无。分布图为问题标签节点（Q1…Q27）与依赖边，非类名虚构。
- 备注：文中 `python/sglang/lang/interpreter.py` 与 `python/sglang/cli/serve.py` 均以**无 `srt/` 前缀**的正确路径引用（SSOT 真实路径即如此），不存在缺失文件。

---

## ISSUES（机器可解析，每条一行，用 ###I 分隔）

###I
FILE: docs/hacking/reading-guide.md
SEVERITY: high
TYPE: anchor_drift
DETAIL: 引用 `python/sglang/srt/managers/scheduler.py:1814`（及 :1814-1815）作为 `launch_batch_sample_if_needed` 的位置（line 73、133）。grep 确认该函数真实定义在 `scheduler.py:3881`；:1814 处实际代码为 `if self.is_generation:`。
SUGGESTED_FIX: 将两处锚点改为 `python/sglang/srt/managers/scheduler.py:3881`（必要时补充说明 :1814 是 overlap 分支附近的相关调用，而非函数定义）。

###I
FILE: docs/hacking/reading-guide.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: 引用 `python/sglang/srt/managers/scheduler.py:901` 表示 `init_model_worker`（line 141）。grep 确认 `def init_model_worker` 实际在 `scheduler.py:986`。
SUGGESTED_FIX: 改为 `python/sglang/srt/managers/scheduler.py:986`。

###I
FILE: docs/quickstart/e2e-observation.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: 引用 `python/sglang/srt/utils/request_logger.py:200-L234` 标注 `_compute_metadata`，实际 `def _compute_metadata` 定义在 `request_logger.py:193`（范围起点偏移）。
SUGGESTED_FIX: 改为 `python/sglang/srt/utils/request_logger.py:193-L234`（或以真实结束行收尾）。

###I
FILE: docs/quickstart/e2e-observation.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: 引用 `metrics_collector.py:1698 / :1708 / :1715` 分别标注 `time_to_first_token_seconds` / `inter_token_latency_seconds` / `e2e_request_latency_seconds` 定义行；实测为 `:1699 / :1709 / :1716`（均 off-by-1，line 126、128 速查表）。
SUGGESTED_FIX: 分别改为 `:1699` / `:1709` / `:1716`。

###I
FILE: docs/hacking/add-a-kernel-backend.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: 引用 `attention_backend_setup.py:L191-L222` 标注 `_build_resolved_backend`（line 149、199）。实测 `def _build_resolved_backend` 定义在 `attention_backend_setup.py:179`（函数体延续至约 222）。
SUGGESTED_FIX: 改为 `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:179-L222`。

###I
FILE: docs/hacking/dev-setup.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: 全文多次以裸文件名引用 `contribution_guide.mdx:<行>`（line 34、62、79、197、199、201 等）。SSOT 中该文件位于 `docs/docs/developer_guide/contribution_guide.mdx`，裸文件名在仓库根无法解析，读者无法跳转。
SUGGESTED_FIX: 统一改为完整相对路径 `docs/docs/developer_guide/contribution_guide.mdx:<行>`（行号本身基本准确，仅需补全目录前缀）。

###I
FILE: docs/appendix/glossary.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: 将 `mix_with_running` / `merge_batch` 归在 `scheduler.py:3440` / `:3085-L3099`（line 104）。二者实为 `ScheduleBatch` 的方法，3440 仅是调用点；在 scheduler.py 内按行号找不到方法定义。
SUGGESTED_FIX: 将 `mix_with_running` / `merge_batch` 锚点改为 `python/sglang/srt/managers/schedule_batch.py` 中的真实定义行（或显式说明 3440/3085 为 scheduler 内的调用点）。

---

## TOP5（按 SEVERITY 排序，最该优先修复）

1. **【high】reading-guide.md — `launch_batch_sample_if_needed` 锚点严重漂移**：`scheduler.py:1814` → 真实 `scheduler.py:3881`。作为新手入口文档，此错误会把读者引到完全不相关的代码行，误导最大。
2. **【medium】reading-guide.md — `init_model_worker` 锚点漂移**：`scheduler.py:901` → 真实 `scheduler.py:986`，影响"启动时做了什么"的追踪。
3. **【low】e2e-observation.md — `request_logger._compute_metadata` 起点漂移**：`:200` → 真实 `:193`。
4. **【low】e2e-observation.md — 三个直方图指标行号 off-by-1**：`metrics_collector.py` 的 `:1698/:1708/:1715` → `:1699/:1709/:1716`。
5. **【low】add-a-kernel-backend.md — `_build_resolved_backend` 起点漂移**：`attention_backend_setup.py:L191` → 真实 `:179`（另建议 dev-setup.md 补全 `contribution_guide.mdx` 目录前缀）。

> 备注：glossary.md 的 `mix_with_running`/`merge_batch` 跨文件归属提示与上述第 5 优先级同档（low），亦建议一并修正。
