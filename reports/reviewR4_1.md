# 评阅报告 R4-1

> 评审对象：8 篇 SGLang 源码文档（docs/appendix/*、docs/architecture/*、docs/dataflow/*）
> 唯一事实来源（SSOT）：/home/kimmo/develop/sglang @ e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7
> 评审方法：符号级锚点核验——解析路径、定位文中紧邻引用的符号、grep 实测该符号真实定义/出现行是否落在 [La,Lb]；行号合法但符号不符即判伪造/符号不符。8 篇文档由 8 个核查子代理并行初核，本报告对 HIGH 级结论逐条用 Bash/Grep/Read 独立复验（model_runner.py 2112 行、schedule_policy.py 1500 行、ConfigArgumentMerger 位置、_forward_raw/ForwardBatch 跨度、deepseek_v3.py 缺失、各虚假类名等均已实测确认）。

## 总体结论

**评分：6 / 10**

最严重问题（一句）：`directory-map.md` 把约 11 个 SSOT 中根本不存在的类名（Program/Interpreter/HTTPServer/TokenToKVPoolAllocator/SpecInfo/DraftWorker/Constraint/MultimodalData/PrefillWorker/DecodeWorker/MetricsCollector）与 1 个不存在的文件（models/deepseek_v3.py）当作"关键类/入口"列出；同时 `overview.md`、`sequence-diagrams.md`、`request-lifecycle.md`、`key-data-structures.md` 存在把符号标到**错误文件或文件越界行号**的伪造锚点（如 model_runner.py:3724 已超出文件 2112 行、init_new 写成 :3386 实为 __str__）。行号"在范围内"不等于"符号准确"——本轮已抓出 20+ 处此类缺陷。

---

## 逐篇

### docs/appendix/config-reference.md
- 评分: 7/10
- 锚点符号准确性: OK≈28 漂移=1 伪造=3（2 处 mermaid 函数虚构 + 1 处函数名错误） 占位=0 缺前缀≈40
- 深度问题：字段分主题表与"坑"章节信息密度高、引用真实 help 文本与 assert 行号，属本批最佳文档之一；唯一泛泛段为 L140 "说明"段（"它们多在 `__post_init__` 之后…读取，因此属于配置之外的运行期覆盖层"），属复述性补充，可接受。
- mermaid问题：
  - 流程图节点 `ConfigArgumentMerger.merge_config_with_args`（L26）不在 `server_args.py`，实测位于 `server_args_config_parser.py:17`（方法 :52）→ **图类名虚构**。
  - 节点 `run_post_process_pass 系列`（L33）不在 `server_args.py`，实测位于 `arg_groups/overrides.py:178` → **图类名虚构**。
  - `ServerArgs.add_cli_args` 标 `server_args.py:8549`，真实 `def add_cli_args` 在 **:8546**（漂移 3 行）。
  - 文中 `_auto_disable_*_cudagraph_if_incompatible`（L160/L180）**无此函数**；真实为 `_apply_cuda_graph_compatibility`（:4479）+ `_disable_*_cudagraph_if_incompatible`（:4521/:4596/:4649）→ 符号不符。

### docs/appendix/glossary.md
- 评分: 8/10
- 锚点符号准确性: OK=28 漂移=6 伪造=1 占位=0
- 深度问题：What/Why/How/坑四节均以真实符号支撑，无明显凑字数段落。
- mermaid问题：4 张图均为概念性 participant，所引函数（match_prefix、cache_finished_req、evict、merge_batch/mix_with_running、EagleDraftWorker、build_tree_kernel_efficient）均实测存在，**无问题**。
- 具体缺陷见 ISSUES I6–I11（scheduler.py:81 为 import 非 PD 入口；parallel_state.py:2286→2285、:2289 实为参数非 attn_dp_size、:2360 断言方向被倒置描述等）。

### docs/appendix/open-questions.md
- 评分: 8/10
- 锚点符号准确性: OK=32 漂移=3 伪造=1 占位=1
- 深度问题：作为开放问题索引，每条含"模块/描述/验证方向/证据锚点"四要素，结构清晰、无泛泛段。
- mermaid问题：Q1–Q27 为问题标签，无需类名校验；无问题。
- 具体缺陷见 ISSUES I12–I15（Q4 把 `PagedTokenToKVPoolAllocator` 锚到 `allocation.py:151-L208`，该类实际在 `allocator/paged.py:105`；Q23 无行号占位等）。

### docs/architecture/directory-map.md
- 评分: 4/10
- 锚点符号准确性: 文件存在 33/34（deepseek_v3.py 缺失）；类名命中 14/25，**11 个类名在 SSOT 中不存在或归属错误** → 伪造/符号不符=11 缺失文件=1
- 深度问题：自述为 Phase 0 地图，"不展开细节"合理；但"关键类/入口"列应至少保证类名真实，本篇未做到。
- mermaid问题：节点 `TokenToKVPoolAllocator (mem_cache/memory_pool.py)` 路径错误，真实类在 `mem_cache/allocator/token.py:28`。
- 具体缺陷见 ISSUES I16–I21（deepseek_v3.py 不存在；Program/Interpreter/HTTPServer 等约 11 个虚构/错归类名）。

### docs/architecture/overview.md
- 评分: 6/10
- 锚点符号准确性: OK=30 漂移=0 伪造=3 占位=0
- 深度问题：进程模型/线程模型章节具体且与源码对应，无泛泛段。
- mermaid问题：两张图（graph TD / sequenceDiagram）边与已验证 socket 名一致，无虚构类名。
- 具体缺陷见 ISSUES I22–I24（3 处把符号标错：init_model_worker↔init_tp_model_worker、handle_batch_token_id_out↔event_loop、cache_finished_req↔insert）。

### docs/architecture/request-lifecycle.md
- 评分: 7/10
- 锚点符号准确性: OK=39 漂移=1 伪造=1 占位=0
- 深度问题：Why/How 各段均绑定具体符号，无凑字数。
- mermaid问题：流程图 `HTTP -->|ZMQ PUSH| TM` 把"同进程内 `await tokenizer_manager.generate_request(...)` 直呼"画成 ZMQ PUSH（实测 http_server.py:882/:913 为进程内生成器调用）→ **图边语义错误**（漂移级）。时序图其余边正确。
- 具体缺陷见 ISSUES I25–I26。

### docs/dataflow/key-data-structures.md
- 评分: 7/10
- 锚点符号准确性: OK≈95 漂移=6 伪造=1 占位=0 缺前缀≈50
- 深度问题：§1–§4 字段表与 prefill/decode 语义差异剖析信息量高；§2 中 "别名…:798-L840" 应细化为 :798-L802 + :835-L839（840 为闭括号），属轻微不精确。
- mermaid问题：节点标签用未加引号的 `[`/`]` 与字面 `\n`（L23/L26），Mermaid 解析会报错，须用 `["..."]` + `<br/>`；`FB -.借用 SB 的 GPU 张量.-> SB` 边方向与"init_new 不得改写 SB"语义相悖。
- 具体缺陷见 ISSUES I27–I35（含 `_forward_raw` 含 init_forward_metadata 的论断错误、ForwardBatch 类体 :412-L638 实为仅字段区、约 50 个缺前缀裸锚点等）。

### docs/dataflow/sequence-diagrams.md
- 评分: 5/10
- 锚点符号准确性: OK≈38 漂移=6 伪造=5 占位=0（mermaid 裸锚点均缺前缀，按占位级处理）
- 深度问题：4+1 张时序图与决策点剖析较扎实；坑 1–8 具体。无明显凑字数段。
- mermaid问题：所有图内裸 `:Lxxx` 缺 `python/sglang/srt/managers/` 等前缀（L62–263）；其中 `prepare_for_extend:2363`、`prepare_for_decode:3021`、`init_next_round_input:1297` 等符号实际在 `schedule_batch.py` 而非默认 `scheduler.py`，按默认解析即变符号不符。
- 具体缺陷见 ISSUES I36–I42（init_new[:3386] 实为 __str__、model_runner.forward[:589] 实为 :1510、maybe_cache_unfinished_req[:280] 实为 common.py:98、schedule_policy.py:3259-3260 越界等 5 处伪造）。

---

## ISSUES

###I
FILE: docs/architecture/directory-map.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: 顶层表 L16 举例 "models/ 如 llama.py、deepseek_v3.py"。实测 `python/sglang/srt/models/` 不存在 `deepseek_v3.py`（仅有 deepseek_v2.py / deepseek_v4.py / deepseek_v4_nextn.py / deepseek_nextn.py / deepseek.py / deepseek_janus_pro.py / deepseek_ocr.py / deepseek_vl2.py / deepseek_common/ 等）。该文件名虚构。
SUGGESTED_FIX: 改为真实存在的 `deepseek_v2.py` 或 `deepseek_v4.py`。

###I
FILE: docs/architecture/directory-map.md
SEVERITY: high
TYPE: symbol_mismatch
DETAIL: "关键类/入口"列列了 11 个 SSOT 中不存在或归属错误的类名：① `lang/` 的 `Program`/`Interpreter`（实测仅有 `SglFunction` ir.py:141、`StreamExecutor` interpreter.py:274、`ProgramState` :852）；② `http_server.py` 的 `HTTPServer`（无此类，文件函数式，`launch_server()` 在 entrypoints/http_server.py:2718）；③ `mem_cache/memory_pool.py` 的 `TokenToKVPoolAllocator`（实际类在 `mem_cache/allocator/token.py:28`）；④ `observability/` 的 `MetricsCollector`（实际 `SchedulerStats`/各 `MetricsCollector` 见 metrics_collector.py，无裸 `MetricsCollector`）；⑤ `speculative/` 的 `SpecInfo`/`DraftWorker`（实际 `SpecInput` spec_info.py:330、`BaseSpecWorker` :147、`EagleDraftWorkerBase` :57）；⑥ `constrained/` 的 `Constraint`（实际 `BaseGrammarObject` :52、`GrammarManager` :26）；⑦ `multimodal/` 的 `MultimodalData`（实际 `MultimodalDataItem` schedule_batch.py:317、`MultimodalInputs` :589）；⑧ `disaggregation/` 的 `PrefillWorker`/`DecodeWorker`（实际 `PrefillBootstrapQueue` prefill.py:119、`DecodeTransferQueue` decode.py:1795）；⑨ `launch_server.py` 的 `launch_server()`（该文件仅 `run_server()` :15）。
SUGGESTED_FIX: 逐条替换为 SSOT 真实类名并在括号标注真实文件路径；本篇为"地图"也应保证类名可被 grep 命中。

###I
FILE: docs/architecture/overview.md
SEVERITY: high
TYPE: symbol_mismatch
DETAIL: L20 与 L168 称 "Scheduler 在其 `init_model_worker()` 中直接构造 `TpModelWorker`" 并锚 `python/sglang/srt/managers/scheduler.py:917-986`。实测：TpModelWorker 构造 `self.tp_worker = TpModelWorker(...)` 在 **:917**，但落在 `def init_tp_model_worker(self):`（def 于 :901）体内；真正的 `def init_model_worker(self):` 定义在 **:986**。所引 917-985 区间不属于 `init_model_worker`，且 917 本身是另一函数体内部行。
SUGGESTED_FIX: 改写为"`init_tp_model_worker`（scheduler.py:901-918）内构造 `TpModelWorker`"，并修正正文 "init_model_worker()" 为 "init_tp_model_worker()"；或锚 `scheduler.py:901-918`（构造点）与 :986（init_model_worker 定义）。

###I
FILE: docs/architecture/overview.md
SEVERITY: high
TYPE: symbol_mismatch
DETAIL: L162 把 `python/sglang/srt/managers/detokenizer_manager.py:166-173` 当作 `DetokenizerManager.handle_batch_token_id_out`。实测 166-173 为 `def event_loop(self):`（166 定义，170 `sock_recv(self.recv_from_scheduler)`，173 `sock_send(self.send_to_tokenizer, output)`）；`handle_batch_token_id_out` 定义在 **:430**。同一 166-173 区间在 L108 已正确用于 `event_loop`，此处符号标注冲突。
SUGGESTED_FIX: 改为 `detokenizer_manager.py:430`（handle_batch_token_id_out）；L108 的 event_loop 锚点保持 :166-174。

###I
FILE: docs/architecture/overview.md
SEVERITY: high
TYPE: symbol_mismatch
DETAIL: L160 把 `python/sglang/srt/mem_cache/radix_cache.py:436-456` 当作 `cache_finished_req` 插入回树。实测 436-456 为 `def insert(self, params):`（436 定义，453 `_insert_helper`），`cache_finished_req` 定义在 **:458**。
SUGGESTED_FIX: 改为 `radix_cache.py:458-L490`（cache_finished_req 真实区间）。

###I
FILE: docs/dataflow/sequence-diagrams.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: L73、L153 在 mermaid/正文写 `ScheduleBatch.init_new(can_run_list) [:3386]`。实测 `ScheduleBatch.init_new` 定义在 `python/sglang/srt/managers/schedule_batch.py:2184`；`:3386` 是 `def __str__(self):`。锚点所指标识符与真实符号错位。
SUGGESTED_FIX: 改为 `python/sglang/srt/managers/schedule_batch.py:2184`。

###I
FILE: docs/dataflow/sequence-diagrams.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: L76、L110 写 `ModelRunner.forward [:589/:609]`、`forward + sample [:609/:651]`。实测 `ModelRunner.forward` 在 `python/sglang/srt/model_executor/model_runner.py:1510`，`sample` 在 **:1771**；文件仅 2112 行，:589/:609/:651 均为无关行（:609 为 `)`、:651 为 `remote_instance_weight_transporter…`）。
SUGGESTED_FIX: 改为 `model_runner.py:1510`（forward）、`:1771`（sample）。

###I
FILE: docs/dataflow/sequence-diagrams.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: L80 写 `maybe_cache_unfinished_req / cache_finished_req [:280/:458]`。实测 `radix_cache.py:280` 是 `def release_host(self):`（TreeNode 方法），并非 `maybe_cache_unfinished_req`；`maybe_cache_unfinished_req` 实际是 `python/sglang/srt/mem_cache/common.py:98` 的模块级函数（在 batch_result_processor.py 内被调用）：`cache_finished_req` 的 :458 部分正确。
SUGGESTED_FIX: 改为 `python/sglang/srt/mem_cache/common.py:98`（maybe_cache_unfinished_req）+ `radix_cache.py:458`（cache_finished_req）。

###I
FILE: docs/dataflow/sequence-diagrams.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: L37 写 `PrefillAdder` 构造接收 `max_prefill_tokens` 与 `chunked_prefill_size`（`schedule_policy.py:3259-3260`）。实测 `python/sglang/srt/managers/schedule_policy.py` 仅 **1500 行**，:3259-3260 超出文件范围；相关参数仅以注释出现在 :1269/:1335。
SUGGESTED_FIX: 定位 `PrefillAdder.__init__` 真实行号（请 grep `def __init__` 于 schedule_policy.py 中 PrefillAdder，约 :1000 附近）后替换；删除越界区间。

###I
FILE: docs/architecture/request-lifecycle.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: L103 写 `forward` 返回 `GenerationBatchResult`… "随后由 `run_batch` 内的 `copy_to_cpu`（如 L3724-3737）把结果 D2H 回 CPU"。实测 `python/sglang/srt/model_executor/model_runner.py` 仅 **2112 行**，L3724-3737 越界；且 `copy_to_cpu` 并非 model_runner.py 的方法（文件中仅 `no_copy_to_cpu` 类 kwarg，无 `copy_to_cpu` 定义）。`copy_to_cpu` 是 `GenerationBatchResult` 的方法，由 `Scheduler.run_batch` 在 `scheduler.py` 内调用（如 :3724/:3901）。
SUGGESTED_FIX: 改为 `python/sglang/srt/managers/scheduler.py:3724`（run_batch 内 copy_to_cpu 调用点），并改写"run_batch 内的 copy_to_cpu"措辞。

###I
FILE: docs/dataflow/key-data-structures.md
SEVERITY: high
TYPE: symbol_mismatch
DETAIL: L42 论断 "注意力后端元数据是在 `ModelRunner._forward_raw` 内调用 `attn_backend.init_forward_metadata(fb)` 时由后端临时构造"。实测 `def _forward_raw(self,...)` 在 `python/sglang/srt/model_executor/model_runner.py:1654`，其函数体内**不含** `init_forward_metadata` 调用；`self.attn_backend.init_forward_metadata(forward_batch)` 的真实调用点在 **:1495**（位于 `forward`/`forward_split_prefill` 路径），另有 eager/cuda-graph runner 分支（如 runner/eager_runner.py:232）。把调用归到 `_forward_raw` 是符号不符。
SUGGESTED_FIX: 改为 "在 `ModelRunner.forward` 路径（model_runner.py:1495，或 EagerRunner.run:/forward_split_prefill:1488-1495）调用 `attn_backend.init_forward_metadata(fb)`"，并加锚点。

###I
FILE: docs/appendix/open-questions.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: Q4（L109-113）把 `python/sglang/srt/mem_cache/allocation.py:151-L208` 当作 `PagedTokenToKVPoolAllocator`/`free_pages` 容量证据。实测 `PagedTokenToKVPoolAllocator` 不在 allocation.py，而在 `python/sglang/srt/mem_cache/allocator/paged.py:105`（其 `num_pages = size // page_size` 约 :125）；allocation.py:151-L208 是顶层 `_alloc_extend_loc` 类辅助函数（先 evict 再 alloc），与 free_pages 容量无关。
SUGGESTED_FIX: 改为 `python/sglang/srt/mem_cache/allocator/paged.py:105`（类）与 `:125`（num_pages 容量计算）。

###I
FILE: docs/appendix/config-reference.md
SEVERITY: medium
TYPE: mermaid_fake
DETAIL: 流程图 L26 节点 `ConfigArgumentMerger.merge_config_with_args` 标注 `server_args.py`。实测 `class ConfigArgumentMerger` 在 `python/sglang/srt/server_args_config_parser.py:17`，方法 `merge_config_with_args` 在 :52；server_args.py 仅在 :9679 调用它。图中把函数归到错误文件。
SUGGESTED_FIX: 节点改名/标为 `server_args_config_parser.py:52`（或注明调用点在 server_args.py:9679）。

###I
FILE: docs/appendix/config-reference.md
SEVERITY: medium
TYPE: mermaid_fake
DETAIL: 流程图 L33 节点 `run_post_process_pass 系列` 标注 `server_args.py`。实测 `def run_post_process_pass` 在 `python/sglang/srt/arg_groups/overrides.py:178`，不在 server_args.py。
SUGGESTED_FIX: 标为 `arg_groups/overrides.py:178`。

###I
FILE: docs/appendix/config-reference.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: L160、L180 写 `__post_init__` 中的 `_auto_disable_*_cudagraph_if_incompatible` 系列。实测 server_args.py 中**无** `_auto_disable_*` 前缀方法；真实为 `def _apply_cuda_graph_compatibility(self):`（:4479），其下调用 `_disable_tc_piecewise_cudagraph_if_incompatible`（:4521）、`_disable_breakable_cudagraph_if_incompatible`（:4596）、`_disable_full_prefill_cudagraph_if_incompatible`（:4649）。文档构造的函数名在 SSOT 不存在。
SUGGESTED_FIX: 改为 `_apply_cuda_graph_compatibility`（server_args.py:4479）及 `_disable_*_cudagraph_if_incompatible`（:4521/:4596/:4649）。

###I
FILE: docs/appendix/config-reference.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: 流程图 L24 `ServerArgs.add_cli_args` 标 `server_args.py:8549`。实测 `def add_cli_args` 定义在 **:8546**（8549 是内部 `add_cli_args_from_dataclass` 调用行）。
SUGGESTED_FIX: `server_args.py:8546`。

###I
FILE: docs/appendix/config-reference.md
SEVERITY: low
TYPE: missing_prefix
DETAIL: 字段表与环境变量表中约 40 个锚点用裸 `server_args.py:NNN`（如 L47-180）及个别 `utils/common.py:210`（L129）缺 `python/sglang/srt/` 前缀（同表 L130-138 已用全路径，风格不一致）。
SUGGESTED_FIX: 统一补全为 `python/sglang/srt/server_args.py:NNN`、`python/sglang/srt/utils/common.py:NNN`。

###I
FILE: docs/appendix/glossary.md
SEVERITY: medium
TYPE: anchor_fake
DETAIL: L47 把 `python/sglang/srt/managers/scheduler.py:81` 用作 "Disaggregated Prefill（PD 分离）" 的证据锚点。实测 :81 是 `from sglang.srt.disaggregation.prefill import (...)` 的**import 行**，并非 PD 分离入口；真实入口 `init_disaggregation` 在 :1278，bootstrap 队列构造在 :1392（该 :1392 同条已正确引用）。
SUGGESTED_FIX: 将 :81 改为 `scheduler.py:1278`（或 :1392），保留 :1392。

###I
FILE: docs/appendix/glossary.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L32 `radix_cache.py:307-L308` 标 (page_size)。实测 :307 = `self.token_to_kv_pool_allocator = ...`；:308 = `self.page_size = ...`。区间起点并非 page_size。
SUGGESTED_FIX: 单引 `:308`，或区间标注为 `token_to_kv_pool_allocator / page_size`。

###I
FILE: docs/appendix/glossary.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L49 `radix_cache.py:306-L307` 命名为 `req_to_token_pool / token_to_kv_pool`。实测 :306 = `self.req_to_token_pool`、:307 = `self.token_to_kv_pool_allocator`（真实符号带 `_allocator` 后缀）。
SUGGESTED_FIX: 文档命名改为 `token_to_kv_pool_allocator` 以与 SSOT 一致。

###I
FILE: docs/appendix/glossary.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L124 `parallel_state.py:2286` 标 `initialize_model_parallel`。实测 `def initialize_model_parallel(` 在 **:2285**，:2286 为首个参数行。
SUGGESTED_FIX: `parallel_state.py:2285`。

###I
FILE: docs/appendix/glossary.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L38、L205 把 `parallel_state.py:2289` 当作 `attn_dp_size`。实测 :2289 = 函数参数 `attention_data_parallel_size: int = 1`；局部 `attn_dp_size = attention_data_parallel_size` 定义在 **:2451**。
SUGGESTED_FIX: :2289 标注为参数 `attention_data_parallel_size`；`attn_dp_size` 引用改标 :2451。

###I
FILE: docs/appendix/glossary.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L200 写 `initialize_model_parallel` 断言 `world_size == tp_size * pp_size`。实测 `parallel_state.py:2360` 代码为 `if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:`（即断言**不等则报错**），方向与文档描述相反（行号正确，语义被倒置）。
SUGGESTED_FIX: 改写为 "assert `world_size == tp_size * pp_size`，否则 raise RuntimeError"。

###I
FILE: docs/appendix/open-questions.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: Q4（L113）把 `python/sglang/srt/mem_cache/common.py:105` 作为 `free_pages` 容量（PagedTokenToKVPoolAllocator）证据。实测 :105 = `def evict_from_tree_cache(tree_cache, num_tokens):`，这是 Q5 的正确锚点，被误归到 Q4。
SUGGESTED_FIX: 从 Q4 移除，保留给 Q5；Q4 第二个锚点改用 allocator/paged.py（见 I12）。

###I
FILE: docs/appendix/open-questions.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: Q13（L175）把 `python/sglang/srt/speculative/eagle_utils.py:442` 作为投机接受准则（verify_tree_greedy）证据。实测 :442 = `def get_draft_input_from_target_hidden_dim(...)`（EAGLE3 draft 宽度推导），属 **Q17** 范畴；`verify_tree_greedy` 实际在 sgl_kernel（非本 SSOT）。
SUGGESTED_FIX: 将 eagle_utils.py:442 移至 Q17；Q13 仅保留 spec_info.py:97-L103 并注明 kernel 在 sgl_kernel。

###I
FILE: docs/appendix/open-questions.md
SEVERITY: low
TYPE: anchor_placeholder
DETAIL: Q23（L239）锚点仅写 "（见 python/sglang/lang/ir.py 与 python/sglang/lang/tracer.py 现状）"，无 `:La-Lb`。
SUGGESTED_FIX: 补具体行号，如 `ir.py:141`（SglFunction）、`tracer.py` 中 `trace_program`/`TracerProgramState` 定义行。

###I
FILE: docs/architecture/request-lifecycle.md
SEVERITY: medium
TYPE: mermaid_fake
DETAIL: L33 流程图 `HTTP[HTTPServer] -->|ZMQ PUSH| TM[TokenizerManager]`。实测 HTTP server 与 TokenizerManager 同在**主进程**（engine.py:209 已注明），二者通过进程内 `await tokenizer_manager.generate_request(...)` 直呼（http_server.py:882/:913），并非 ZMQ PUSH。文本描述（L41、L52）正确，仅图边错误。
SUGGESTED_FIX: 该边改为 "直接调用 / generator yield（同进程）"，仅 TM→Scheduler、Scheduler→Detokenizer、Detokenizer→TM 标 ZMQ。

###I
FILE: docs/dataflow/key-data-structures.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: L112 写 `ForwardBatch` "定义见 `python/sglang/srt/model_executor/forward_batch_info.py:L412-L638`"；L42 又写 `L411-L638`。实测 `class ForwardBatch` 始于 :412（@dataclass :411），但类体延伸到 **:1693**（`def can_run_tbo`）；:413-638 仅 dataclass 字段区，:657 `needs_forward_metadata_init`、:1305 `prepare_mlp_sync_batch` 均为类内方法。文档把"字段区"误当"类定义区"，且 L42/L112 起点行不一致。
SUGGESTED_FIX: 统一为 "`class ForwardBatch` forward_batch_info.py:412（字段区 :413-638，类体至 :1693）"。

###I
FILE: docs/dataflow/key-data-structures.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L140 引 `:431-L434` 为 FIXME(lsyin)。实测 :431 = `# === Borrowed from ScheduleBatch ... ===` 组标题；FIXME 注释在 **:432-434**。
SUGGESTED_FIX: `:432-L434`。

###I
FILE: docs/dataflow/key-data-structures.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L174 `ForwardBatch.init_new` 标 `forward_batch_info.py:L738`。实测 :738 为 `@classmethod`，`def init_new(` 在 **:739**。
SUGGESTED_FIX: `:739`（或 :738-L739）。

###I
FILE: docs/dataflow/key-data-structures.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L123 `positions` 标 `:528`/`:906`/`:926`。实测 decode 分支 `ret.positions = clamp_position(batch.seq_lens)` 在 **:907**（:906 为 `if ret.positions is None:`）；extend 分支 :926 正确。
SUGGESTED_FIX: `:907`。

###I
FILE: docs/dataflow/key-data-structures.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L46 `Req.__init__` 标 `schedule_batch.py:L813-L1204`。实测 `def __init__` :813，函数体终于 **:1203**（:1204 空行，:1205 @property）。
SUGGESTED_FIX: `:813-L1203`。

###I
FILE: docs/dataflow/key-data-structures.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L198 写 prefix_lens/extend_lens "只在 prepare_for_extend 之后填充（:2366-L2404）"。实测 :2366 = `if self.is_dllm():`；实际 `self.prefix_lens =` 在 **:2401**、`self.extend_lens =` 在 **:2402**。
SUGGESTED_FIX: `:2401-L2402`（或较宽的 :2376-L2402）。

###I
FILE: docs/dataflow/key-data-structures.md
SEVERITY: medium
TYPE: mermaid_fake
DETAIL: L23、L26 节点 `SB[ScheduleBatch\nreqs: List[Req]\nforward_mode / seq_lens / prefix_lens ...]` 与 `FB[ForwardBatch\ninput_ids / positions / out_cache_loc ...]` 使用未加引号的 `[`/`]` 且用字面 `\n`（Mermaid 不渲染换行、且 `[]` 在标签内会破坏解析）。`FB -.借用 SB 的 GPU 张量.-> SB` 边方向与现实"SB 张量别名引用进 FB"相反，也违背文中"init_new must not mutate SB"语义。
SUGGESTED_FIX: 改为 `SB["ScheduleBatch<br/>reqs: List[Req]<br/>..."]`；边改为 `SB -.张量别名引用.-> FB`。

###I
FILE: docs/dataflow/key-data-structures.md
SEVERITY: medium
TYPE: missing_prefix
DETAIL: 全文约 50 个裸 `:Lxxx` 锚点缺 `python/sglang/srt/managers/` 或 `python/sglang/srt/model_executor/` 前缀。表 1.2（schedule_batch.py）与表 1.4（forward_batch_info.py）共用同一套裸语法，如 `:800` 在前者是 `ReqKvInfo`、在后者是 `seq_lens=`，按默认解析会产生歧义/符号不符。
SUGGESTED_FIX: 每张表头声明 `路径前缀：python/sglang/srt/managers/schedule_batch.py` 等，或所有锚点写全相对 SSOT 路径。

###I
FILE: docs/dataflow/sequence-diagrams.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: L79、L112 写 `process_batch_result_prefill [:3917/:193]`、`process_batch_result_decode [:3917/:805]`。实测 `scheduler.py:3917` = `def process_batch_result(`（通用分发器），并非 `_prefill`/`_decode` 具体方法；`:193`/`：805` 分别是 `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` 中 `process_batch_result_prefill`(:193)/`process_batch_result_decode`(:805)。混合物件跨文件行号易误导。
SUGGESTED_FIX: 改为 `batch_result_processor.py:193`（prefill）/`:805`（decode），或 `scheduler.py:3917` 仅标通用分发器。

###I
FILE: docs/dataflow/sequence-diagrams.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: L200 写 `schedule_policy.py:1374`（`_update_prefill_budget`）。实测 `_update_prefill_budget` 定义在 :857，调用点约 **:1418**（doc 写 1374 为相邻预算更新处，行号偏移）。
SUGGESTED_FIX: `schedule_policy.py:1418`（调用点）或 :857（定义）。

###I
FILE: docs/dataflow/sequence-diagrams.md
SEVERITY: medium
TYPE: missing_prefix
DETAIL: 全部 mermaid 图内裸 `:Lxxx`（L62-263）缺 `python/sglang/srt/managers/` 或 `mem_cache/`、`model_executor/`、`scheduler_components/` 前缀。其中 `prepare_for_extend:2363`、`prepare_for_decode:3021`、`init_next_round_input:1297`、`set_extend_range:1270`、`filter_batch:2847`、`release_req:2897` 等符号实际在 `schedule_batch.py` 而非默认 `scheduler.py`；若读者按裸锚点默认落到 scheduler.py 即变符号不符。文末速查表（L285-297）已用全路径，图内应一致。
SUGGESTED_FIX: 图中每个裸锚点补全为完整相对 SSOT 路径。

---

## TOP8（最该优先修的 8 条）

1. **directory-map.md：约 11 个"关键类"在 SSOT 中不存在**（I17）— 把 Program/Interpreter/HTTPServer/TokenToKVPoolAllocator/SpecInfo/DraftWorker/Constraint/MultimodalData/PrefillWorker/DecodeWorker/MetricsCollector 等虚构/错归类名列为地图核心，属全批最严重的事实性错误，整篇"关键类"列需重写。
2. **directory-map.md：`models/deepseek_v3.py` 文件不存在**（I16）— 作为模型举例的文件名完全虚构，应改为 deepseek_v2.py / deepseek_v4.py。
3. **overview.md：scheduler.py:917-986 把 `init_tp_model_worker` 误标为 `init_model_worker`**（I22）— 核心架构论断（"Scheduler 在 init_model_worker 内构造 TpModelWorker"）与源码不符，且 917 落在另一函数体内。
4. **overview.md：detokenizer_manager.py:166-173 误标为 `handle_batch_token_id_out`**（I23）— 该区间实为 `event_loop`；同一区间在 L108 已正确用于 event_loop，符号标注自相矛盾。
5. **overview.md：radix_cache.py:436-456 误标为 `cache_finished_req`**（I24）— 该区间为 `insert`；真实 `cache_finished_req` 在 :458。回写 radix 的关键论断锚点错。
6. **sequence-diagrams.md：4 处越界/错文件锚点**（I36–I39）— `ScheduleBatch.init_new [:3386]`（实 :2184/__str__）、`model_runner.forward [:589/:609]`（实 :1510/:1771）、`maybe_cache_unfinished_req [:280]`（实 common.py:98）、`schedule_policy.py:3259-3260`（文件仅 1500 行越界）。时序图是本文主体，多处落空。
7. **request-lifecycle.md：model_runner.py:3724-3737 `copy_to_cpu` 越界且符号不在该文件**（I25）— 文件仅 2112 行，且无此法；真实调用在 scheduler.py run_batch。核心数据流论断落空。
8. **key-data-structures.md：`ModelRunner._forward_raw` 内含 `init_forward_metadata` 论断错误**（I27）— `_forward_raw`（:1654）不含该调用；真实调用在 :1495（forward/forward_split_prefill 路径）。架构性误述。

> 备注：config-reference.md 的 2 处 mermaid 虚构函数（I13/I14）与 1 处错误函数名（I15）、glossary 的 scheduler.py:81 import 误用（I6）亦建议随 TOP8 一并修复；其余 low 级漂移/缺前缀问题可在统一复核中批量修正。
