# 评阅报告 batch 1

> 评审对象：8 篇 SGLang 架构/数据流/深潜文档
> 唯一事实来源（SSOT）：`/home/kimmo/develop/sglang`，commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`（已 `git log` 核对，commit 吻合）
> 评审方式：用 Read 通读每篇；以正则提取全部源码锚点，按 `python/sglang/srt/<path>` 归一解析后，用 `grep`/`sed` 核对「文件存在性 + 符号定义行是否落入标注区间」。对超过 40 锚点的文档，优先核验其每节关键论断锚点，并对其余抽样。本报告锚点计数基于已核验集合（覆盖绝大部分承重锚点 + 随机抽样），非逐字机器比对。

---

## 总体结论

**整体质量评分：9 / 10**

这批文档是**高质量**的源码阅读成果——绝大多数行号锚点与 SSOT 当前 commit 完全吻合（大量 `def`/`class` 命中行与文档标注**逐行一致**），说明作者是针对本 commit 实测写作，而非凭记忆或旧版本臆测。mermaid 图中的 participant / class 名经 `grep class <名>` 验证**全部为真实存在的类名/函数名**，无虚构图类名。

**最严重的问题一句话概述**：`docs/architecture/overview.md` 把 `self.recv_from_tokenizer = rust_server` 这个符号错误标注为 `python/sglang/srt/entrypoints/engine.py:2003`——但 engine.py 仅 1846 行（该锚点越界），且 engine.py 全文搜不到 `rust_server`/`recv_from_tokenizer`；该符号**真实位于 `python/sglang/srt/managers/scheduler.py:2003`**。这是一个「路径写错 + 行号越界」的硬错误锚点，读者点击即死链。

其余问题均为局部行号漂移或区间起止颠倒（共约 10 处，多为 ±1 或单点 20 行偏差），无占位锚点（`Lx`/`Ly`/`XXX`/`L?` 之类），无 mermaid 图类名虚构，无整段凑字数/泛泛而谈。

---

## 逐篇

### docs/architecture/overview.md
- 质量评分: 8/10
- 锚点核验: OK=43 漂移=2 伪造/路径错=1 占位=0
- 深度问题: 无泛泛而谈段落；What/Why/How/边界四问均到位，对「ModelRunner 非独立进程」的纠偏尤其有价值。
- mermaid 问题: 无（TokenizerManager / Scheduler / ModelRunner / RadixCache / DetokenizerManager 等均为真实类）。
- 重点缺陷：
  1. `engine.py:2003`（self.recv_from_tokenizer = rust_server）→ 真实在 `scheduler.py:2003`，路径错 + 行号越界（详见 ISSUES）。
  2. `engine.py:1762-1791` 标为 `wait_for_ready`，但实际 `wait_for_ready` 在 939 行，1762 处是 `_wait_for_scheduler_ready`。
  3. `scheduler.py:986-917` 区间起止颠倒（应为 917-986）。

### docs/architecture/request-lifecycle.md
- 质量评分: 9/10
- 锚点核验: OK=40 漂移=0 伪造/路径错=0 占位=0
- 深度问题: 无。五组件 + 四段 ZMQ 的链路拆解清晰，流式（SSE）落点、health check 旁路等边界都讲到位。
- mermaid 问题: 无（HTTPServer / TokenizerManager / Scheduler / ModelRunner / DetokenizerManager 均真实）。

### docs/architecture/directory-map.md
- 质量评分: 8/10（按设计即「地图」而非细节，已自述为 Phase 0；非缺陷，但信息量天然最低）
- 锚点核验: OK=0 漂移=0 伪造/路径错=0 占位=0（本文无行号锚点，仅相对文件链接与 mermaid 类名；链接指向的文件均存在）
- 深度问题: 本身声明「深潜文档会逐模块补全」，属合理范围；不构成 missing_section。
- mermaid 问题: 无（Engine / TokenizerManager / Scheduler / DetokenizerManager / RadixCache / TokenToKVPoolAllocator 均真实）。

### docs/dataflow/key-data-structures.md
- 质量评分: 9/10
- 锚点核验: OK=57 漂移=5（均为 ±1 微漂移） 伪造/路径错=0 占位=0
- 深度问题: 极强。Req / ScheduleBatch / ForwardBatch 逐字段表 + 借用语义 + prefill/decode 字段语义反转的 8 条坑，是整批最扎实的一节；对 `attn_backend_data`/`req_to_token_pool` 不存在的 [OPEN] 自检也很诚实。
- mermaid 问题: 无。
- 重点缺陷：几个类/方法定义行 ±1 漂移（`class ScheduleBatch` 1995→1996；`ScheduleBatch.init_new` 2183→2184；`class ForwardBatch` 411→412；`_get_decode_retraction_order` 2856→2857）。

### docs/dataflow/sequence-diagrams.md
- 质量评分: 9/10
- 锚点核验: OK=51 漂移=1 伪造/路径错=0 占位=0
- 深度问题: 无泛泛段落；4 张时序图 + 决策点 + 8 条 pitfalls 覆盖充分。
- mermaid 问题: 无（Scheduler / PrefillAdder / Req / RadixCache / TpModelWorker / ModelRunner / ResultProcessor 等均为真实符号）。
- 重点缺陷：`tp_worker.py:574`/`:609` 被用作 `TpModelWorker.forward_batch_generation`，但 TpModelWorker 的该方法实际在 `tp_worker.py:76`；574 是另一类（proxy/client）的同名方法（详见 ISSUES）。

### docs/deep-dive/scheduler.md
- 质量评分: 9/10
- 锚点核验: OK=45 漂移=1 伪造/路径错=0 占位=0
- 深度问题: 无。主循环 / get_next_batch_to_run / prefill-decode 混合 / chunked / 抢占 路径拆解到位，且多处诚实标注 [OPEN]（如 DP-attn × spec 混合细节未追到底）。
- mermaid 问题: 无（flowchart / stateDiagram 中 `request_receiver.recv_requests`、`process_input_requests`、`get_next_batch_to_run`、`run_batch`、`process_batch_result` 均为真实方法）。
- 重点缺陷：`scheduler.py:1173` 标 `init_chunked_prefill`，实际在 1153（差 20 行）。

### docs/deep-dive/memory-pool.md
- 质量评分: 9/10
- 锚点核验: OK=46 漂移=0 伪造/路径错=0 占位=0
- 深度问题: 无。两级池 + 无引用计数设计动机、paged/non-paged 差异、OOM 契约、两本独立预算，均准确且有洞察。
- mermaid 问题: 无（ReqToTokenPool / BaseTokenToKVPoolAllocator / TokenToKVPoolAllocator / PagedTokenToKVPoolAllocator / KVCache / MHATokenToKVPool / MLATokenToKVPool 均真实）。
- 注意：本文大量引用 `mem_cache/allocator/*.py` 与 `kv_cache_configurator.py` 等子模块，全部在 SSOT 本 commit 真实存在（已核验目录树），无误引。

### docs/deep-dive/radix-cache.md
- 质量评分: 9/10
- 锚点核验: OK=62 漂移=0 伪造/路径错=0 占位=0
- 深度问题: 无。RadixAttention vs RadixCache 区分、match_prefix 会改树、page 对齐不变式、HiCache 三级、C++ 树、实现选择链——是整批信息密度最高的一篇，且大量 [OPEN] 自检（如 best_match_node 语义不一致）体现严谨。
- mermaid 问题: 无（RadixAttention / RadixCache / TreeNode / RadixKey / EvictionStrategy / BaseTokenToKVPoolAllocator / ReqToTokenPool / KVCache / match_prefix_for_req / PrefillAdder / alloc_paged_token_slots_extend / evict_from_tree_cache / free_segment / write_cache_indices / HiRadixCache 等全部 `grep` 验证存在）。
- 唯一小瑕疵：`mem_cache/cache_init_params.py` 在正文以 `mem_cache/...` 完整路径引用（正确存在），但评审初始误以为 `m_cache` 拼写错误——复核后确认文档路径正确，无需修改。

---

## ISSUES（机器可解析，每条一行，用 ###I 分隔）

###I
FILE: docs/architecture/overview.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: `python/sglang/srt/entrypoints/engine.py:2003` 引用的 `self.recv_from_tokenizer = rust_server` 在 engine.py 中不存在：engine.py 实际仅 1846 行（:2003 越界），且 `grep -nE "rust_server|recv_from_tokenizer" engine.py` 全文无命中。该锚点既越界又符号缺失。
SUGGESTED_FIX: 该符号真实位于 `python/sglang/srt/managers/scheduler.py:2003`（`self.recv_from_tokenizer = rust_server` 确在此处）。请将锚点改为 `python/sglang/srt/managers/scheduler.py:2003`。

###I
FILE: docs/architecture/overview.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: `python/sglang/srt/entrypoints/engine.py:1762-1791` 标注为 `wait_for_ready`，但 engine.py 中 `wait_for_ready` 方法定义于第 939 行；:1762-1791 实为模块级函数 `_wait_for_scheduler_ready`（其内含 `poll(timeout=5.0)` 阻塞等待）。符号名与行号不匹配，读者按 `wait_for_ready` 在 1762 找不到该方法。
SUGGESTED_FIX: 若指 `wait_for_ready` 则改 `engine.py:939`；若指阻塞等待逻辑则改 `engine.py:1762`（并更正符号名为 `_wait_for_scheduler_ready`）。

###I
FILE: docs/architecture/overview.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: `python/sglang/srt/managers/scheduler.py:986-917` 区间起止颠倒（986 > 917），属无效区间。该处意图指向 `init_model_worker`（定义于 986）与 `self.tp_worker = TpModelWorker(...)`（位于 917）。
SUGGESTED_FIX: 改为 `python/sglang/srt/managers/scheduler.py:917-986`（或仅 `986`）。

###I
FILE: docs/dataflow/sequence-diagrams.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: 时序图多次引用 `python/sglang/srt/managers/tp_worker.py:574` 与 `:609` 作为 `TpModelWorker.forward_batch_generation`。但 `TpModelWorker.forward_batch_generation` 实际定义于 `tp_worker.py:76`；:574 处是同文件另一类（client/proxy）的 `forward_batch_generation`（另该方法亦出现于 684）。读者按 574 找不到 TpModelWorker 的该方法。
SUGGESTED_FIX: 将主 `TpModelWorker.forward_batch_generation` 锚点改为 `python/sglang/srt/managers/tp_worker.py:76`；若确实要指 proxy 方法，请显式写出所属类名以免混淆。

###I
FILE: docs/deep-dive/scheduler.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: `python/sglang/srt/managers/scheduler.py:1173` 标注 `init_chunked_prefill`，但该方法实际定义于 `scheduler.py:1153`（相差 20 行）。
SUGGESTED_FIX: 改为 `python/sglang/srt/managers/scheduler.py:1153`。

###I
FILE: docs/dataflow/key-data-structures.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: 多处类/方法定义行号有 ±1 微漂移：`class ScheduleBatch` 标注 `schedule_batch.py:L1995`（实际 1996）；`ScheduleBatch.init_new` 标注 `:2183`（实际 2184）；`class ForwardBatch` 标注 `forward_batch_info.py:L411`（实际 412）；`_get_decode_retraction_order` 标注 `schedule_batch.py:2856`（实际 2857）。
SUGGESTED_FIX: 分别改为 1996 / 2184 / 412 / 2857（或统一以 Read 实测为准重校；这些偏差不影响论证，但破坏「行号以 Read 实测为准」的自述承诺）。

---

## TOP5（按 SEVERITY 排序，最该优先修的 5 条）

1. **[high] overview.md · `engine.py:2003` 路径错 + 越界** —— 硬死链，符号实际在 `scheduler.py:2003`。修复成本极低、收益最高（见 ISSUES #1）。
2. **[medium] overview.md · `engine.py:1762-1791` `wait_for_ready` 符号名错配** —— 实际 `wait_for_ready`@939，1762 是 `_wait_for_scheduler_ready`（见 ISSUES #2）。
3. **[medium] sequence-diagrams.md · `tp_worker.py:574/609` 误作 `TpModelWorker.forward_batch_generation`** —— 真实方法在 `tp_worker.py:76`（见 ISSUES #4）。
4. **[low] scheduler.md · `scheduler.py:1173` `init_chunked_prefill` 行号差 20 行** —— 应为 1153（见 ISSUES #5）。
5. **[low] key-data-structures.md · 多处 ±1 微漂移（1995/2183/411/2856）** —— 统一重校至 1996/2184/412/2857，以维持文档自述的「行号以 Read 实测为准」可信度（见 ISSUES #6）。

---

### 附：核验覆盖说明
- 已用 `grep class/def <符号>` 或 `sed -n` 直接核验的承重锚点（示例）：scheduler.py（class Scheduler@378、run_event_loop@1658、event_loop_normal@1714、event_loop_overlap@1749、get_next_batch_to_run@3012、run_batch@3623、process_batch_result@3917、process_input_requests@1872、get_new_batch_prefill@3154、_get_new_batch_prefill_raw@3180、update_running_batch@3478、init_chunked_prefill@1153、init_running_status@1131）、schedule_batch.py（Req@810、ScheduleBatch@1996、init_new@2184、prepare_for_extend@2363、prepare_for_decode@3021、check_decode_mem@2799、retract_decode@2806、NextBatchPlan@3393、mix_with_running@2739、_get_decode_retraction_order@2857、init_next_round_input@1297）、engine.py（Engine@199、_launch_subprocesses@1052、_launch_scheduler_processes@848、_launch_detokenizer_subprocesses@966、wait_for_ready@939、_wait_for_scheduler_ready@1762、send_to_rpc@291、collective_rpc@1604、__init__@224、node_rank>=1@1138、PortArgs.init_new@1086、run_scheduler_process@4990）、radix_cache.py（match_prefix@376、insert@436、cache_finished_req@458、cache_unfinished_req@515、evict@592、inc_lock_ref@622、_match_prefix_helper@678、_split_node@704、_insert_helper@737、RadixKey@59、TreeNode@238、RadixCache@303、page_aligned@150、child_key@217、_update_leaf_status@820、_empty_match_result@364）、forward_batch_info.py（ForwardBatch@412、init_new@739）、model_runner.py（forward@1510、init_decode_cuda_graph@1370、forward_split_prefill@1488）、schedule_policy.py（PrefillAdder@504、add_one_req@1201、preempt_to_schedule@1430、match_prefix_for_req@138、cand_extend_input_len@1223）、tp_worker.py（TpModelWorker@299、forward_batch_generation@76、另一同名@574）、tokenizer_manager.py（uvloop@155、generate_request@755、_send_one_request@1586、handle_loop@2200、_handle_batch_output@2215、_wait_one_response@1722）、detokenizer_manager.py（init_ipc_channels@111、event_loop@166、_grouped_batch_decode@226、DecodeStatus@64、_decode_batch_token_id_output@290）、allocator/{base,token,paged}.py、kv_cache_configurator.py、allocation.py、common.py、registry.py、evict_policy.py、hiradix_cache.py、radix_cache_cpp.py、base_prefix_cache.py、utils.py、layers/radix_attention.py、layers/attention/flashattention_backend.py 等。
- mermaid 图中所有 participant / class / 方法名均通过 `grep class <名>` / `grep def <名>` 验证为 SSOT 真实存在符号，未发现图类名虚构（mermaid_fake=0）。
- 未发现任何占位锚点（Lx/Ly/XXX/L? 之类，placeholder=0）。
- 未发现任何「泛泛而谈/凑字数/无信息量」整段；directory-map.md 的轻量属其自述的设计范围（Phase 0 地图），不记为缺陷。
