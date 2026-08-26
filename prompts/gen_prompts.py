#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成每篇文档的自包含 prompt 文件 prompts/task_<i>.txt。
子会话看不到主会话上下文，所以每个 prompt 必须自包含：背景、目标文件、必读源码、写作要求、验收标准。
"""
import os

SSOT = "/home/kimmo/develop/sglang"          # 唯一事实来源
OUT = "/home/kimmo/develop/sglangReading/docs"  # 文档输出根
COMMIT = "e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7"
COMMIT_DATE = "2026-08-14"

PREAMBLE = f"""你是资深 LLM 推理引擎架构师 + 技术文档工程师。现在要为 SGLang 推理引擎写一套高质量中文源码文档。

【唯一事实来源 SSOT】
路径：{SSOT}
对齐 commit：{COMMIT}（{COMMIT_DATE}）
你只能依据上述本地源码阅读得出结论，严禁依赖记忆、网络搜索或任何猜测。所有论断必须能在该 commit 的源码中找到证据。

【铁律（违反即不合格）】
1) 代码优先：结论来自源码阅读，不靠记忆。
2) 可追溯：每个关键论断后给出证据锚点，格式 `python/sglang/srt/managers/scheduler.py:L120-L180`（相对 SSOT 的路径 + 行号区间，行号用 Read 工具确认真实值）。关键函数/类给出真实签名（含关键参数）。
3) 不确定就标注：读不懂或存在多种可能的地方，不要编造。把它写入 `{OUT}/appendix/_openq_<你的文档名>.md`（独立文件，文件名含你的文档名，例如 scheduler 任务写 _openq_scheduler.md），格式：`### <问题>\n<描述与可能的方向>`；同时在该文档正文对应处用 `> **[OPEN]** <简短说明>` 标注。绝对不要直接改 `open-questions.md` 本身（避免并发冲突）。
4) 深度优先：每节回答 What（是什么）/ Why（设计动机与权衡）/ How（关键代码路径，贴真实函数调用）/ 边界与坑。禁止写"XXX 负责管理 XXX"这类无信息量空话。
5) 图必须有：架构/流程/时序/状态机用 Mermaid（```mermaid 围栏代码块）。图中组件名必须与代码中真实类名/函数名一致（如 `Scheduler`、`RadixCache`、`ModelRunner`）。
6) 不要执行任何 git 命令（主会话统一提交）。不要修改 SSOT 源码。只写目标文档文件（必要时可追加 `_openq_*.md`）。
7) 站内交叉链接：禁止创建指向其他 .md 的 markdown 链接（会触发 strict 构建断链）。如需提及别的文档，用纯文本写其相对路径，例如"见 architecture/overview.md"。也不要嵌入图片文件（用 mermaid 代替）。
8) 不要运行 mkdocs build（本机沙箱对 site 目录清理有拦截，会误报失败）；只需保证内容质量与下述验收标准。

【验收标准（必须全部满足，否则视为失败）】
- 目标文件已被完全重写，不再包含 "TODO: 待子任务填充（占位）" 这一行。
- 字数达标（见各任务要求）。
- 至少包含 8 处真实证据锚点（指向 SSOT 真实路径与行号区间）；锚点中的行号必须真实（用 Read 确认）。
- 含至少 1 个 Mermaid 图（architecture/deep-dive/dataflow 类强制；quickstart/hacking/appendix 鼓励）。
- 关键类名/函数名与源码一致，不得杜撰 API 或行号。
- 结构含 What / Why / How / 坑 四节（或等价分节，可细化）。
- 中文写作，术语准确。
"""

def task(num, relpath, words, mermaid, src, focus, reqs, docname):
    body = f"""# 本次任务

目标文件（绝对路径）：{OUT}/{relpath}
文档名（用于 _openq 文件命名）：{docname}
文档类型：{'deep-dive/architecture/dataflow（强制含 mermaid + 字数≥1500）' if mermaid else 'quickstart/hacking/appendix（鼓励 mermaid，字数≥800）'}
最低字数：{words} 字

## 必读源码（先用 Read / Grep / search_content 阅读，再下笔；行号以 Read 实测为准）
"""
    for s in src:
        body += f"- {SSOT}/{s}\n"
    body += """
## 写作重点（回答 What/Why/How/坑）
"""
    for r in focus:
        body += f"- {r}\n"
    body += """
## 具体写作要求
"""
    for r in reqs:
        body += f"- {r}\n"
    body += f"""
## 完成后自检
- 删除文件里原有的 "TODO: 待子任务填充（占位）" 整行。
- 用 Read 读回目标文件确认：无 TODO 占位、含 ≥8 个真实锚点、含 ≥1 个 mermaid 图、字数达标。
- 若有 OPEN 问题，已写入 `{OUT}/appendix/_openq_{docname}.md`。

现在开始：先 Read 上述源码文件，再 Write 完整文档到目标路径。
"""
    path = os.path.join("prompts", f"task_{num}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n" + body)
    print("wrote", path)

# ----------------- 任务定义 -----------------
# 每个任务: (num, relpath, words, mermaid, src列表, focus列表, reqs列表, docname)

T = []

# 波次1：主干 + 核心深潜
T.append((1, "architecture/overview.md", 1800, True,
  ["python/sglang/launch_server.py", "python/sglang/srt/entrypoints/engine.py",
   "python/sglang/srt/managers/tokenizer_manager.py", "python/sglang/srt/managers/scheduler.py",
   "python/sglang/srt/managers/detokenizer_manager.py", "python/sglang/srt/model_executor/model_runner.py",
   "python/sglang/srt/mem_cache/radix_cache.py"],
  ["全局架构：前端 DSL + srt 后端引擎的关系", "多进程模型：TokenizerManager / Scheduler / DetokenizerManager / Worker(ModelRunner) 各进程职责与通信方式（ZMQ socket 名称）",
   "线程模型：每个进程内的主循环/线程", "数据流：请求对象如何在不同进程间流转"],
  ["画出全局架构图（mermaid graph，组件名用真实类名：TokenizerManager、Scheduler、DetokenizerManager、ModelRunner、RadixCache）",
   "说明为什么采用多进程（解耦 tokenizer 阻塞与 GPU 计算）", "给出启动入口 `launch_server.py` 如何组装各组件的真实调用链（带锚点）"],
  "overview"))

T.append((2, "architecture/request-lifecycle.md", 1800, True,
  ["python/sglang/srt/entrypoints/http_server.py", "python/sglang/srt/entrypoints/http_server_engine.py",
   "python/sglang/srt/entrypoints/engine.py", "python/sglang/srt/managers/tokenizer_manager.py",
   "python/sglang/srt/managers/scheduler.py", "python/sglang/srt/model_executor/model_runner.py",
   "python/sglang/srt/managers/detokenizer_manager.py"],
  ["一次 HTTP 请求从进来到 token 返回的完整链路", "跨进程/跨队列的边界（哪些数据通过 ZMQ 发送）", "prefill 与 decode 阶段在链路中的位置"],
  ["用 mermaid sequenceDiagram 画出跨进程时序（participant 用真实类名：HTTPServer、TokenizerManager、Scheduler、ModelRunner、DetokenizerManager）",
   "标注每一步涉及的关键函数与文件:行号", "说明 token 流式返回（SSE/stream）在哪一步发生"],
  "request-lifecycle"))

T.append((3, "dataflow/key-data-structures.md", 2200, True,
  ["python/sglang/srt/managers/schedule_batch.py", "python/sglang/srt/model_executor/forward_batch_info.py",
   "python/sglang/srt/sampling/sampling_params.py", "python/sglang/srt/mem_cache/memory_pool.py"],
  ["Req 结构：逐字段解释（rid、input_ids、output_ids、tokenizer、sampling_params、req_pool_idx 等）",
   "ScheduleBatch：一批请求如何组织，含哪些元数据（batch_size、seq_lens、prefix_lens 等）",
   "ForwardBatch：喂给模型前向的数据结构（forward_mode、attn_backend_data、req_to_token_pool 等）",
   "Req / ScheduleBatch / ForwardBatch 三者转换关系"],
  ["用表格逐字段解释 Req、ScheduleBatch、ForwardBatch（字段名 | 类型 | 含义 | 代码锚点）",
   "用 mermaid 图表达三者关系与转换时机", "指出哪些字段在 prefill/decode 时含义不同（坑）"],
  "key-data-structures"))

T.append((4, "deep-dive/scheduler.md", 3000, True,
  ["python/sglang/srt/managers/scheduler.py", "python/sglang/srt/managers/schedule_batch.py",
   "python/sglang/srt/managers/scheduler_components/request_receiver.py",
   "python/sglang/srt/managers/scheduler_components/batch_result_processor.py",
   "python/sglang/srt/managers/scheduler_input_blocker.py"],
  ["Scheduler 主循环（event loop）的结构：如何等待新请求、如何组装 batch、如何触发 prefill/decode",
   "batch 组装逻辑：get_new_batch_prefill / get_new_batch_decode（或等价函数）如何挑选请求",
   "prefill/decode 混合调度：同一 batch 内能否混部、如何隔离",
   "chunked prefill：长 prompt 如何分块（chunk size 来源与限制）",
   "抢占（preemption）与重试：显存不足时如何驱逐/重算，beam/radix 缓存交互"],
  ["给出主循环关键函数真实签名与调用顺序（带锚点）", "用 mermaid 图/状态机表达调度状态（RUNNING/WAITING/PREFILL/DECODE/PAUSED）",
   "指出 chunked prefill 与 radix 缓存命中协同的坑", "解释抢占触发条件（OOM、显存预算）与恢复路径"],
  "scheduler"))

T.append((5, "deep-dive/memory-pool.md", 2400, True,
  ["python/sglang/srt/mem_cache/memory_pool.py", "python/sglang/srt/mem_cache/allocator/token.py",
   "python/sglang/srt/mem_cache/kv_cache_builder.py", "python/sglang/srt/mem_cache/kv_cache_configurator.py"],
  ["KV cache 池的抽象：TokenToKVPoolAllocator 如何管理显存（page/token 粒度）",
   "req_to_token 映射：请求逻辑 token 到物理 KV slot 的映射表（req_to_token_pool）",
   "分配与释放：alloc/free 接口，引用计数/租约",
   "显存预算与 OOM 处理：如何计算可用 slot、OOM 时如何反馈给 scheduler"],
  ["画出池/分配器/映射表的关系（mermaid）", "给出 allocator 关键方法签名（alloc 一个请求的 token 数、free）与锚点",
   "解释 paged 与非 paged 的差异（若代码支持多种）", "说明 OOM 与 scheduler 抢占的接口契约"],
  "memory-pool"))

T.append((6, "deep-dive/radix-cache.md", 2600, True,
  ["python/sglang/srt/mem_cache/radix_cache.py", "python/sglang/srt/mem_cache/radix_cache_cpp.py",
   "python/sglang/srt/layers/radix_attention.py"],
  ["RadixAttention / RadixCache 的核心思想：用 radix 树对 prompt 前缀做复用",
   "匹配（match_prefix）：新请求如何沿树查找最长可复用前缀，返回复用的 token 数与节点",
   "插入与淘汰（eviction）：新前缀如何写树、LRU/显存压力下如何淘汰节点释放 KV",
   "多级缓存：代码中是否存在 HiCache / CPU/Disk 层级缓存（grep 'HiCache'/'hicache' 核实），若有说明结构"],
  ["用 mermaid 画 radix 树匹配/插入示意（节点代表 token 序列，边代表 token）", "给出 match_prefix / insert / evict 等关键方法签名与锚点",
   "说明命中前缀后如何避免重复 prefill（与 scheduler 协作）", "指出淘汰策略与 memory-pool 释放的耦合点"],
  "radix-cache"))

T.append((7, "deep-dive/model-runner.md", 2600, True,
  ["python/sglang/srt/model_executor/model_runner.py", "python/sglang/srt/model_executor/forward_batch_info.py",
   "python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py"],
  ["ModelRunner 职责：持有模型、接收 ForwardBatch、执行前向、回收结果",
   "forward_batch 的构造与在 prefill/decode 下的差异",
   "CUDA Graph 捕获与 replay：capture 时机、replay 条件、graph 池管理、shape 约束",
   "torch.compile 的使用（若启用）：哪里调用、与 CUDA Graph 的取舍"],
  ["给出 ModelRunner.forward / capture_cuda_graph 等关键签名与锚点", "用 mermaid 表达一次 forward 的流程（含 graph replay 分支）",
   "说明 CUDA Graph 的限制（固定 shape、不能含动态控制流）与 sglang 的应对（piecewise cuda graph 若有）",
   "指出哪些算子无法纳入 graph（坑）"],
  "model-runner"))

T.append((8, "deep-dive/attention-backends.md", 2400, True,
  ["python/sglang/srt/layers/attention/attention_registry.py",
   "python/sglang/srt/layers/attention/base_attn_backend.py",
   "python/sglang/srt/layers/attention/flashinfer_backend.py",
   "python/sglang/srt/layers/attention/flashattention_backend.py",
   "python/sglang/srt/layers/attention/flashmla_backend.py",
   "python/sglang/srt/layers/attention/cutlass_mla_backend.py"],
  ["注意力后端抽象：AttentionBackend 基类接口（forward、init_metadata、metadata 结构）",
   "后端选择逻辑：registry 如何根据模型/硬件/编译选项挑选后端（FlashInfer / FlashAttention / FlashMLA / CutlassMLA 等）",
   "metadata 构造：每种后端需要的 metadata（如 paged kv、ragged、mLA）如何由 scheduler/runner 提供",
   "MLA（Multi-head Latent Attention）相关后端（DeepSeek）的特殊处理"],
  ["画出后端抽象与选择的 mermaid 图（含 registry、base、具体后端类名）", "给出 base backend 关键方法签名与锚点",
   "说明不同后端的适用场景与切换开关（ServerArgs 字段）", "指出 metadata 构造是易错点（坑）"],
  "attention-backends"))

# 波次2
T.append((9, "deep-dive/frontend-language.md", 2200, True,
  ["python/sglang/lang/__init__.py", "python/sglang/lang/backend/", "python/sglang/lang/program.py",
   "python/sglang/lang/compiler.py", "python/sglang/lang/interpreter.py"],
  ["SGLang 前端 DSL 提供了哪些原语（gen、select、fork、image、+ 拼接等）",
   "程序如何表示：IR/Program 结构；编译执行 vs 解释执行两条路径",
   "状态与并行原语：fork/join、分支、变量绑定如何实现",
   "前端如何与后端 srt 交互（提交请求、获取结果）"],
  ["若 lang 目录结构不同，用 search_content 找真实文件后据实写（带锚点）", "用 mermaid 表达一次 DSL 程序的执行/调度（含 fork 并行）",
   "给出关键类/函数签名（如 Program、Interpreter.run）", "说明编译与解释各自的取舍"],
  "frontend-language"))

T.append((10, "deep-dive/server-entrypoint.md", 2400, True,
  ["python/sglang/srt/entrypoints/http_server.py", "python/sglang/srt/entrypoints/http_server_engine.py",
   "python/sglang/srt/entrypoints/engine.py", "python/sglang/srt/server_args.py",
   "python/sglang/srt/arg_groups/arg_utils.py", "python/sglang/srt/entrypoints/openai/"],
  ["HTTP 服务入口：http_server 如何路由请求、如何与 Engine 交互",
   "API 兼容层：OpenAI / Anthropic / Ollama 兼容入口如何映射到内部请求",
   "ServerArgs 参数体系：dataclass 结构、与 argparse 的关系、默认值来源",
   "Engine 封装：Engine 如何聚合 TokenizerManager/Scheduler/DetokenizerManager"],
  ["用 mermaid 画 http_server → engine → managers 的关系", "给出 ServerArgs 关键字段分组（模型、并行、缓存、采样）与真实签名锚点",
   "说明一个 /v1/chat/completions 请求如何被翻译成内部 Req", "指出参数校验与冲突处理（坑）"],
  "server-entrypoint"))

T.append((11, "deep-dive/tokenizer-detokenizer.md", 2000, True,
  ["python/sglang/srt/managers/tokenizer_manager.py", "python/sglang/srt/managers/detokenizer_manager.py"],
  ["TokenizerManager 职责：请求预处理、tokenize、管理异步请求状态、与 scheduler 通信",
   "DetokenizerManager 职责：把 scheduler 返回的 token id 流解码成文本并回流",
   "进程间通信：两者与 Scheduler 之间用 ZMQ 的哪种 socket（PUSH/PULL/REQ/REP），消息格式",
   "为何 tokenizer/detokenizer 要独立成进程（避免阻塞 GPU 线程）"],
  ["用 mermaid 表达三者 ZMQ 通信拓扑与消息流向", "给出 TokenizerManager.tokenize/abort 等关键方法签名与锚点",
   "说明流式输出时 detokenizer 如何处理部分 token（含特殊 token/空格）", "指出多 tokenizer / 多模态 token 的坑"],
  "tokenizer-detokenizer"))

T.append((12, "deep-dive/model-impl.md", 2600, True,
  ["python/sglang/srt/models/llama.py", "python/sglang/srt/models/deepseek_v3.py",
   "python/sglang/srt/model_loader/weight_utils.py", "python/sglang/srt/model_loader/__init__.py"],
  ["模型接入规范：一个模型类需要继承什么基类、实现哪些方法（forward、load_weights 等）",
   "以 Llama（或等价基础模型）逐行讲：模型结构、每层组成、如何对接 RadixAttention",
   "以 DeepSeek-V3（MoE/MLA）为例讲差异：MoE 专家层、MLA 注意力、共享专家",
   "权重加载：HuggingFace 权重如何映射到模型参数（load_weights 的 key 映射、量化权重处理）"],
  ["用 mermaid 表达模型类继承/组合关系（真实类名）", "给出模型 forward / load_weights 真实签名与锚点",
   "逐行解释 Llama 一层的关键代码（带行号）", "说明 MoE 的专家并行（EP）在模型层如何体现（指向 parallelism 文档主题）"],
  "model-impl"))

T.append((13, "deep-dive/parallelism.md", 2400, True,
  ["python/sglang/srt/distributed/__init__.py", "python/sglang/srt/distributed/parallel_state.py",
   "python/sglang/srt/distributed/device_communicators/", "python/sglang/srt/eplb/__init__.py",
   "python/sglang/srt/eplb/eplb_algorithms/"],
  ["TP/PP/DP/EP 四种并行的含义与在 sglang 中的实现位置",
   "通信原语：基于哪些库（pynccl/torch.distributed）、GroupCoordinator 如何管理进程组",
   "专家并行（EP）与 MoE：专家如何分片、all-to-all 通信；EPLB 负载均衡算法的作用",
   "并行配置如何与 ServerArgs（tp_size/pp_size/dp_size/ep_size）联动"],
  ["用 mermaid 表达 TP/PP/DP/EP 的数据/模型切分示意", "给出分布式初始化关键函数签名与锚点",
   "说明 EPLB 如何决定专家到 rank 的映射（指向代码）", "指出 EP 与 radix 缓存/通信重叠的坑"],
  "parallelism"))

T.append((14, "deep-dive/quantization.md", 2200, True,
  ["python/sglang/srt/layers/quantization/__init__.py", "python/sglang/srt/layers/quantization/fp8.py",
   "python/sglang/srt/layers/quantization/awq.py", "python/sglang/srt/layers/quantization/gptq.py",
   "python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py"],
  ["支持的量化方案：FP8 / AWQ / GPTQ / INT4 / FP4-KV 等，在代码里真实存在哪些（grep 核实）",
   "量化方法的注册与选择：如何根据权重/配置挑选 method，如何挂到 Linear/Attention 层",
   "kernel 选择：不同量化对应哪些计算 kernel（如 FP8 GEMM）",
   "KV cache 量化（FP8/FP4 KV）的特殊处理"],
  ["用表格列出量化方案 | 代码类 | 适用层 | 锚点", "画出量化层替换流程（mermaid）",
   "给出量化 method 的关键方法签名与锚点", "指出权重加载时反量化的坑"],
  "quantization"))

T.append((15, "dataflow/sequence-diagrams.md", 2000, True,
  ["python/sglang/srt/managers/scheduler.py", "python/sglang/srt/model_executor/model_runner.py",
   "python/sglang/srt/mem_cache/radix_cache.py"],
  ["关键路径时序图集合：prefill 全流程、decode 步、radix 缓存命中、抢占与恢复",
   "每张图必须对应真实函数调用序列（带锚点说明）"],
  ["提供 ≥4 个 mermaid sequenceDiagram：①纯 prefill ②decode 循环 ③cache hit（前缀复用跳过 prefill）④抢占+恢复",
   "每张图 participant 用真实类名、消息标注关键函数[:行号]", "说明各路径触发的 scheduler 决策点"],
  "sequence-diagrams"))

T.append((16, "deep-dive/observability.md", 1800, True,
  ["python/sglang/srt/observability/metrics_collector.py", "python/sglang/srt/observability/req_time_stats.py",
   "python/sglang/srt/observability/metrics_exporter.py"],
  ["指标（metrics）：暴露哪些关键指标（吞吐、时延、队列长度、KV 利用率、batch 大小等）",
   "日志：关键路径日志点", "profiling：是否提供 torch profiler / 自研 profiling 工具",
   "benchmark 工具链：benchmark/ 下有哪些脚本（grep 核实）"],
  ["用表格列出主要指标 | 含义 | 代码锚点", "说明如何开启 metrics（端口/路径）", "指出前缀命中率、抢占次数等诊断指标的价值"],
  "observability"))

# 波次3
T.append((17, "deep-dive/constrained-decoding.md", 2000, True,
  ["python/sglang/srt/constrained/__init__.py", "python/sglang/srt/constrained/grammar.py",
   "python/sglang/srt/constrained/constraint.py", "python/sglang/srt/constrained/xgrammar.py"],
  ["约束解码解决什么问题：按 JSON schema / 正则 / 语法限制输出",
   "集成方式：xgrammar / outlines 如何接入（grep 核实实际用了哪个）",
   "约束如何与采样/投票结合：在 logits 层面如何 mask 非法 token",
   "约束状态机：如何增量维护解析状态"],
  ["用 mermaid 表达约束解码数据流（token → mask → 采样）", "给出约束核心类/方法签名与锚点",
   "说明 JSON schema 模式下如何保证合法且高效（tokens 预查）", "指出与投机解码/并行采样的冲突坑"],
  "constrained-decoding"))

T.append((18, "deep-dive/speculative-decoding.md", 2000, True,
  ["python/sglang/srt/speculative/__init__.py", "python/sglang/srt/speculative/eagle.py",
   "python/sglang/srt/speculative/spec_info.py", "python/sglang/srt/speculative/draft_worker.py"],
  ["投机解码思想：用草稿模型/草稿头一次预测多 token，再由目标模型并行验证",
   "EAGLE 等实现：草稿如何生成、验证如何并行（tree attention）",
   "SpecInfo / draft 结构：草稿 token 树如何表示、如何拼接进 batch",
   "接受/拒绝准则：如何根据概率比接受"],
  ["用 mermaid 表达投机解码的草稿-验证循环", "给出 SpecInfo / draft 相关关键结构签名与锚点",
   "说明投机与 radix 缓存、CUDA Graph 的协同（坑）", "指出接受率低时的退化成本"],
  "speculative-decoding"))

T.append((19, "deep-dive/sampling.md", 1800, True,
  ["python/sglang/srt/sampling/sampling_params.py", "python/sglang/srt/sampling/sampling_batch.py",
   "python/sglang/srt/sampling/penaltylib/", "python/sglang/srt/sampling/logit_processors.py"],
  ["SamplingParams 参数体系：temperature/top_p/top_k/min_p/repetition_penalty 等",
   "logits processor：如何在采样前修改 logits（penalty、logit bias、bad words）",
   "penalty 实现：frequency/presence penalty 如何增量统计",
   "采样执行：在 batch 内如何对多请求高效采样（vectorized）"],
  ["用表格列出采样参数 | 含义 | 代码锚点", "画出采样流程（mermaid：logits→penalty→softmax→sample）",
   "给出 SamplingParams 关键字段与锚点", "指出 penalty 与约束解码/投机解码的耦合坑"],
  "sampling"))

T.append((20, "deep-dive/lora-multimodal.md", 2000, True,
  ["python/sglang/srt/lora/manager.py", "python/sglang/srt/lora/layer.py",
   "python/sglang/srt/multimodal/__init__.py", "python/sglang/srt/multimodal/processors/"],
  ["LoRA：适配器如何加载、如何合并/旁路到 base 层、多 LoRA 批内切换",
   "多模态：图像/音频输入如何预处理、如何注入模型（placeholder token、encoder 缓存）",
   "多模态数据在 Req 中的表示与调度影响"],
  ["用 mermaid 表达 LoRA 旁路 / 多模态注入流程", "给出 LoRAManager / MultimodalData 关键方法签名与锚点",
   "说明多 LoRA 与 batch 内不同适配器的隔离（坑）", "指出多模态与 radix 前缀复用的冲突"],
  "lora-multimodal"))

T.append((21, "deep-dive/disaggregation.md", 2200, True,
  ["python/sglang/srt/disaggregation/__init__.py", "python/sglang/srt/disaggregation/base/",
   "python/sglang/srt/disaggregation/prefill.py", "python/sglang/srt/disaggregation/decode.py",
   "python/sglang/srt/disaggregation/connector/", "python/sglang/srt/disaggregation/nixl/",
   "python/sglang/srt/disaggregation/mooncake/"],
  ["PD 分离（Prefill/Decode Disaggregation）架构：prefill 与 decode 拆到不同实例",
   "KV 传输：prefill 完成后 KV cache 如何传输到 decode 实例（connector：nixl/mooncake 等）",
   "router / 负载均衡：请求如何在 prefill 与 decode 集群间路由",
   "一致性：decode 端如何等待 KV 到达"],
  ["用 mermaid 画 PD 分离架构与 KV 传输路径", "给出 PrefillWorker/DecodeWorker 关键方法签名与锚点",
   "说明传输的同步/异步与可能的阻塞点（坑）", "指出 PD 分离下 radix 缓存的边界"],
  "disaggregation"))

T.append((22, "quickstart/install.md", 900, False,
  ["README.md", "docker/Dockerfile", "pyproject.toml", "setup.py", "python/requirements.txt"],
  ["安装方式：pip 安装（editable）、Docker 镜像构建、源码编译",
   "依赖与编译：CUDA 版本、torch、flashinfer 等对版本的要求",
   "验证安装：如何确认 sglang 可导入、可用"],
  ["给出的每条安装/编译命令必须来自源码或 CI（grep scripts/ci、docker/Dockerfile、README 核实），不得编造",
   "区分『已核实命令』与『通用建议』", "列出最小依赖与可选组件（如多模态、量化 kernel）"],
  "install"))

T.append((23, "quickstart/minimal-example.md", 900, False,
  ["python/sglang/launch_server.py", "python/sglang/cli/", "examples/usage/", "test/registered/"],
  ["最小可跑服务端启动命令（模型路径、端口、关键参数）",
   "最小客户端调用（OpenAI 兼容或 sglang native）",
   "常见本地模型来源（HF 缓存路径）与离线运行"],
  ["启动命令需从 launch_server.py 的 argparse 与 examples/test 中核实", "给出 curl / python 客户端最小示例（可运行）",
   "说明服务起来后如何确认健康（/health 等端点）"],
  "minimal-example"))

T.append((24, "quickstart/e2e-observation.md", 1000, False,
  ["python/sglang/srt/managers/tokenizer_manager.py", "python/sglang/srt/managers/scheduler.py",
   "python/sglang/srt/model_executor/model_runner.py", "python/sglang/srt/managers/detokenizer_manager.py",
   "python/sglang/srt/observability/metrics_collector.py"],
  ["一次真实请求从发起到返回的端到端观测", "在哪些阶段可以加日志/打印来跟踪（关键函数与行号）",
   "返回的日志/指标里能看到哪些有用信息（时延分解、batch 大小、前缀命中）"],
  ["给出一条真实 curl 请求 + 预期的日志片段（引用代码中实际的 logger/print 点锚点）", "说明如何打开详细日志（verbose/环境变量）",
   "给出 metrics 端点查看示例"],
  "e2e-observation"))

# 波次4
T.append((25, "hacking/dev-setup.md", 1000, False,
  ["README.md", "pyproject.toml", ".pre-commit-config.yaml", "scripts/lint/", "scripts/ci/"],
  ["开发环境搭建：可编辑安装、pre-commit、代码风格（ruff/black/isort）",
   "测试布局：test/registered 与 test/manual 的区别、如何运行单个测试",
   "debug 技巧：常用打印点、attach 调试器、环境变量开关",
   "常用环境变量清单：grep `os.getenv`/`os.environ` 于 srt 目录，列出真实存在的变量与含义"],
  ["环境变量表须来自源码 grep 结果（变量名 | 作用 | 默认 | 锚点），禁止编造", "给出运行 lint / 单个测试的真实命令",
   "列出已知 debug 打印/断点建议（指向真实函数）"],
  "dev-setup"))

T.append((26, "hacking/add-a-model.md", 1400, False,
  ["python/sglang/srt/models/llama.py", "python/sglang/srt/models/registry.py",
   "python/sglang/srt/model_loader/weight_utils.py", "python/sglang/srt/model_loader/__init__.py"],
  ["模型注册机制：模型类如何被注册/被发现（MODEL_REGISTRY / get_model / config 映射）",
   "新增模型的最小步骤：实现哪些方法、如何声明支持的 config、如何映射权重",
   "权重加载关键点：config.json 字段、state_dict key 映射、量化权重"],
  ["给出注册相关真实函数/装饰器签名与锚点（grep 'register'/'MODEL_REGISTRY' 核实）",
   "用有序步骤清单写出『手把手新增一个类 Llama 的模型』", "指出常见踩坑（config 字段、rotary、lm_head 绑定）"],
  "add-a-model"))

T.append((27, "hacking/add-a-kernel-backend.md", 1400, False,
  ["python/sglang/srt/layers/attention/attention_registry.py",
   "python/sglang/srt/layers/attention/base_attn_backend.py",
   "python/sglang/srt/layers/attention/flashinfer_backend.py"],
  ["注意力后端注册机制：如何把一个新 backend 注册进 registry",
   "Backend 基类要实现哪些接口（init_metadata、forward、metadata 结构）",
   "如何被 scheduler/runner 选中（选择逻辑、ServerArgs 开关）"],
  ["给出 AttentionBackend 基类关键抽象方法签名与锚点", "用有序步骤写出『新增一个 dummy backend』需要改哪些文件",
   "指出 metadata 构造与不同 forward_mode 的坑"],
  "add-a-kernel-backend"))

T.append((28, "hacking/reading-guide.md", 1000, False,
  ["python/sglang/launch_server.py", "python/sglang/srt/managers/scheduler.py",
   "python/sglang/srt/model_executor/model_runner.py", "python/sglang/srt/mem_cache/radix_cache.py"],
  ["推荐阅读顺序（从入口到核心，循序渐进）", "每个关键文件建议关注的入口函数/类",
   "断点与打印建议：在哪些函数设断点能最快理解一次请求", "如何用小模型/单请求做最小复现"],
  ["给出分阶段的阅读路线（新手→进阶→专家）", "对 scheduler/model_runner/radix_cache 给出具体 breakpoint 建议（函数名+锚点）",
   "列出最小可调试启动参数（如 tp=1、单请求）"],
  "reading-guide"))

T.append((29, "appendix/glossary.md", 1000, False,
  ["python/sglang/srt/mem_cache/radix_cache.py", "python/sglang/srt/managers/scheduler.py",
   "python/sglang/srt/distributed/parallel_state.py", "python/sglang/srt/speculative/"],
  ["术语表：RadixAttention、Chunked Prefill、Continuous Batching、TP/PP/DP/EP、MoE、EAGLE、CUDA Graph、KV Cache、Prefix Cache 等",
   "每个术语给出一句话定义 + 在 sglang 中的对应实现锚点"],
  ["用表格：术语 | 定义 | 代码锚点", "覆盖 ≥20 个核心术语", "定义必须准确且对应源码"],
  "glossary"))

T.append((30, "appendix/config-reference.md", 1800, False,
  ["python/sglang/srt/server_args.py", "python/sglang/srt/arg_groups/arg_utils.py"],
  ["ServerArgs 全量字段参考：字段名、类型、默认值、含义、影响模块",
   "环境变量参考：与 ServerArgs 互补或覆盖的环境变量（grep getenv）"],
  ["用大表格列出 ServerArgs 主要字段 | 类型 | 默认 | 说明 | 锚点（按主题分组：模型/并行/缓存/采样/日志）",
   "环境变量表来自 grep 结果", "标注哪些字段互相关联/互斥（坑）"],
  "config-reference"))

T.append((31, "appendix/open-questions.md", 800, False,
  ["docs/appendix/_openq_scheduler.md", "docs/appendix/_openq_memory-pool.md", "docs/appendix/_openq_radix-cache.md",
   "docs/appendix/_openq_model-runner.md", "docs/appendix/_openq_attention-backends.md",
   "docs/appendix/_openq_frontend-language.md", "docs/appendix/_openq_server-entrypoint.md",
   "docs/appendix/_openq_tokenizer-detokenizer.md", "docs/appendix/_openq_model-impl.md",
   "docs/appendix/_openq_parallelism.md", "docs/appendix/_openq_quantization.md",
   "docs/appendix/_openq_constrained-decoding.md", "docs/appendix/_openq_speculative-decoding.md",
   "docs/appendix/_openq_sampling.md", "docs/appendix/_openq_lora-multimodal.md",
   "docs/appendix/_openq_disaggregation.md", "docs/appendix/_openq_observability.md",
   "docs/appendix/_openq_overview.md", "docs/appendix/_openq_request-lifecycle.md",
   "docs/appendix/_openq_key-data-structures.md", "docs/appendix/_openq_sequence-diagrams.md",
   "docs/appendix/_openq_install.md", "docs/appendix/_openq_minimal-example.md",
   "docs/appendix/_openq_e2e-observation.md", "docs/appendix/_openq_dev-setup.md",
   "docs/appendix/_openq_add-a-model.md", "docs/appendix/_openq_add-a-kernel-backend.md",
   "docs/appendix/_openq_reading-guide.md", "docs/appendix/_openq_glossary.md",
   "docs/appendix/_openq_config-reference.md"],
  ["本文件是待验证/未解疑问的汇总。由你（最后一个任务）整合其他任务产生的 _openq_*.md",
   "若某些 _openq_*.md 尚不存在（其他任务未产生疑问），忽略即可"],
  ["先 Read 所有存在的 docs/appendix/_openq_*.md", "把内容按『所属模块』归类整理为 open-questions.md（替换 TODO 占位）",
   "每条问题保留：模块 | 问题描述 | 可能的验证方向", "若无任何 _openq 文件，则写『暂无未解疑问』并说明"],
  "open-questions"))

T.append((32, "appendix/changelog-of-docs.md", 600, False,
  ["docs/index.md"],
  ["文档站自身的变更日志：记录各篇文档的完成进度与对应源码 commit", "本任务只初始化结构，后续由主会话维护"],
  ["初始化变更日志，列出已规划的文档清单与状态（对照 PROGRESS.md）", "记录所对齐的源码 commit",
   "说明更新规范（每完成一篇追加一条）"],
  "changelog-of-docs"))

if __name__ == "__main__":
    for t in T:
        task(*t)
    print(f"\nGenerated {len(T)} prompt files.")
