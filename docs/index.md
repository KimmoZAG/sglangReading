# SGLang 源码精读

> 一套「能让人从零到深入掌握 SGLang」的中文文档站。所有结论均来自本地源码阅读，禁止依赖记忆或网络传闻。
> 唯一事实来源（SSOT）：`/home/kimmo/develop/sglang`。

## 项目是什么

SGLang（**S**tructured **G**eneration **Lang**uage）是一个用于大规模语言模型（LLM）与多模态模型推理的高性能服务框架与前端 DSL。它在两个层面解决推理效率问题：

1. **前端 DSL / 运行时**（Python 包 `sglang` 的 `lang` 部分）：提供一套结构化生成原语（`gen`、`select`、`fork`、`image` 等），把「多轮调用 + 控制流 + 约束解码」收敛成一次可调度、可缓存的程序。
2. **后端推理引擎**（包路径 `sglang/srt`，*S*erver *R*untime）：一个多进程、多 GPU 的生产级推理服务器，核心能力包括 RadixAttention 前缀复用、Chunked Prefill、continuous batching、多种并行（TP/PP/DP/EP）、投机解码、量化、PD 分离等。

本站点聚焦**后端推理引擎 `srt`** 的代码级解读（也涵盖前端 DSL 的编译/执行链路），目标是让有 Transformer/PyTorch 基础、想读懂并二次开发 SGLang 的工程师，能够独立定位代码、理解设计权衡、动手加模型/后端/特性。

## 能力矩阵

| 能力 | 是否覆盖 | 关键模块（代码锚点） |
| --- | --- | --- |
| 结构化生成 DSL | ✅ | `python/sglang/lang/` |
| RadixAttention 前缀复用 | ✅ | `python/sglang/srt/mem_cache/radix_cache.py` |
| Continuous / chunked prefill | ✅ | `python/sglang/srt/managers/scheduler.py` |
| 多进程架构（Tokenizer/Scheduler/Worker） | ✅ | `python/sglang/srt/managers/` |
| 张量/流水/数据/专家并行 | ✅ | `python/sglang/srt/distributed/`、`eplb/` |
| 量化（FP8/AWQ/GPTQ/INT4/FP4KV 等） | ✅ | `python/sglang/srt/layers/quantization/` |
| 约束解码（JSON schema / 语法） | ✅ | `python/sglang/srt/constrained/` |
| 投机解码（EAGLE 等） | ✅ | `python/sglang/srt/speculative/` |
| LoRA | ✅ | `python/sglang/srt/lora/` |
| 多模态 | ✅ | `python/sglang/srt/multimodal/` |
| PD 分离 | ✅ | `python/sglang/srt/disaggregation/` |
| 可观测性（指标/日志/profiling） | ✅ | `python/sglang/srt/observability/` |

## 本文档对应版本

> 铁律 #4：所有论断必须先做版本对齐。本站内容对齐以下源码快照。

| 项 | 值 |
| --- | --- |
| 源码路径 | `/home/kimmo/develop/sglang` |
| Git Commit | `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7` |
| Commit 时间 | `2026-08-14 11:11:02 +0800` |
| `git describe` | `gateway-v0.3.1-7844-ge1c4db9621` |
| 版本号获取方式 | 动态：`sglang/_version.py` → `setuptools_scm` → `importlib.metadata` → 兜底 `0.0.0.dev0`（见 `python/sglang/version.py`） |
| Python 文件数 | 5496（`.py`） |
| 模型实现文件数 | 216（`python/sglang/srt/models/*.py`） |

> 注意：仓库根 `README.md` 标注的版本与 `git describe` 可能不一致（本仓同时含 `sgl-model-gateway` 等子项目）。文档以 **commit hash** 为准。

## 如何阅读本站

建议路径（详见 [阅读路线](hacking/reading-guide.md)）：

1. **先建立全貌**：[全局架构总览](architecture/overview.md) → [请求生命周期](architecture/request-lifecycle.md) → [目录与模块地图](architecture/directory-map.md)。
2. **吃透核心数据结构**：[核心数据结构](dataflow/key-data-structures.md)（Req / ScheduleBatch / ForwardBatch）。
3. **逐模块深潜**：按 `scheduler → memory-pool → radix-cache → model-runner → attention-backends → …` 顺序（见 [深潜模块](deep-dive/scheduler.md)）。
4. **动手验证**：[安装](quickstart/install.md) → [最小示例](quickstart/minimal-example.md) → [端到端观测](quickstart/e2e-observation.md)。
5. **二次开发**：[开发环境](hacking/dev-setup.md) → [新增模型](hacking/add-a-model.md) → [新增注意力后端](hacking/add-a-kernel-backend.md)。

## 文档约定（铁律速览）

- **代码优先**：结论必须来自本地源码，证据锚点形如 `python/sglang/srt/managers/scheduler.py:L120-L180`。
- **不确定就标注**：存疑处写入 [未解问题](appendix/open-questions.md)，不编造。
- **深度优先**：每节回答 What / Why（设计动机与权衡）/ How（关键代码路径）/ 边界与坑。
- **图必须有**：架构/流程/时序/状态机用 Mermaid，组件名与真实类名一致。

## 站点导航

- 架构总览：[architecture/overview.md](architecture/overview.md)
- 模块深潜索引：[deep-dive/scheduler.md](deep-dive/scheduler.md)
- 术语表：[appendix/glossary.md](appendix/glossary.md)
- 配置参考：[appendix/config-reference.md](appendix/config-reference.md)
