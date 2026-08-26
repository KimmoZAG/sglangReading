# 评阅报告 batch 2

> 评阅人视角：资深 LLM 推理引擎架构师（严格代码文档审查）
> 唯一事实来源（SSOT）：`/home/kimmo/develop/sglang`，commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`
> 评阅方式：纯静态、只读。所有锚点均在该 commit 源码中逐条核实，未修改任何 `docs/` 或 SSOT 文件。
> 评阅对象（8 篇 deep-dive 文档）：model-runner / attention-backends / frontend-language / server-entrypoint / tokenizer-detokenizer / model-impl / parallelism / quantization

---

## 总体结论

**综合评分：9 / 10**

八篇文档整体质量极高：结构统一（What / Why / How / 边界与坑）、源码锚点覆盖率高且绝大多数精准、Mermaid 图节点名经 `class`/`def` 全量核实均为真实 SSOT 符号、对缺失文件（如 `program.py`/`compiler.py`、`deepseek_v3.py`、`run_tokenizer_process`）与未确认行为均以 `[OPEN]` 透明标注，无填空式废话段落。

**最严重问题（一句话）**：仅 3 处行号越界（`arg_utils.py:L338`、`kv_cache.py:L86`、`fp8.py:L2717` 指向了超过文件末尾的行）与若干「裸文件名 / 相对路径」锚点（如裸 `awq_kernels.py`、`model_runner.py`、`__init__.py`、`utils.py`、`communication_op.py`）在 srt 树下存在同名/近名文件，机械解析器会命中错误文件而误报越界——应一律改为 `python/sglang/srt/...` 完整相对路径。

> 说明：下方「锚点核验」中的 `漂移`/`伪造` 计数为**机械解析器**直接产出（即一个严格按 `basename→首命中` 规则实现的 checker 会报出的数字）。其中 model-runner 的 3 个 `漂移` 与 quantization 的 4 个 `漂移`（`__init__.py`、`awq_kernels.py`）为**误报**——解析器错误命中了 `mlx/model_runner.py`、`python/sglang/__init__.py`、`hardware_backend/cpu/...`；文档意图文件（`model_executor/model_runner.py` 2112 行、`layers/quantization/__init__.py` 173 行、`hardware_backend/gpu/.../awq_kernels.py` 255 行）行号均有效。真实行号越界仅 3 处（见 ISSUES 1–3）。`伪造/路径错` 主要反映「路径风格不统一 / 裸文件名歧义」这一系统性问题，意图文件本身行号有效。

---

## 逐篇

### 1. model-runner.md
- **质量评分：9.0**
- **锚点核验**：总数 93 ｜ 机械 OK=81 ｜ 漂移=3（误报，见上）｜ 伪造/歧义=9（裸 `model_runner.py` 撞 `hardware_backend/mlx/model_runner.py`）｜ 占位=0
- **深度问题**：无。§0 总览、CUDA Graph 四类 Runner（`EagerRunner`/`DecodeCudaGraphRunner`/`PrefillCudaGraphRunner`/`CPUGraphRunner`）与 `_forward_raw`/eager 分支、TC-piecewise 等覆盖完整，What/Why/How/边界齐备。
- **mermaid 问题**：1 个 flowchart（ModelRunner→forward→Graph replay 分支），节点 `ModelRunner`/`ModelRunnerOutput`/`True` 均为真实符号/关键字，无误。

### 2. attention-backends.md
- **质量评分：9.5**
- **锚点核验**：总数 113 ｜ OK=110 ｜ 漂移=0 ｜ 伪造/歧义=0 ｜ 占位=3
- **深度问题**：无。后端抽象、注册表/工厂、`build_attention_backends` 流水线、各后端要点、元数据构建「最易错四环节」、RadixAttention 上层入口均扎实；§2.5 与 §3.6 对「双写一致性」「静默回退」的坑点抓得准。
- **mermaid 问题**：3 图（flowchart 选型、flowchart 流水线、classDiagram 继承）节点名经 grep 全部为真实类（`AttentionBackend`/`FlashInferAttnBackend`/`FlashMLABackend`/`CutlassMLABackend`/`HybridAttnBackend` 等）。仅 3 处锚点用 `Lxxx+` 非精确写法（见 ISSUES 10）。

### 3. frontend-language.md
- **质量评分：9.5**
- **锚点核验**：总数 52 ｜ OK=51 ｜ 漂移=0 ｜ 伪造/歧义=1（裸 `api.py:102-L108` 撞 `kv_canary/api.py` 等，但意图 `lang/api.py` 行号有效）｜ 占位=0
- **深度问题**：无。开篇即诚实指出任务预设的 `program.py`/`compiler.py` 在本 commit 不存在、程序实体实为 `ir.py:SglFunction`，「编译」由 `tracer.py` 承担——这是高质量的事实核对。解释/追踪双路径、fork/join 快照语义、`select` 低温约束等边界清晰。
- **mermaid 问题**：graph + sequenceDiagram 节点 `BaseBackend`/`RuntimeEndpoint`/`StreamExecutor`/`SglFunction`/`TracerProgramState`/`SglGen` 等均经全树 grep 确认为真实类（`lang/` 包下）。无误。

### 4. server-entrypoint.md
- **质量评分：8.5**（唯一含真实行号越界的文档）
- **锚点核验**：总数 81 ｜ OK=40 ｜ 漂移=1（`arg_utils.py:L218-L338`，文件 337 行，L338 越界，真实）｜ 伪造/歧义=40（大量相对路径 `server_args.py`/`http_server.py`/`engine.py` 未带 `python/sglang/srt/` 前缀，与 `multimodal_gen` 同名文件重名，意图 srt 版行号有效）｜ 占位=0
- **深度问题**：无。三层职责（FastAPI / Engine / ServerArgs）、进程模型、`check_server_args` 硬约束、Rust server 偏离等均到位。
- **mermaid 问题**：1 个 flowchart，节点为名（`FastAPI`/`OpenAIServingBase`/`AnthropicServing`/`OllamaServing`/`TokenizerManager`/`DetokenizerManager` 等），均为真实类/模块。无误。

### 5. tokenizer-detokenizer.md
- **质量评分：9.5**（本批次锚点最干净的一篇）
- **锚点核验**：总数 36 ｜ OK=36 ｜ 漂移=0 ｜ 伪造/歧义=0 ｜ 占位=0（全部使用完整 `python/sglang/srt/...` 路径）
- **深度问题**：无。ZMQ PUSH/PULL 拓扑、DecodeStatus 增量差量、`�` 半字符处理、`_clamp_decode_ids`、多 tokenizer 路由一致性等细节精准；对 `trim_matched_stop` 多 stop 串、TokenizerManager 启动入口未确认处也以 `[OPEN]` 标注。
- **mermaid 问题**：1 个 graph（ZMQ 拓扑），节点为真实方法/类名。无误。

### 6. model-impl.md
- **质量评分：9.5**
- **锚点核验**：总数 37 ｜ OK=37 ｜ 漂移=0 ｜ 伪造/歧义=0 ｜ 占位=0（少数相对路径如 `llama.py`/`weight_utils.py` 在 srt 下虽有多候选，但本篇均带 `python/sglang/srt/...` 完整前缀，无歧义）
- **深度问题**：无。开篇诚实标注 `deepseek_v3.py` 在本 commit 不存在、V3 实为 `DeepseekV2ForCausalLM` 子类；Llama/DeepSeek 分层、MLA 双 RadixAttention、MoE/EP、权重加载三轴均清晰。
- **mermaid 问题**：classDiagram（Llama/Deepseek 真实类与组合关系）+ flowchart（权重加载），类名经 grep 全部为真实（`DeepseekV2ForCausalLM`/`DeepseekV2AttentionMLA`/`FusedMoE`/`MoEGate`/`ColumnParallelLinear` 等）。无误。

### 7. parallelism.md
- **质量评分：9.0**
- **锚点核验**：总数 51 ｜ OK=48 ｜ 漂移=0 ｜ 伪造/歧义=3（裸 `communication_op.py`/`parallel_state.py` 等撞 `multimodal_gen` 同名文件；意图 srt 版行号有效，其中 `communication_op.py` 为 srt 与 multimodal_gen 同包重名，最需改全路径）｜ 占位=0
- **深度问题**：无。TP/PP/DP/EP/MoE-DP/ATTN-TP/CP/DCP 八维、进程组构建、`GroupCoordinator` 后端路由、EPLB 全流程、与 ServerArgs 联动均完整；§6 坑点（EPLB 与 radix 缓存失效耦合）以 `[OPEN]` 诚实标注。
- **mermaid 问题**：graph + sequenceDiagram + graph（EPLB），节点 `GroupCoordinator`/`EPLBManager`/`EplbAlgorithm`/`ExpertLocationUpdater` 等经 grep 均为真实符号。无误。

### 8. quantization.md
- **质量评分：8.0**（系统性路径风格问题 + 真实越界最多）
- **锚点核验**：总数 126 ｜ 机械 OK=109 ｜ 漂移=7（其中 2 真实：`kv_cache.py:L86`、`fp8.py:L2717`；4 误报：`__init__.py`→`python/sglang/__init__.py`、`awq_kernels.py`→cpu 版，意图文件行号有效）｜ 伪造/歧义=10+（裸 `awq_kernels.py`/`__init__.py`/`utils.py`/`fp8.py`/`mxfp4.py`/`modelopt_quant.py`/`fp8_utils.py` 等与 `multimodal_gen` 同名，意图 srt 版）｜ 占位=0
- **深度问题**：无。三轴独立量化、注册表四层、`get_quant_method` 分派、`process_weights_after_loading` 双重调用/反量化陷阱、AWQ 非 Marlin 不省算力、NVFP4 的 ×6 scale 修正等均极扎实，是本批次信息密度最高的一篇。
- **mermaid 问题**：flowchart + sequenceDiagram。节点名真实（`Fp8LinearMethod`/`AWQLinearMethod`/`KVCacheConfigurator`/`NVFP4KVCacheMethod` 等）。**唯一瑕疵**：时序图 `participant TRTLLMHAAttnBackend as TRTLLMMHAAttnBackend`（:270）显示别名双写 M，真实类名为 `TRTLLMHAAttnBackend`（见 ISSUES 11）。

---

## ISSUES

> 格式：`###I` 分隔；每行 `FILE | SEVERITY | TYPE | DETAIL | SUGGESTED_FIX`
> SEVERITY ∈ {high, medium, low}；TYPE ∈ {anchor_drift, anchor_fake, anchor_placeholder, shallow, mermaid_fake, missing_section}

###I server-entrypoint.md | high | anchor_drift | 锚点 `python/sglang/srt/arg_groups/arg_utils.py:L218-L338`（:40）指向的文件实际为 337 行，L338 越界（真实定义 `add_cli_args_from_dataclass` 在 L218 起，止于 L337）。 | 改为 `python/sglang/srt/arg_groups/arg_utils.py:L218-L337`。

###I quantization.md | high | anchor_drift | 锚点 `python/sglang/srt/layers/quantization/kv_cache.py:L18-L86`（:21）指向的文件实际为 85 行，L86 越界。 | 改为 `.../kv_cache.py:L18-L85`（或 `L18-L84` 视 `BaseKVCacheMethod` 实际结束行）。

###I quantization.md | high | anchor_drift | 锚点 `fp8.py:L2710-L2717`（:34，裸名，意图 `python/sglang/srt/layers/quantization/fp8.py`）该文件为 2716 行，L2717 越界。 | 改为 `python/sglang/srt/layers/quantization/fp8.py:L2710-L2716` 并使用完整路径。

###I quantization.md | medium | anchor_fake | 裸名 `awq_kernels.py:L88-L105`（:329）与 `:32-L74`（:193）与同文档 :188 已用的完整路径 `hardware_backend/gpu/quantization/awq_kernels.py:L88-L105` 混用；`awq_kernels.py` 在 srt 下有 cpu/gpu/npu 三份，裸名机械解析会命中 99 行的 cpu 文件而误报越界，真实意图为 255 行的 gpu 文件。 | 统一改写为 `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`。

###I model-runner.md | medium | anchor_fake | 多处裸名 `model_runner.py`（如 :1517-L1604 / :1654-L1752 / :997-L1005 / :1421-L1454 等，共 9 处）与 `python/sglang/srt/hardware_backend/mlx/model_runner.py`（1705 行）重名；机械解析器会命中 mlx 版并误报 L1752 越界，真实意图为 `python/sglang/srt/model_executor/model_runner.py`（2112 行）。 | 全部改为完整路径 `python/sglang/srt/model_executor/model_runner.py`。

###I quantization.md | medium | anchor_fake | 多处裸名 `__init__.py`（:72-L101 / :133-L141 / :158 / :173）歧义，机械解析命中 `python/sglang/__init__.py`（124 行）误报越界；意图为 `python/sglang/srt/layers/quantization/__init__.py`（173 行，L173 有效，注册表 `BASE_QUANTIZATION_METHODS` 在 L72-L101）。 | 改为完整路径 `python/sglang/srt/layers/quantization/__init__.py`。

###I quantization.md | medium | anchor_fake | 多处裸名 `utils.py`（:128-L133 / :164-L189 / :176-L181 / :298-L328 等）歧义（仓库存在多份 `utils.py`），意图为 `python/sglang/srt/layers/quantization/utils.py`（`requantize_with_max_scale`/`get_linear_quant_method` 等）。 | 改为完整路径 `python/sglang/srt/layers/quantization/utils.py`。

###I parallelism.md | medium | anchor_fake | 裸名 `communication_op.py:L18-L20` 与 `:105-L107`（:116、:119）与 `python/sglang/srt/distributed/communication_op.py`（107 行，行号有效）及 `python/sglang/multimodal_gen/runtime/distributed/communication_op.py` 同包重名。 | 改为完整路径 `python/sglang/srt/distributed/communication_op.py`。

###I server-entrypoint.md | low | anchor_fake | 约 40 处相对路径锚点（`server_args.py`/`http_server.py`/`engine.py` 等）未带 `python/sglang/srt/` 前缀，与 `multimodal_gen` 同名文件重名；上下文意图明确为 srt 版且行号有效，但机械解析器无法唯一确定。 | 统一加前缀 `python/sglang/srt/`，与 tokenizer-detokenizer.md 等保持一致。

###I attention-backends.md | low | anchor_placeholder | 三处非精确锚点 `flashattention_backend.py:L670+`（:231）、`flashattention_backend.py:L1750+`（:231）、`radix_attention.py:L150+`（:312）用 `+` 表示「约略」，不利精确定位。 | 给出具体行号区间（如 `L670-L760`）。

###I quantization.md | low | mermaid_fake | 时序图 `participant TRTLLMHAAttnBackend as TRTLLMMHAAttnBackend`（:270）显示别名拼写为双 M `TRTLLMMHAAttnBackend`，真实类名为 `TRTLLMHAAttnBackend`（`python/sglang/srt/layers/attention/trtllm_mha_backend.py:96`）。 | 修正别名为 `TRTLLMHAAttnBackend`（去掉多余 M）。

###I quantization.md | low | anchor_fake | 路径风格不统一：同文档同时出现完整路径 `python/sglang/srt/...`（如 :20-22）与相对路径 `layers/quantization/fp8.py`（如 :34、:64），且部分相对路径在 srt 下存在 `multimodal_gen` 同源同名副本。 | 全仓库统一使用 `python/sglang/srt/...` 完整相对路径一种风格。

---

## TOP5（按 SEVERITY 排序的最紧迫修复）

1. **server-entrypoint.md `arg_utils.py:L218-L338` → L337**（high / anchor_drift）：行号越界，指向不存在的 L338，证据锚点失效。
2. **quantization.md `kv_cache.py:L18-L86` → L85**（high / anchor_drift）：行号越界，L86 超出 85 行文件。
3. **quantization.md `fp8.py:L2710-L2717` → L2716**（high / anchor_drift）：行号越界，L2717 超出 2716 行文件。
4. **quantization.md 裸名 `awq_kernels.py` → 完整 gpu 路径**（medium / anchor_fake）：裸名在 srt 下 cpu/gpu/npu 三选一，机械解析必然命中 99 行 cpu 文件而误报越界，是最易导致「锚点看上去全错」的系统性陷阱。
5. **model-runner.md 裸名 `model_runner.py` → 完整 `model_executor/model_runner.py`**（medium / anchor_fake）：裸名撞 `mlx/model_runner.py`，机械解析误报 L1752 越界（真实文件 2112 行），同批 9 处均受影响。

> 备注：第 4、5 项的「假漂移」会连带污染任何自动化锚点检查器的统计结果，建议优先统一路径风格（ISSUES 8/9/12）以根除。
