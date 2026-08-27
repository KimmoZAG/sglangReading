# 评阅报告 R4-4

> 评审对象：9 篇 SGLang 源码文档（docs/ 下）
> 唯一事实来源 SSOT：`/home/kimmo/develop/sglang` @ `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`
> 方法：对每个 `path:La-Lb` 锚点，按规则解析路径→定位文中紧邻引用符号→grep/Read 核验真实定义/出现行是否落在 [La,Lb]。每个判定均经实测（grep/Read），未凭记忆。
> 路经解析规则：缺 `python/sglang/srt/` 前缀→按 `python/sglang/srt/<path>`；缺 `python/sglang/`→按 `python/sglang/<path>`；其余相对路径先试 SSOT 根、再试 srt/。

## 总体结论

**评分：7 / 10**

最严重问题（一句）：`quickstart/install.md` 有 **3 个整段伪造锚点**（把 `python/sglang/test/test_utils.py` 里的 `popen_launch_server` / `_wait_for_server_health` / `SGLANG_WAIT_PORT_TIMEOUT` 错标成 `scripts/ci/cuda/ci_install_dependency.sh` 的对应行号，连文件都不存在该符号），且 `hacking/dev-setup.md` 把调试器线索写错（`Scheduler.event_loop` 方法不存在、Dumper 环境变量前缀实为 `DUMPER_` 而非 `SGLANG_DUMP_`），会直接误导排障。

整体锚点质量高（deep-dive、add-a-model、reading-guide、e2e-observation、minimal-example 几乎全 OK），失分集中在 `install.md`、`dev-setup.md`、`add-a-kernel-backend.md` 三篇的「符号/路径错位」类错误。

---

## 逐篇

### docs/deep-dive/tokenizer-detokenizer.md
- 评分: 9/10
- 锚点符号准确性: OK=43  漂移=0  伪造=0  占位=0
- 深度问题: 无凑字数段落；What/Why(权衡：序列化开销 vs 批量解码)/How/坑 齐全；两处 `[OPEN]` 为诚实标注。
- mermaid问题: 无虚构类名。participant（`recv_from_detokenizer`/`send_to_scheduler` 等）、router（`MultiTokenizerRouter`/`MultiDetokenizerRouter`/`SocketMapping`）均为真实符号。
- 备注: 仅 1 个承重锚点 SYMBOL_MISMATCH（见 ISSUES I-01）。

### docs/hacking/add-a-kernel-backend.md
- 评分: 6/10
- 锚点符号准确性: OK≈58  漂移=0  伪造=0  占位=0（另有 ~12 个缺目录前缀 + 1 SYMBOL_MISMATCH + 1 畸形区间）
- 深度问题:
  - 第 139 行链路描述漏掉 `_build_backend_from_str`（setup:236），并暗示 `attn_backend_wrapper` 仅 hybrid 才套——实际**每条路径都套**（`attn_backend_wrapper` 自身就是模型级包装，见 setup:202/239）。
  - 第 215 行把 hybrid 组合顺序写反：源码是 `attn_backend_wrapper(HybridAttnBackend(...))`（先建 Hybrid 再包一次），并非「先 wrapper 再做模型级包装」。
  - 第 213 行把 triton 断言写成 `assert not is_encoder_decoder`，真实代码为 `assert not runner.model_config.is_encoder_decoder`（且是 assert 非工厂阶段 raise）。
  - 第 36 行 `FlasInfer` 拼写错误（应为 FlashInfer）。
- mermaid问题: 图 1 `RES --> BUILD` 边暗示 `build_attention_backends` 调用 `resolve_attention_backend_strs`，实际 `resolve` 在调用 `build` 之前已完成（setup:75-82 注释明确「已在 runner 上盖章」）；图 2 把 `_build_resolved_backend`/`_build_backend_from_str` 折叠进 `_build_full_attention_backend_from_str` participant，且函数签名写成位置参数（实为 keyword-only `*, model_runner`）。

### docs/hacking/add-a-model.md
- 评分: 9/10
- 锚点符号准确性: OK=32  漂移=0  伪造=0  占位=0
- 深度问题: 无凑字数；What/Why/How/坑 齐全。
- mermaid问题: 无虚构类；`MODEL_REGISTRY` 为文档自造概念标签（已显式映射到 `ModelRegistry`），可接受。
- 备注: 行 117 把 `RemapRegistry` 当真实注册表引用，真实符号是 `_REMAP_REGISTRY`/`get_weight_remap`（仅注释中出现），属「名虚构/概念名」，低危，建议改指真实符号。

### docs/hacking/dev-setup.md
- 评分: 6/10
- 锚点符号准确性: OK=67  漂移=1  伪造=1(图目录)  占位=0（另 SYMBOL_MISMATCH=4，缺前缀若干）
- 深度问题:
  - 第 123 行「tokenizer/detokenizer 边界」bullet 无锚点、无具体断点，为低信息段。
  - 第 200 行坑 4 与第 58 行坑 1 重复（CUDA_HOME），无独立锚点。
  - 第 147/157 行纯过渡句，无新增信息。
- mermaid问题:
  - 图 2（L83-95）节点 `test/registered/jit/` **虚构目录**：仓库无 `test/registered/jit/`，JIT kernel 测试实际在 `test/registered/kernels/`（上游 `test/README.md:75-77` 自身过时，本文照抄）。→【图类名虚构】
  - 图 1（L11-20）边 `python/sglang/srt --> rust/` 结构错误：Rust 扩展由 `python/setup.py` 从 `../rust`（python 的兄弟目录）发现，非 srt 子目录。
  - 图 4（L204-213）节点 `B[pip install -e python[test]]` 中未转义的 `[`/`]` 会导致 mermaid 解析失败，须写成 `B["pip install -e python[test]"]`。

### docs/hacking/reading-guide.md
- 评分: 9/10
- 锚点符号准确性: OK=56  漂移=0  伪造=0  占位=1
- 深度问题: 交叉参考段（L147-151）为泛指引，无新锚点，但非错误。
- mermaid问题: 全部 flowchart 实体（run_server / dispatch_event_loop / event_loop_overlap / get_next_batch_to_run / ModelRunner.forward / _forward_raw / RadixCache.match_prefix 等）均真实存在，无虚构。
- 备注: 唯一瑕疵为 L55 的 `scheduler.py:388-478+`，`+` 为占位符（见 ISSUES I-02）。其余承重锚点（含大量单点行号）实测均命中真实符号。

### docs/index.md
- 评分: 8/10
- 锚点符号准确性: OK（commit/time/describe/文件数/目录全部实测吻合）；仅版本链顺序 DRIFT=1
- 深度问题: 本篇为落地/总览页，无行号代码锚点，本身无需深核；元数据表全部准确。
- mermaid问题: 无。
- 备注: 第 42 行「版本号获取方式」把 `importlib.metadata` 与 `setuptools_scm` 的**顺序写反**（真实：`sglang/_version.py` → `importlib.metadata` → `setuptools_scm` → `0.0.0.dev0` 兜底）。`0.0.0.dev0` 兜底值与 `version.py` 存在性本身正确。

### docs/quickstart/e2e-observation.md
- 评分: 9/10
- 锚点符号准确性: OK=71  漂移=0  伪造=0  占位=0（另有多处缺 `python/sglang/srt/` 前缀——按规则可解析，建议补全）
- 深度问题: 无凑字数；What/Why/How/坑 齐全，两处 `[OPEN]` 诚实。
- mermaid问题: 无虚构；participant 与调用函数（`request_logger.log_received_request`/`ModelRunner.forward`/`_decode_batch_token_id_output`/`collect_metrics`→`observe_*`/`log_time_stats`）均真实。
- 备注: `observability/trace_async.py:16` 实为模块 docstring，真实 `SGLANG_TRACE_ASYNC` 读取在 `trace_async.py:234`；`detokenizer_manager.py:430-L487` 方法真实止于 488（差 1 行）。均为轻微。

### docs/quickstart/install.md
- 评分: 6/10
- 锚点符号准确性: OK≈27  漂移=1  伪造=3  占位=0（另 SYMBOL_MISMATCH=1）
- 深度问题: 大量无锚点背景叙述（L7、L41-48、L100-102、L131-133），与首行「所有结论均来自逐行阅读」的承诺略有张力；但被引用的版本号/行号事实准确。
- mermaid问题: 图（L17-35）flowchart 语法正确，节点 ID 全部先定义后使用，可正常渲染。
- 备注: 3 个 FAKE 锚点（I-03/I-04/I-05）+ 1 DRIFT（I-06）+ 1 SYMBOL_MISMATCH（I-07）。9 个版本号（torch==2.13.0、transformers==5.12.1、flashinfer_python[cu13]==0.6.17、flash-attn-4>=4.0.0b18、sglang-kernel==0.4.6.post1、sgl-deep-gemm==0.1.5.post2、sgl-deep-ep==0.1.0、nvidia-cutlass-dsl[cu13]==4.6.2、xgrammar==0.2.1）、Rust channel 1.92、CUDA_VERSION=13.0.3 全部实测吻合。

### docs/quickstart/minimal-example.md
- 评分: 9/10
- 锚点符号准确性: OK=69  漂移=0  伪造=0  占位=0
- 深度问题: 无凑字数；入口/Why/How/坑 10 条齐全。
- mermaid问题: 两图实体（`cli.main.main`/`cli.serve.serve`/`launch_server.run_server`/`Engine._launch_subprocesses`/`_wait_and_warmup` 等）均真实，无虚构。
- 备注: 轻微——L76-83 把 CI sanity 用例参数「转述」成含 `--host 0.0.0.0/--port 30000/--log-level info`（真实 `test_basic_sanity.py:L43-L58` 只有 `--cuda-graph-max-bs-decode 4` + `--mem-fraction-static 0.7` + `--enable-metrics`）；`io_struct.py:L160-L176` 的 `sampling_params` 实际在 L208（略超范围）；`test_utils.py:L403-L458` 写 `HF_HUB_OFFLINE="1"` 实际在 L488。均非错误。

---

## ISSUES（###I 分隔）

###I
FILE: docs/deep-dive/tokenizer-detokenizer.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: 文中第 57 行锚点 `python/sglang/srt/managers/detokenizer_manager.py:399-426`，紧邻引用符号为 `multi_http_worker_event_loop`（声称「结果改由 multi_http_worker_event_loop 经 SocketMapping 直接扇出」）。实测：`detokenizer_manager.py:399-409` 是 `_decode_batch_token_id_output` 的尾部，`411-428` 是 `_b64_encode_per_request`；**`multi_http_worker_event_loop` 的真实定义位于 `python/sglang/srt/managers/multi_tokenizer_mixin.py:399`**（类 `MultiHttpWorkerDetokenizerMixin`）。该文中 `:116-122` 关于 send_to_tokenizer 不创建的论述本身正确，但 `:399-426` 这一锚点指错了文件。
SUGGESTED_FIX: 将 `python/sglang/srt/managers/detokenizer_manager.py:399-426` 改为 `python/sglang/srt/managers/multi_tokenizer_mixin.py:399`（并在该处核对 `SocketMapping` 扇出逻辑），或删去该锚点、仅保留 `:116-122`。

###I
FILE: docs/hacking/reading-guide.md
SEVERITY: low
TYPE: anchor_placeholder
DETAIL: 第 55 行锚点 `python/sglang/srt/managers/scheduler.py:388-478+`，`+` 为占位符，无真实结束行。与其「锚点精确到行号」的自我要求相悖，且不可跳转核对。
SUGGESTED_FIX: 用真实范围替换，例如 `python/sglang/srt/managers/scheduler.py:388-485`（以 `def dispatch_event_loop` 之前的最后一个 `init_*` 调用为界；请用 grep 确认 `__init__` 实际结束行后填入）。

###I
FILE: docs/quickstart/install.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: 第 44 行锚点 `scripts/ci/cuda/ci_install_dependency.sh:L744-L756`，文中称是 `popen_launch_server` 构造的命令数组（`["sglang","serve",...]`）。实测：`ci_install_dependency.sh:744-756` 是 `install_test_tools()`（kernels 下载 + human-eval 克隆），**不含 `popen_launch_server`**。`popen_launch_server` 真实定义在 `python/sglang/test/test_utils.py:668`。
SUGGESTED_FIX: 改为 `python/sglang/test/test_utils.py:668`（或对应行），并确认 `popen_launch_server` 内 `command` 数组首项为 `["sglang","serve",...]`。

###I
FILE: docs/quickstart/install.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: 第 211 行锚点 `scripts/ci/cuda/ci_install_dependency.sh:L620-L665`，文中称是就绪判定 `_wait_for_server_health`（轮询 `/health_generate`）。实测：`ci_install_dependency.sh:620-629` 是 `download_flashinfer_cache()`，`631-680` 是 `stabilize_flashinfer_jit_paths()`，**无 `_wait_for_server_health`**。该符号真实位于 `python/sglang/test/test_utils.py:620`（且 minimal-example.md 第 211 行已正确引用 `test_utils.py:L620-L665`）。
SUGGESTED_FIX: 改为 `python/sglang/test/test_utils.py:620-L665`。

###I
FILE: docs/quickstart/install.md
SEVERITY: high
TYPE: anchor_fake
DETAIL: 第 222 行锚点 `scripts/ci/cuda/ci_install_dependency.sh:L713-L716`，文中称是 `SGLANG_WAIT_PORT_TIMEOUT=120`。实测：`ci_install_dependency.sh:713-716` 是 `install_extra_deps` 的 nixl 块；**无 `WAIT_PORT_TIMEOUT`**。真实为 `python/sglang/test/test_utils.py:713-L716`，且变量名是 **`SGLANG_WAIT_PORT_TIMEOUT="120"`**（文中漏了 `SGLANG_` 前缀）。
SUGGESTED_FIX: 改为 `python/sglang/test/test_utils.py:713-L716`，并把正文变量名更正为 `SGLANG_WAIT_PORT_TIMEOUT`。

###I
FILE: docs/quickstart/install.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: 第 214 行锚点 `python/pyproject.toml:L5-L14`，文中称含「包名、requires-python、**构建后端**」。实测：`build-backend="setuptools.build_meta"` 在 `L3`；L5 是 `[project]` 起始，L6 为 `name="sglang"`，L10 为 `requires-python=">=3.10"`，L5-L14 不含构建后端声明。
SUGGESTED_FIX: 改为 `python/pyproject.toml:L3-L14`（或拆成 `L3` 为 build-backend、`L5-L14` 为其余）。

###I
FILE: docs/quickstart/install.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: 第 219 行锚点 `python/setup.py:L152-L174`，文中称是 `_declared_rust_extensions()`（含 `SGLANG_BUILD_RUST_EXTS=none` 短路）。实测：`setup.py:152-174` 是 **`_selected_rust_extensions(declared)`**（对 declared 列表做 none/all/逗号过滤）；真正的 `_declared_rust_extensions()` 在 `L177-L182`（含 `if ... =="none": return []`）。
SUGGESTED_FIX: 若为「过滤逻辑」用 `python/setup.py:L152-L174`；若为「=none 短路」用 `python/setup.py:L177-L182`。两处讲的不是同一个函数，需对应修正。

###I
FILE: docs/hacking/dev-setup.md
SEVERITY: high
TYPE: symbol_mismatch
DETAIL: 第 124 行「`debug_utils/dumper.py` 的 `Dumper` 类读取 `SGLANG_DUMP_*` 系列环境变量」。实测：`dumper.py` **没有 `class Dumper`**（配置类为 `DumperConfig`@132，运行时类 `_Dumper`@224）；环境变量前缀是 **`DUMPER_`**，并非 `SGLANG_DUMP_`（`dumper.py:167-169` `_env_prefix()` 显式返回 `"DUMPER_"`，注释写明「should not be `SGLANG_DUMPER_`」）。读者据此设 `SGLANG_DUMP_*` 不会生效。
SUGGESTED_FIX: 改为「`dumper.py` 的 `DumperConfig`/`_Dumper` 读取 `DUMPER_*` 环境变量（如 `DUMPER_ENABLE`）」，锚点 `python/sglang/srt/debug_utils/dumper.py:167-169`（前缀定义）、`:239`（`DUMPER_ENABLE=1` 用法示例）。

###I
FILE: docs/hacking/dev-setup.md
SEVERITY: high
TYPE: symbol_mismatch
DETAIL: 第 121 行「`Scheduler` 的 `event_loop` / `run_batch` 是请求处理核心，断点可放 `python/sglang/srt/managers/scheduler.py`」。实测：`scheduler.py` **不存在 `event_loop` 方法**（仅有 `event_loop_normal`@1714 与 `event_loop_overlap`@1749）；`run_batch`@3623 真实存在。把断点设在「`event_loop`」会落空。
SUGGESTED_FIX: 改为「`Scheduler.event_loop_normal` / `event_loop_overlap`（二选一，取决于 overlap 开关）与 `run_batch`（`python/sglang/srt/managers/scheduler.py:1714` / `:1749` / `:3623`）」。

###I
FILE: docs/hacking/dev-setup.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: 第 122 行「`ModelRunner` 的 `forward` / `forward_decode` 是每层实际计算入口」。实测：`ModelRunner.forward`@1510 存在，但 **`forward_decode` 不在 `ModelRunner`**（它是注意力后端基类 `AttentionBackend.forward_decode`，见 `python/sglang/srt/layers/attention/base_attn_backend.py:261`）。且「每层实际计算入口」粒度也不准——`ModelRunner.forward` 是 per-batch。
SUGGESTED_FIX: 改为「`ModelRunner.forward`（`python/sglang/srt/model_executor/model_runner.py:1510`）负责整批前向；逐层 decode 计算入口在注意力后端 `AttentionBackend.forward_decode`（`python/sglang/srt/layers/attention/base_attn_backend.py:261`）」。

###I
FILE: docs/hacking/dev-setup.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: 第 33 行「`python/pyproject.toml` 用 setuptools-rust + setuptools-scm 自动发现 `rust/` 工作区里的扩展模块（见 python/pyproject.toml:1-3, 242-247）」。实测：`pyproject.toml:242-247` 是 `[tool.setuptools_scm]`（root/version_file/git_describe_command/fallback_version），**与 Rust 发现无关**；Rust 工作区发现逻辑在 `:249-252` 注释 + `python/setup.py:46`(`_RUST_WORKSPACE_DIR`)/`:104`(`_discovered_rust_extensions`)。
SUGGESTED_FIX: 把行号改为 `python/pyproject.toml:249-252` + `python/setup.py:46,104`（或仅保留 setup.py 两条）。

###I
FILE: docs/hacking/dev-setup.md
SEVERITY: medium
TYPE: anchor_drift
DETAIL: 第 70 行锚点 `.pre-commit-config.yaml:1-188`。实测该文件共 **187 行**，`:1-188` 超出 EOF 1 行。
SUGGESTED_FIX: 改为 `.pre-commit-config.yaml:1-187`。

###I
FILE: docs/hacking/dev-setup.md
SEVERITY: medium
TYPE: mermaid_fake
DETAIL: 第 83-95 行 mermaid 图中节点 `RJ[test/registered/jit/<br/>JIT kernel 测试]` 引用的 `test/registered/jit/` **目录在 SSOT 中不存在**（grep `test/registered/jit*` 无结果）。JIT kernel 测试实际位于 `test/registered/kernels/`（如 `test/registered/kernels/test_kernel_inventory.py`）。
SUGGESTED_FIX: 将 `RJ` 节点改为 `test/registered/kernels/`，或注明「JIT kernel 测试见 `test/registered/kernels/`」。

###I
FILE: docs/hacking/add-a-kernel-backend.md
SEVERITY: high
TYPE: missing_prefix
DETAIL: 全文多处将 `attention_backend_setup.py` 写作**裸文件名**（如 L145/L147/L191/L201/L215 的 `attention_backend_setup.py:L155-L176` / `:L67-L140` / `:L249-L255` / `:L252-L253` / `:L255`），但 SSOT 中该文件**仅存在于** `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py`（裸 `python/sglang/srt/attention_backend_setup.py` 不存在）。按本任务解析规则，裸文件名无法解析到正确文件；且同篇 L149/L199/L224 又用了完整路径，内部不一致。行号本身（155-176 等）经核验是正确的，属纯路径前缀缺陷。
SUGGESTED_FIX: 统一补全为 `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:Lxx-Lyy`。同时修正 L149/L199 中畸形的 `...attention_backend_setup.py:179-L222`（缺前导 `L`）为 `:L179-L222`。

###I
FILE: docs/hacking/add-a-kernel-backend.md
SEVERITY: medium
TYPE: symbol_mismatch
DETAIL: 第 135 行锚点 `python/sglang/srt/layers/attention/flashinfer_backend.py:L1414-L1416`，文中称经 `self.forward_metadata.prefill_wrappers[...]` 读取。实测：该区间是 `forward_decode`（@1405）内的 `self.forward_metadata.decode_wrappers[self._get_wrapper_idx(layer)]`（L1414-1416），**是 `decode_wrappers` 而非 `prefill_wrappers`**。`prefill_wrappers` 出现在 `forward_extend`@1244 的 L1253-L1255。
SUGGESTED_FIX: 将 `:L1414-L1416` 改为 `decode_wrappers`（或把锚点改到 `flashinfer_backend.py:L1253-L1255` 以匹配 `prefill_wrappers`），避免一段文字把两个不同 wrapper 都指向同一锚点。

###I
FILE: docs/hacking/dev-setup.md
SEVERITY: low
TYPE: missing_prefix
DETAIL: 多处锚点缺 `python/sglang/srt/` 前缀（按规则可解析，但建议补全以消除歧义）：`model_runner.py:389`→`python/sglang/srt/model_executor/model_runner.py:389`；`environ.py:272/326/328/332/362-369/413-416/440/442/450/464`→`python/sglang/srt/environ.py:...`；`install.mdx:74`/`install.mdx:55-58`/`install.mdx:28-36`→`docs/docs/get-started/install.mdx:...`。
SUGGESTED_FIX: 按上补全相对 SSOT 的完整路径。

###I
FILE: docs/quickstart/e2e-observation.md
SEVERITY: low
TYPE: missing_prefix
DETAIL: 大量锚点缺 `python/sglang/srt/` 前缀（如 `server_args.py:1467`、`tokenizer_manager.py:792`、各 `schedule_batch.py`/`req_time_stats.py`/`metrics_collector.py`/`request_logger.py`/`output_streamer.py`/`trace.py`/`trace_async.py` 行号）。按规则可解析到 srt 下对应文件，但建议统一补全部缀以与 SSOT 路径风格一致。
SUGGESTED_FIX: 为所有裸文件名补 `python/sglang/srt/<相对目录>/` 前缀。

###I
FILE: docs/index.md
SEVERITY: low
TYPE: anchor_drift
DETAIL: 第 42 行「版本号获取方式：sglang/_version.py → setuptools_scm → importlib.metadata → 0.0.0.dev0」。实测 `python/sglang/version.py` 解析顺序为：`sglang._version`（try）→ `importlib.metadata`（except ImportError）→ `setuptools_scm`（except Exception）→ `0.0.0.dev0` 兜底。文中把 `importlib.metadata` 与 `setuptools_scm` **顺序写反**。（`sglang/_version.py` 为构建期生成文件，源码树中不存在，作为首步仅概念正确。）
SUGGESTED_FIX: 改为「`sglang._version`（构建生成）→ `importlib.metadata` → `setuptools_scm` → 兜底 `0.0.0.dev0`」。

###I
FILE: docs/hacking/add-a-model.md
SEVERITY: low
TYPE: symbol_mismatch
DETAIL: 第 117 行 `[OPEN]` 将 `RemapRegistry` 作为真实注册表引用（「`AutoWeightsLoader` + `RemapRegistry`」「各模型的 `RemapRegistry` 注册细节」）。实测 SSOT 中**无 `class RemapRegistry` / 注册 API**，仅 `auto_loader.py:20`、`llama.py:744` 注释出现该词；真实机制是 `_REMAP_REGISTRY`（`auto_loader.py:177`）+ `get_weight_remap`。属「名虚构（概念名）」。
SUGGESTED_FIX: 将 `RemapRegistry` 更正为 `get_weight_remap` / `_REMAP_REGISTRY`，或明确标注其为概念名。

---

## TOP8（最该优先修的 8 条）

1. **install.md 三处 FAKE 锚点（I-03/I-04/I-05，high）**：把 `test_utils.py` 的 `popen_launch_server`/`_wait_for_server_health`/`SGLANG_WAIT_PORT_TIMEOUT` 错标到 `ci_install_dependency.sh` 的错误行号（连文件都无该符号，变量名还漏 `SGLANG_`）。会直接把读者引到错误文件。→ 改指 `python/sglang/test/test_utils.py` 对应行，并修正变量名。
2. **dev-setup.md Dumper 前缀错误（I-08，high）**：`SGLANG_DUMP_*` 实为 `DUMPER_`，无 `class Dumper`。照做设环境变量无效，排障踩坑。→ 改为 `DUMPER_*`，锚点 `dumper.py:167-169`/`:239`。
3. **dev-setup.md `Scheduler.event_loop` 不存在（I-09，high）**：仅有 `event_loop_normal`/`event_loop_overlap`。照搬设断点会落空。→ 改为两个真实方法 + `run_batch`（scheduler.py:1714/1749/3623）。
4. **add-a-kernel-backend.md `attention_backend_setup.py` 缺目录前缀（I-13，high）**：裸文件名无法解析（真实路径含 `model_executor/model_runner_components/`），且与同篇完整路径写法不一致。→ 全篇统一补全。
5. **tokenizer-detokenizer.md `:399-426` 指错文件（I-01，medium）**：`multi_http_worker_event_loop` 真实在 `multi_tokenizer_mixin.py:399`，detok 同区间是 `_decode_batch_token_id_output` 尾部。→ 改锚到 mixin。
6. **add-a-kernel-backend.md flashinfer `:1414-L1416` decode/prefill 混淆（I-14，medium）**：该区间是 `decode_wrappers`，文中称 `prefill_wrappers`。→ 锚点改 `:1253-L1255` 或改描述。
7. **dev-setup.md `pyproject.toml:242-247` Rust 发现错位（I-11，medium）**：该区间是 `[tool.setuptools_scm]`，与 Rust 无关；真实在 `:249-252`+`setup.py:46,104`。→ 改行号。
8. **dev-setup.md mermaid 虚构目录 `test/registered/jit/`（I-12，medium）**：该目录不存在，真实为 `test/registered/kernels/`。→ 改节点路径。

> 次优先（low，建议一并处理）：index.md 版本链顺序（I-16）、reading-guide.md 占位符 `:388-478+`（I-02）、e2e-observation.md / dev-setup.md 缺前缀（I-15、I-17）、add-a-model.md `RemapRegistry` 概念名（I-18）、install.md `pyproject.toml:5-14` 构建后端漂移（I-06）与 `setup.py:152-174` 函数名错位（I-07）、dev-setup.md `.pre-commit-config.yaml:1-188` 越界（I-10）。
