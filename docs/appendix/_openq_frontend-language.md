# Open Questions — frontend-language

> 本文件记录对 SGLang 前端 DSL（lang 模块）源码阅读中仍存在不确定、或多义性的点。
> 请勿直接修改 open-questions.md（避免并发冲突），新问题追加到本文件即可。

### 1. `position_ids_offset` 与 `Backend.fork_program` 是否真的生效？

- 现象：`ProgramState.fork(size, position_ids_offset)`（`python/sglang/lang/interpreter.py:L888-L896`）与 `StreamExecutor.fork(size, position_ids_offset)`（`python/sglang/lang/lang/interpreter.py:L370-L402`）都接收 `position_ids_offset`，但 `StreamExecutor.__init__` 根本没有该参数，fork 体里也只是把它透传给并未存在的构造参数（实际忽略）。
- 同时 `BaseBackend.fork_program`（`python/sglang/lang/backend/base_backend.py:L38-L44`）在整个 `python/sglang/` 中**没有任何调用方**（grep 仅命中定义处）。
- 可能方向：
  1. 当前 commit 的 fork 是纯客户端快照式实现——fork 时把父 `StreamExecutor` 的 `variables/text_/messages_` 复制给子执行器，后端只看到彼此独立的 `/generate` 请求；`position_ids_offset` 与 `fork_program` 可能是早期/未完成的多卡 KV 共享方案的遗留接口，目前是 dead code。
  2. 也可能 `position_ids_offset` 仅在某些非 `RuntimeEndpoint` 后端（如未来分布式后端）中才会被消费，而 `RuntimeEndpoint` 路径确实忽略它。
- 需要确认：在 `RuntimeEndpoint` 路径下，fork 子请求之间是否真的共享 KV（除 `concate_and_append` 模式外）。若是纯独立请求，则 `position_ids_offset` 对单活体 server 无实际效果。

### 2. `concate_and_append` 模式下父状态在 fork 与 join 之间能否继续生成？

- 现象：`_execute_concatenate_and_append_kv_cache`（`python/sglang/lang/interpreter.py:L738-L752`）在合并每个子状态前断言 `exe.fork_start_text_pos == self_len`，其中 `self_len = len(self.text_)` 是父执行器的当前文本长度，`fork_start_text_pos` 在 fork 时记录为父彼时的 `len(self.text_)`（`python/sglang/lang/interpreter.py:L397`）。
- 可能方向：这意味着 `concate_and_append` 仅在「fork 后父状态不再继续推进文本」时才成立；若父在 fork 与 join 之间又 `+=` 了内容，断言会失败。
- 待确认：是否在文档层面要求用户「fork 后父仅做并行分支、join 前不再 add 文本」？还是该路径本身只服务于 demo（`test_programs.py:295` 用到 `mode="concate_and_append"`），生产不常用。

### 3. `compiler.py` / `program.py` 的去向

- 任务预设要求阅读 `python/sglang/lang/program.py` 与 `compiler.py`，但本 commit（`e1c4db9…`）的 `lang/` 目录实际不含这两个文件，也没有 `program.py`。
- 程序（Program）实体由 `ir.py` 中的 `SglFunction` 承担；所谓「编译」角色由 `tracer.py` 的 `trace_program`/`TracerProgramState` 承担（记录 IR 而非生成字节码）。需要确认：是否社区在某个版本将 `program.py` 重命名为 `ir.py` 并删除 `compiler.py`，或文档源对应了不同 commit。
