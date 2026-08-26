# Open Questions: tokenizer-detokenizer

> 本文件为 `docs/deep-dive/tokenizer-detokenizer.md` 的待澄清问题清单，由子任务写作时根据源码阅读记录。请勿直接编辑 `open-questions.md`（避免并发冲突）。

### Q1: 多 stop string 同时命中时的裁剪语义
`DetokenizerManager.trim_matched_stop` 当前只读取 `finished_reason.get("matched", None)` 的**单个**匹配项进行裁剪。源码 `python/sglang/srt/managers/detokenizer_manager.py:186` 有注释 TODO：`handle the case where multiple stop strs are hit`。当一次生成同时命中多个 stop string 时，究竟保留/裁剪哪一个、按何种优先级，需结合 `Scheduler` 侧 `finished_reason` 的构造逻辑进一步确认。

### Q2: TokenizerManager 进程启动入口位置
与 `DetokenizerManager` 明确存在 `run_detokenizer_process`（`python/sglang/srt/managers/detokenizer_manager.py:515`）不同，在必读的两个文件（`tokenizer_manager.py` / `detokenizer_manager.py`）中未找到等价的主进程启动函数与 `setproctitle("sglang::tokenizer")` 调用。推测 `TokenizerManager` 由引擎装配层（如 `sglang/srt/entrypoints/*` 或 `Engine`）负责 spawn，并用 `auto_create_handle_loop`（`python/sglang/srt/managers/tokenizer_manager.py:2175`）拉起 asyncio 事件循环。确切入口与进程名设置位置待在 `entrypoints` 侧源码确认。

### Q3: 多 detokenizer 下 decode_status 跨进程一致性的极端情况
`MultiDetokenizerRouter` 用 `zlib.crc32(http_worker_ipc) % num_workers` 静态钉选（`python/sglang/srt/managers/multi_tokenizer_mixin.py:572-573`）。若运行期 `num_workers` 变化（弹性扩缩容）或某 worker 崩溃重建导致 `http_worker_ipc` 重新分配，同一 rid 可能被路由到不同 detokenizer，使 `decode_status` 状态缺失。该边界是否在调度层有保护（如要求停服后再扩缩）待进一步确认。
