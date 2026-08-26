# Open Questions — reading-guide

本文件记录 `docs/hacking/reading-guide.md` 中遗留的不确定点。不要在 `open-questions.md` 中直接追加，避免并发冲突。

### `run_scheduler_process` 与 http_server 的进程边界如何划分？

文档正文中提到 `launch_server.py` 在 HTTP 模式下调用 `sglang.srt.entrypoints.http_server.launch_server`（python/sglang/launch_server.py:50-52），并锚定了 `run_scheduler_process`（python/sglang/srt/managers/scheduler.py:4990），但未深入二者之间的进程/线程拓扑。

可能方向：
- `http_server.launch_server` 可能通过 `multiprocessing`/`torch.multiprocessing` 拉起 `run_scheduler_process`，同时另起 `TokenizerManager`、`RequestDispatcher` 等进程/线程。
- 进程间通信可能经由 `io_struct.py` 中定义的 `*`/`*ReqInput`/`*ReqOutput` 结构 + ZMQ/共享内存通道。
- 建议下一步阅读 `python/sglang/srt/entrypoints/http_server.py` 与 `python/sglang/srt/managers/io_struct.py`，确认每个组件是独立进程还是同一进程内的线程，以及它们如何与 `Scheduler` 的 `request_receiver`（python/sglang/srt/managers/scheduler.py:2007 的 `init_request_receiver`）对接。
