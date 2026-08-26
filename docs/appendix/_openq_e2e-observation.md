# Open Questions — e2e-observation

> 本文档记录 e2e-observation 文档中遗留的未决/不确定点。请勿删除，除非已解决并回填正文。
> 关联正文：docs/quickstart/e2e-observation.md

### 1. `_wait_one_response` 的流式 chunk 还原与多 tokenizer worker 返回路径

正文在 Why/How 中仅写到 `TokenizerManager.generate_request` 通过 `async for response in self._wait_one_response(obj, request)` 把 Scheduler→Detokenizer→回传的 `BatchStrOutput` 逐步 yield 给客户端（证据：`python/sglang/srt/managers/tokenizer_manager.py:807`）。但以下两点未深入阅读源码确认：

- `_wait_one_response` 内部如何把 `BatchStrOutput` 还原成 SSE/JSON 流式 chunk（增量 delta 的切片逻辑、`last_output_offset` 如何推进）。
- 当 `server_args.tokenizer_worker_num > 1` 时，DetokenizerManager 通过 `SocketMapping` 直接把结果 push 回对应 HTTP worker（见 `detokenizer_manager.py:116-L122` 注释 "results are pushed back to each TokenizerWorker directly via SocketMapping"），这一分支与单 worker 的 `send_to_tokenizer` PUSH 路径差异未展开。

可能的方向：直接读 `TokenizerManager._wait_one_response` 与 `MultiHttpWorkerDetokenizerMixin.multi_http_worker_event_loop`（detokenizer_manager.py:530-L531）补全这两条路径。

### 2. `--log-requests-level` 与 `enable_request_time_stats_logging` 易混淆

正文已在"边界与坑"第 1 条说明二者不同（前者控制请求"内容"日志，后者控制"时延分解"日志）。但二者在 `server_args.py` 中同属 `NS("observability")`，且都默认不打印，用户很容易以为开了 `--log-requests` 就能看到时延分解。

可能的方向：在正式文档（或 server_args 帮助文本）中并列展示两者差异；也可确认是否在 CLI 帮助里补充交叉提示。相关定义锚点：`python/sglang/srt/server_args.py:1473-L1493`（`log_requests` / `log_requests_level`）与 `python/sglang/srt/server_args.py:1602-L1604`（`enable_request_time_stats_logging`）。
