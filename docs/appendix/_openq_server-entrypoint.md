# Open Questions — server-entrypoint

本文件记录 `deep-dive/server-entrypoint.md` 写作过程中未完全确认、需要进一步追踪源码的问题。
请勿直接修改 `open-questions.md`（避免并发冲突），新问题追加到本文件即可。

---

### 多 tokenizer worker 与单 worker 的共享状态一致性边界

`_setup_and_run_http_server` 在 `tokenizer_worker_num > 1` 时改用
`uvicorn.run("sglang.srt.entrypoints.http_server:app", workers=N)`
（`python/sglang/srt/entrypoints/http_server.py:L2650-L2664`）。
每个 worker 是独立进程，通过共享内存 `multi_tokenizer_args_<main_pid>` 重建自己的
`TokenizerManager`（`init_multi_tokenizer`，`http_server.py:L216-L266`）。

**问题描述**：单 worker 模式把 `server_args` / `scheduler_info` 直接挂在 `app` 对象上
（`http_server.py:L2498-L2499`），由 lifespan 读取；多 worker 模式则每个 worker 进程各自
在 `init_multi_tokenizer` 中从 shm 重建 `_GlobalState`（`http_server.py:L258-L264`）。
两条路径在 `lifespan` 中的分支（`http_server.py:L274-L282`）对全局状态（如
`tokenizer_manager.max_req_input_len`、`startup_time`、模板）的初始化是否完全一致、
并发请求在多 worker 间如何路由（是否有 `MultiTokenizerRouter` 做分发），尚未完整确认。

**可能的方向**：
- 阅读 `sglang.srt.managers.multi_tokenizer_mixin`（`MultiTokenizerRouter`、
  `TokenizerWorker`、`read_from_shared_memory` / `write_data_for_multi_tokenizer`）
  确认 shm 协议与请求分发方式。
- 确认 `app_has_admin_force_endpoints` 与 API key 中间件为何仅在单 worker 模式挂载
  （`http_server.py:L2513-L2524` 的注释提到 multi-tokenizer 不支持 api_key 鉴权，
  见 `http_server.py:L231-L233` 的 assert）。

---

### serving 层 default_sampling_params 的来源与合并规则

`ServerArgs.sampling_defaults` 默认 `"model"`
（`python/sglang/srt/server_args.py:L1400-L1407`），文档称其决定默认采样参数取自模型
`generation_config.json` 还是 SGLang/OpenAI 默认（`server_args.py:L1403`）。
在 `/v1/chat/completions` 路径中，`OpenAIServingChat._convert_to_internal_request` 调用
`request.to_sampling_params(stop=..., model_generation_config=self.default_sampling_params, ...)`
（`python/sglang/srt/entrypoints/openai/serving_chat.py:L953-L958`）。

**问题描述**：`self.default_sampling_params` 在 `OpenAIServingChat` 中的初始化来源、
它与 `ServerArgs.sampling_defaults` / `preferred_sampling_params`
（`server_args.py:L1418-L1425`）之间的优先级关系，以及 per-request 的
`temperature` / `top_p` 如何与默认值合并（覆盖还是 supplement），未在本文档写作时完整追踪。

**可能的方向**：
- 在 `serving_chat.py` 中搜索 `default_sampling_params` 的赋值处，确认其来自
  `sampling_defaults` 还是 `preferred_sampling_params` 或模型 config。
- 阅读 `ChatCompletionRequest.to_sampling_params` 的实现（位于
  `python/sglang/srt/entrypoints/openai/protocol.py`），确认合并语义。
