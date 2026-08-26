# OPEN Questions: request-lifecycle

### 多 tokenizer 模式下的回程路径是否与单进程一致
在 `request-lifecycle.md` 的"边界与坑 #1"中标注。当 `tokenizer_worker_num > 1` 时：
- `TokenizerManager.init_ipc_channels`（`python/sglang/srt/managers/tokenizer_manager.py:534-560`）改用 `tokenizer_worker_ipc_name`，且 `DetokenizerManager.send_to_tokenizer` 被绕过（`python/sglang/srt/managers/detokenizer_manager.py:116-120`）。
- 具体分发/聚合由 `MultiTokenizerRouter` 与 `TokenizerWorker` 承担（见 `python/sglang/srt/managers/multi_tokenizer_mixin.py`），本文未展开其回程机制。

可能的方向：需核对 `MultiHttpWorkerDetokenizerMixin`（`detokenizer_manager.py` 的基类）与 `multi_http_worker_event_loop` 的实现，确认 `BatchStrOutput` 是否经 `SocketMapping` 直接按 `http_worker_ipc` 回推到对应 worker，而非走单一 `send_to_tokenizer` socket；并确认其与单 tokenizer 模式在 `rid_to_state` 状态槽写回逻辑上是否等价。
