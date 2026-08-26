# Open Questions: overview（架构总览）

### ModelRunner 是否在某些部署形态下是独立 OS 进程？

题面把 `Worker(ModelRunner)` 列为与 TokenizerManager / Scheduler / DetokenizerManager 平级的独立进程，但默认部署下 `ModelRunner` 通过 `TpModelWorker` 持有并运行在 Scheduler 子进程内部（证据：`python/sglang/srt/managers/tp_worker.py:466` 的 `self._model_runner = ModelRunner(...)`；`python/sglang/srt/managers/scheduler.py:1018` 的 `self.model_worker = self.tp_worker`）。

可能的方向：
- 在 disaggregation 模式（prefill/decode 分离）或启用 Rust server 时，GPU 计算被独立的 server/进程接管（证据：`python/sglang/srt/entrypoints/engine.py:2003` 的 `self.recv_from_tokenizer = rust_server`）。
- 历史版本曾存在独立 `model_worker` 进程；需确认当前 SSOT 是否仍保留该代码路径，或已完全收敛为进程内 worker。
- 应基于源码实测，而非记忆，确认上述每种形态下进程边界的真实划分，并据此校正架构图。
