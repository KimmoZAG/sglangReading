# Open Questions: model-runner

### DecodeCudaGraphRunner.enable_torch_compile 在 CUDA decode 图路径下的实际作用范围

`DecodeCudaGraphRunner.enable_torch_compile` 由 `get_flags().capture.enable_torch_compile` 赋值（`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:213`），若为真则调用 `set_torch_compile_config()`（`decode_cuda_graph_runner.py:341-L342`）。但同时 `base_runner.py:448-L459` 在 warmup 阶段对不兼容模型（如动态 rope scaling）会把它关闭。

疑问：在 CUDA 上的 decode CUDA Graph 路径里，torch.compile 究竟是
(a) 只用于 MoE / attention kernel 的融合编译（graph 本身仍由 `torch.cuda.CUDAGraph` 录制与 `backend.replay` 重放），还是
(b) 会在 `model.forward` 外层再包一层 `torch.compile` 并进入录制图？

源码中 decode 主路径使用 `self.backend.replay(self._replay_graph_key, forward_batch)`（`decode_cuda_graph_runner.py:1328`），未见对 `model.forward` 显式 `torch.compile`；而 `enable_torch_compile` 与 `set_torch_compile_config` 的来源（`srt.compilation.torch_compile_decoration`）与 warmup 的交互未在一处清晰串联。

可能的方向：追踪 `BaseRunner.warmup()` 与 `_run_compile_pass` 实际调用链，确认 `enable_torch_compile` 是否只设置全局编译配置、影响具体哪些子模块（MoE runner / attention backend）的编译，而不直接包裹 decode 录制图。可对照 `cpu_graph_runner.py` 与 `tc_piecewise_cuda_graph_backend.py` 中 torch.compile 的明确用法来反推 decode 路径的差异。
