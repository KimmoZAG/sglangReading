# Open Questions: glossary

> 本文件记录 glossary.md 写作时读不懂或存在多种可能的点，请勿直接修改 `open-questions.md`。

### 顶层 Data Parallel 与并行组内 DP 的关系
`python/sglang/srt/distributed/parallel_state.py` 中的 `initialize_model_parallel`
（约 2286 行）接收 `attention_data_parallel_size` 与 `moe_data_model_parallel_size` 参数，
并在函数内构造 attention DP 组（`attn_dp_size`，约 2451 行）与 MoE DP/EP 组（约 2528 行起）。
但 SGLang 在 server/引擎层还有"多副本数据并行"（用 `--data-parallel-size` 起多个模型副本）。

**问题**：顶层"引擎级 data parallel"的进程组是否也在此 `initialize_model_parallel` 内构造？
实读该函数未找到名为 `data_parallel` 的顶层组（只有 attn/moe 细分），怀疑顶层 DP 由更上层
的 server 启动逻辑（`python/sglang/srt/server.py` 或 entrypoints）负责，而非在此函数内。

**可能的方向**：
- 在 `python/sglang/srt/entrypoints/` 与 server 启动路径中搜索 `data_parallel_size` 的实际使用，
  确认顶层副本的 rank 划分位置。
- 确认 attn DP / moe DP 与顶层 DP 在 `world_size` 计算上的组合关系（是否乘积）。
- 注意 `get_world_group` 与 `get_tp_group`（`parallel_state.py:1861`、`:1944`）返回的是 global / TP 组，
  顶层 DP 若独立建组应有对应 `get_dp_group` 之类的访问器，可据此反查。
