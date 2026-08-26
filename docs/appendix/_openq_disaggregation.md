### 外部 Router 的扇出与 bootstrap_room 分配策略在引擎侧不可见

本文档 §4 描述的路由只覆盖引擎侧可见逻辑：请求携带 `bootstrap_host`/`bootstrap_room`（`python/sglang/srt/entrypoints/openai/protocol.py:376,378`），两端用相同 `bootstrap_room` 在 Bootstrap Server 上 rendezvous。但真正“把请求扇出到哪个 prefill / decode 实例、如何分配 `bootstrap_room` 的取值”的外部 Router（如 sglang-router 项目）不在本 commit 的 `python/sglang/srt/` 源码树内。

- 可能方向 1：Router 把同一请求同时投递到（选定的 prefill 实例 + 选定的 decode 实例），并生成唯一 `bootstrap_room`；两端各自 `recv_requests` 后通过 room 配对。依据是 `event_loop_normal_disagg_prefill`（`python/sglang/srt/disaggregation/prefill.py:568`）与 `event_loop_normal_disagg_decode`（`python/sglang/srt/disaggregation/decode.py:2147`）都从 `request_receiver.recv_requests()` 收请求。
- 需要到 Router 侧源码确认：a) 是否真的双投；b) `bootstrap_room` 是否保证跨 prefill DP rank 均匀分散（影响 `bootstrap_room % dp_size` 这层内部负载均衡是否生效，见 `python/sglang/srt/disaggregation/decode.py:593`）。

### bootstrap_room 的离散性对 prefill DP rank 倾斜的影响

`DecodePreallocQueue._resolve_prefill_dp_rank`（`python/sglang/srt/disaggregation/decode.py:577`）在 `follow_bootstrap_room=True` 时用 `bootstrap_room % dp_size` 选 prefill DP rank。该取模是否成立，取决于 Router 端 `bootstrap_room` 的分配是否均匀覆盖 `[0, dp_size)`。若 Router 顺序分配相邻 room 且 prefill `dp_size` 与取模基数不匹配，会造成 DP rank 热点。该问题只能在 Router 侧源码/部署配置中确认，引擎侧无证据。
