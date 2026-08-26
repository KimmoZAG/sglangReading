# Open Questions: add-a-kernel-backend

### 注册表与 ATTENTION_BACKEND_CHOICES 的一致性缺少启动期校验
问题描述：新增后端时，名字需要同时出现在两处——注册表 `ATTENTION_BACKENDS`（`python/sglang/srt/layers/attention/attention_registry.py:L31`）与 CLI choices 列表 `ATTENTION_BACKEND_CHOICES`（`python/sglang/srt/server_args.py:L179-L207`）。目前若只在注册表登记而忘记把名字加进 choices，错误不会在 argparse 阶段暴露，而是延迟到运行时 `_build_full_attention_backend_from_str` 的 `if backend_str not in ATTENTION_BACKENDS: raise ValueError(f"Invalid attention backend: {backend_str}")`（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:L252-L253`），且错误信息不含「choices 缺项」提示。

可能的方向：
- 在 ServerArgs 构造后或 `init_attention_backends` 早期加一个一致性自检：遍历 `ATTENTION_BACKENDS` 的 key，校验其为 `ATTENTION_BACKEND_CHOICES` 子集，反之也校验 choices 中每一项确实可解析（避免拼写错误导致选型阶段才报错）。
- 也可考虑用一个共享的 `register_attention_backend` 副作用，在注册时自动把名字 `add_attention_backend_choices([name])`，从根上保证两处同步。但需评估是否会影响插件式按需加载（惰性 import）的语义。
- 需要确认是否存在合法的「注册但不在 choices 中」的使用场景（例如仅供内部测试、或仅由 `attn_backend_wrapper` 间接构造而非 CLI 选择），以免自检误报。
