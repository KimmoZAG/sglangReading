# Open Questions: quickstart/minimal-example

对齐 commit：`e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`

### `sglang serve --config x.yaml`（model_path 仅写在 YAML 中）是否会在读取配置前就失败？

`python/sglang/cli/serve.py:L101-L105` 的执行顺序是：先 `_extract_model_type_override`，再 `_normalize_positional_model_path`，然后立刻 `model_path = get_model_path(dispatch_argv)`。而 `get_model_path` 只扫描 argv 里的 `--model-path` / `--model`（含 `=` 写法），找不到就 `raise Exception("Error: --model-path is required. ...")`（`python/sglang/cli/utils.py:L99-L124`）。

但 YAML 配置的合并发生在更下游：`prepare_server_args` 在 `"--config" in argv` 时才构造 `ConfigArgumentMerger` 并把配置项插入 argv（`python/sglang/srt/server_args.py:L9669-L9681`，合并逻辑见 `python/sglang/srt/server_args_config_parser.py:L51-L80`）。也就是说 `serve()` 判断 model_path 的时刻，YAML 里的 `model_path` 还没有进入 argv。

推断：`sglang serve --config x.yaml`（且 `model_path` 只存在于 YAML）会在 `get_model_path` 处抛异常，而 `python -m sglang.launch_server --config x.yaml` 走的是 `prepare_server_args(sys.argv[1:])`（`python/sglang/launch_server.py:L68`），不经过 `get_model_path`，因此能正常工作。两个入口在这一点上不等价。

未能确认的原因与可能方向：
1. 本次仅做静态阅读，没有执行 `sglang serve --config`，无法排除某处对 `--config` 的前置展开（例如 plugin 在 `load_plugins()` 中改写 argv，`python/sglang/cli/serve.py:L97-L99`）。
2. 也可能实际用法约定为"`--model-path` 仍写在命令行，`--config` 只承载其余参数"，此时不构成问题。
3. 验证方式：构造一个只含 `model_path` 的 YAML，分别用 `sglang serve --config` 和 `python -m sglang.launch_server --config` 启动，对比是否抛出 `--model-path is required`。
