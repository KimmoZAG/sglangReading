# Open Questions — quantization

> 本文件由 quantization.md 文档化过程中遗留的未决问题汇总而成。请勿直接修改 `open-questions.md`（避免并发冲突），如需补充请追加到此文件。

### `original_isinstance` 悬空赋值

`python/sglang/srt/layers/quantization/__init__.py:L173` 有一行 `original_isinstance = builtins.isinstance`。全仓库 grep 只有这一处出现，无任何消费者。疑似从 vLLM 的 `isinstance` 全局 patch（用于让量化层在 `builtins.isinstance` 下被识别为原始类型）移除后的遗留物。未能确认是否存在动态引用（例如通过 `exec`/`setattr` 在别处被回读），因此未在主文档中下"已无用"的结论，仅标注为 OPEN。建议方向：在 `builtins.isinstance` 上挂 hook 或启动期扫描 `gc.get_referrers`，确认该模块级绑定确无读取方后再清理。

### `UnquantizedKVCacheMethod.create_buffers` 返回 `None`

`python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py:L299-L300` 中 `UnquantizedKVCacheMethod.create_buffers` 的函数体只有 `pass`，即返回 `None`。在 FP4 KV 三方分工（quant_method ► Pool ► Backend）中，Pool 侧 `create_buffers` 的调用方（`mem_cache/kv_cache_configurator.py:L249-L261` 经 `get_kv_cache_quant_method`）通常只对真正的量化方法调用它，而未量化路径直接用标准 BF16 池子、`create_buffers` 不被调用。但文档未能**完全确认**所有 `MHATokenToKVPool` / `MHAChunkedTokenToKVPool` 的初始化分支都不会走到 `UnquantizedKVCacheMethod.create_buffers`。若未来某条路径误调，会拿到 `None` 而非 `dict`，在 `memory_pool.py` 的 buffer 布局解包处静默崩溃。建议方向：逐条核对 `kv_cache_configurator.py` 与 `memory_pool.py` 中所有 `get_kv_cache_quant_method` 的返回值使用点，确认 `UnquantizedKVCacheMethod` 实例在构造期就被短路、永不进入需要 buffer 的 Pool 分支。

### `get_quantization_config` 错误消息与实际可用集合不一致

`python/sglang/srt/layers/quantization/__init__.py:L146-L170` 中，CPU+AMX 平台下 `get_quantization_config` 会把可用集合二次收窄到 `CPU_QUANTIZATION_METHODS`（`__init__.py:L133-L141`，仅 7 种方案）。但其抛出的 `ValueError` 消息打印的却是 `QUANTIZATION_METHODS.keys()`（即全量注册表），与实际可加载集合不符。用户在 CPU+AMX 上看到错误提示里列了某方案、却仍加载失败时会困惑。建议方向：将错误消息改为打印实际参与查找的那个字典（`cpu_quant_methods` 或 `QUANTIZATION_METHODS`），或显式说明"当前平台仅支持 CPU_QUANTIZATION_METHODS 子集"。（该条因在正文 3.1 节已点出，此处仅汇总留痕。）
