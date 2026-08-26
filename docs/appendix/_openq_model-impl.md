# OPEN Questions — model-impl

> 本文档记录 `deep-dive/model-impl.md` 写作过程中遇到的不确定 / 多种可能之处，供后续补全。请勿直接修改 `open-questions.md`（避免并发冲突）。

### 问题 1：deepseek_v3.py 在本 commit 不存在
任务给定的必读清单包含 `python/sglang/srt/models/deepseek_v3.py`，但本 commit（`e1c4db9…`）下该文件不存在；实际文件为 `deepseek_v2.py`、`deepseek.py`、`deepseek_v4.py` 等。DeepSeek-V3 / V3.2 在源码中以 `DeepseekV3ForCausalLM(DeepseekV2ForCausalLM)` 子类实现于 `deepseek_v2.py`（见 `deepseek_v2.py:L3220-L3221`），MoE/MLA 逻辑全部复用于 `DeepseekV2ForCausalLM`。
- 描述：文档已按真实源码以 `deepseek_v2.py` 为准撰写。
- 可能方向：若后续 commit 拆分出独立 `deepseek_v3.py`，需回填对照表，确认类层次与 `EntryClass` 是否有变化。

### 问题 2：DeepseekV2AttentionMLA 的 forward 路径分派
`DeepseekV2AttentionMLA` 同时继承 `DeepseekMHAForwardMixin`、`DeepseekMHARocmForwardMixin`、`DeepseekMLAForwardMixin`、`DeepseekMLARocmForwardMixin`、`DeepseekMLAFusedRopeRocmForwardMixin`、`DeepseekMLACpuForwardMixin` 等多个 forward mixin（见 `deepseek_v2.py:L1711-L1718`），运行时由哪个 mixin 的 forward 接管依赖后端、`maybe_use_decode_attn_tp`、`attn_mqa`/`attn_mha` 选择等上下文。本文未逐 mixin 展开具体分派逻辑。
- 描述：文档仅说明它同时持有 `attn_mqa` 与 `attn_mha` 两套 `RadixAttention`，未描述完整分派图。
- 可能方向：后续补充 mla_forward 分派时序图，覆盖 CUDA / ROCm / CPU / NPU 各后端如何选路径，以及 `prepare_qkv_latent` 在 `LayerCommunicator` 中的角色。
