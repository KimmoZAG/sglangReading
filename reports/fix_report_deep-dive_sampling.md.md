# 修复报告：docs/deep-dive/sampling.md

SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）
目标文件：`/home/kimmo/develop/sglangReading/docs/deep-dive/sampling.md`

## 修正的锚点（旧 → 新）

### 问题 1（anchor_drift，§5.6）
- `sampling_batch_info.py:414-L443` → `sampling_batch_info.py:388-L443`
  - 复核：`def merge_batch(self, other: SamplingBatchInfo)` 真实定义在 `python/sglang/srt/sampling/sampling_batch_info.py:388`；原 L414 实际是 merge_batch 函数体内的注释，区间起点漂移约 26 行。
- `sampling_batch_info.py:430-L432` → 拆为两处：
  - `__len__` 定义：`sampling_batch_info.py:236`（`def __len__(self):` → `return len(self.temperatures)`）
  - merge_batch 内关于 len 的注释：`sampling_batch_info.py:428-L445`（L430-L432 为该注释，L433-L443 为 temperatures 等张量的合并循环）

## grep 复核（修正后行号确为所引符号）

```
python/sglang/srt/sampling/sampling_batch_info.py:236:    def __len__(self):
python/sglang/srt/sampling/sampling_batch_info.py:388:    def merge_batch(self, other: SamplingBatchInfo):
python/sglang/srt/sampling/sampling_batch_info.py:430:        # Note: because the __len()__ operator is defined on the temperatures tensor,
```

- L236 确为 `def __len__`（返回 `len(self.temperatures)`）。
- L388 确为 `def merge_batch`，其函数体覆盖至 L443+（414 起为 "Merge logit bias" 注释，430 起为 len 注释，433-443 为 temperatures/top_ps/... 合并循环）。
- L428-L445 确为 merge_batch 内关于 `__len__`/temperatures 的注释与合并逻辑。

## 范围
仅修改 §5.6 第 119 行中的两处源码锚点字符串，未增删任何非锚点内容，未改动 mermaid 代码块，未引入新 TODO。
