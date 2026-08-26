# 修复报告：key-data-structures.md

- 目标文件：`/home/kimmo/develop/sglangReading/docs/dataflow/key-data-structures.md`
- 事实来源 SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）
- 修复范围：仅修正源码锚点行号 ±1 微漂移；未改动任何非锚点内容、未引入新 TODO、mermaid 代码块保持完整。

## 修改的锚点（旧 → 新）

| 符号 | 旧锚点（文档中） | 新锚点（grep 复核后） | SSOT 核验 |
|---|---|---|---|
| `class ScheduleBatch` | `python/sglang/srt/managers/schedule_batch.py:L1995` | `python/sglang/srt/managers/schedule_batch.py:L1996` | `grep -n "class ScheduleBatch"` → `schedule_batch.py:1996` ✓ |
| `ScheduleBatch` 字段表区间（派生范围） | `...schedule_batch.py:L1995-L2181` | `...schedule_batch.py:L1996-L2181` | 同上，起点随定义行校正 ✓ |
| `ScheduleBatch.init_new` | `python/sglang/srt/managers/schedule_batch.py:L2183` | `python/sglang/srt/managers/schedule_batch.py:L2184` | `grep -n "def init_new"` → `schedule_batch.py:2184` ✓ |
| `class ForwardBatch` | `python/sglang/srt/model_executor/forward_batch_info.py:L411` | `python/sglang/srt/model_executor/forward_batch_info.py:L412` | `grep -n "class ForwardBatch"` → `forward_batch_info.py:412` ✓ |
| `ForwardBatch` 定义区间（派生范围） | `...forward_batch_info.py:L411-L638` | `...forward_batch_info.py:L412-L638` | 同上，起点随定义行校正 ✓ |

## 未处理项（说明）

- 问题清单 DETAIL 中提及的 `_get_decode_retraction_order` 标注 `schedule_batch.py:2856`（实际 2857）——**经 grep 核验，目标文档中根本不存在该锚点字符串**，故按「不编造」原则未做任何改动。

## 验收复核（修复后）

- `schedule_batch.py:1996` → `class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):` ✓
- `schedule_batch.py:2184` → `def init_new(` ✓
- `forward_batch_info.py:412` → `class ForwardBatch(ForwardBatchDeepSeekMHAMixin):` ✓
- 文档内不再残留 `L1995` / `L2183` / `L411` 旧锚点。
