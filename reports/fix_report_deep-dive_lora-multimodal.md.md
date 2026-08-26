# 锚点修复报告：docs/deep-dive/lora-multimodal.md

- SSOT：`/home/kimmo/develop/sglang`，commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`（已用 `git rev-parse HEAD` 复核）
- 目标文件：`/home/kimmo/develop/sglangReading/docs/deep-dive/lora-multimodal.md`
- 修改数量：2 处锚点（仅锚点字符串，未增删其它内容）

## 修改明细

### 1. `get_new_expanded_mm_items`（问题 1，anchor_fake）

- 位置：文档 L126「按图拆分提升缓存粒度」条目
- 旧锚点：`python/sglang/srt/multimodal/processors/base_processor.py:1693-1695`
- 新锚点：定义见 `python/sglang/srt/managers/mm_utils.py:1090`；调用处 `python/sglang/srt/multimodal/processors/base_processor.py:1693-1695`
- grep 复核：
  - `python/sglang/srt/managers/mm_utils.py:1090:def get_new_expanded_mm_items(original_mm_items):`
  - `base_processor.py:1693:        from sglang.srt.managers.mm_utils import get_new_expanded_mm_items`
  - `base_processor.py:1695:        all_collected_items = get_new_expanded_mm_items(all_collected_items)`
- 说明：原锚点把「import + 调用」误标为定义；保留调用处并补上真实定义位置。

### 2. `get_mm_items_offset`（问题 2，anchor_drift）

- 位置：文档 L137「为每个 item 计算 `offsets`」条目
- 旧锚点：`python/sglang/srt/multimodal/processors/base_processor.py:1681-1690`
- 新锚点：定义于 `python/sglang/srt/multimodal/processors/base_processor.py:1297-1310`，调用处 `base_processor.py:1687`
- grep 复核：
  - `base_processor.py:1297:    def get_mm_items_offset(`
  - `base_processor.py:1310:        return list(zip(start_positions.tolist(), end_positions.tolist()))`（函数体最后一行；L1312 起为 `@staticmethod` + `get_mm_items_offset_by_pair`）
  - `base_processor.py:1687:            mm_item.offsets = self.get_mm_items_offset(`
- 说明：SUGGESTED_FIX 给的 `1297-L1312` 中 1311-1312 已属于下一个方法的空行与装饰器，故取实测定义区间 `1297-1310`；调用行实测为 1687（原标注 1681-1690 为粗略区间）。

## 未改动项

- 未触碰 mermaid 代码块、Python 代码块、`[OPEN]` 段落及其它任何锚点。
- 未运行 `mkdocs build`（按要求跳过）。
