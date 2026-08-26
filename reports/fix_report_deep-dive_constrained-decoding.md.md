# 锚点修复报告：constrained-decoding.md

- 目标文件：`/home/kimmo/develop/sglangReading/docs/deep-dive/constrained-decoding.md`
- SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）

## 修复的锚点

| 条目 | 旧锚点 | 新锚点 | 说明 |
|------|--------|--------|------|
| 问题 1（§3.2） | `_pp_sync_ready_failed` L72-L107（缺路径、def 行号错误） | `python/sglang/srt/constrained/grammar_manager.py:77-L107` | 补完整 SSOT 相对路径；`def _pp_sync_ready_failed` 真实定义行为 L77（grep 复核）；区间 77-107 覆盖函数体（L107 为 `return data`），结论不受影响。 |

## 复核（grep/Read）

- `grep -n "def _pp_sync_ready_failed" python/sglang/srt/constrained/grammar_manager.py` → `77:    def _pp_sync_ready_failed(`
- `sed -n '77p;107p' grammar_manager.py` → L77 = `def _pp_sync_ready_failed(`，L107 = `return data`（函数体内）。
- 文档中修正后锚点字符串：`python/sglang/srt/constrained/grammar_manager.py:77-L107`，行 75。

## 未改动内容

- 仅修改 §3.2 中一条锚点字符串，未新增/删除任何非锚点内容，mermaid 代码块保持完整，无新 TODO。
