# 锚点修复报告：scheduler.md

- 目标文件：`/home/kimmo/develop/sglangReading/docs/deep-dive/scheduler.md`
- SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）

## 修复的锚点

| 条目 | 旧锚点 | 新锚点 | 说明 |
|------|--------|--------|------|
| 问题 1（§3.5，行 142） | `python/sglang/srt/managers/scheduler.py:1173`（标注 `init_chunked_prefill`） | `python/sglang/srt/managers/scheduler.py:1153` | `grep -n "def init_chunked_prefill" scheduler.py` 实测定义为 L1153（相差 20 行），按 SUGGESTED_FIX 复核更正。 |
| 全文锚点汇总（行 228） | `:1153`、`:1173` | `:1153` | 汇总行中的 `:1173` 同指 `init_chunked_prefill`，随正文一并更正，并消除重复锚点（合并为单一 `:1153`）。 |

## 复核（grep/Read）

- `grep -n "def init_chunked_prefill" python/sglang/srt/managers/scheduler.py` → `1153:    def init_chunked_prefill(self):`
- `Read scheduler.py:1153` → `    def init_chunked_prefill(self):`（确为 `init_chunked_prefill` 定义行）。
- 文档修正后锚点字符串：
  - 行 142：`python/sglang/srt/managers/scheduler.py:1153`，上下文引用 `init_chunked_prefill`。
  - 行 228 汇总：`:1153` 保留单一，无残留 `:1173`。
- 文档内已无 `1173` 字样（`grep -n "1173"` → No matches）。

## 未改动内容

- 仅修改 §3.5 正文一条锚点及文末汇总锚点行，未新增/删除任何非锚点内容，mermaid 代码块保持完整，无新 TODO。
