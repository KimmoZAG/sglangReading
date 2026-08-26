# 修复报告：deep-dive/disaggregation.md

目标文件：`/home/kimmo/develop/sglangReading/docs/deep-dive/disaggregation.md`
SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）

## 修改的锚点（旧 → 新）

| 问题 | 位置 | 旧锚点 | 新锚点 | 复核 |
| --- | --- | --- | --- | --- |
| 问题 1（anchor_drift）§6.3 | 文件第 224 行 | `match_prefix_for_req`（`python/sglang/srt/disaggregation/decode.py:561`） | `_match_prefix_and_lock`（`python/sglang/srt/disaggregation/decode.py:561`） | 行号 561 保留；grep 复核 SSOT `decode.py:561` = `def _match_prefix_and_lock(self, req: Req) -> DecodePrefixMatch:` ✅ |

## 修改说明

- 仅改正符号名：`match_prefix_for_req` → `_match_prefix_and_lock`（原文误把内部方法名写成了其 import 的辅助函数名 `match_prefix_for_req`）。
- 行号 `561` 经 grep 复核确为 `_match_prefix_and_lock` 的定义行，予以保留。
- 未新增/删除任何非锚点内容，mermaid 代码块与 TODO 均保持不变。

## 二次复核（grep）

- SSOT：`grep -n "_match_prefix_and_lock" python/sglang/srt/disaggregation/decode.py` → `561:    def _match_prefix_and_lock(self, req: Req) -> DecodePrefixMatch:`
- 文档：`disaggregation.md` 第 224 行现为 `…用 `_match_prefix_and_lock` 匹配自身 radix 树（python/sglang/srt/disaggregation/decode.py:561）…`

修正后锚点行号处确为文中引用的符号 `_match_prefix_and_lock`。✅
