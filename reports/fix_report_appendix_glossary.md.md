# 锚点修复报告：docs/appendix/glossary.md

SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）

## 修复的锚点（旧 → 新）

| 术语/符号 | 旧锚点（错误） | 新锚点（grep 复核） | 说明 |
|-----------|---------------|---------------------|------|
| `mix_with_running` / `merge_batch` | `scheduler.py:3085-L3099`、`:3440`（glossary.md:103-104） | `python/sglang/srt/managers/schedule_batch.py:3194`（`merge_batch`）、`:2739`（`mix_with_running`） | 二者均为 `ScheduleBatch` 的方法，定义于 `schedule_batch.py`；原 `scheduler.py` 行号只是调用点（`running_batch.merge_batch(last_batch)` @ 3089、`new_batch.mix_with_running(running_batch)` @ 3440），并非定义。 |

## 复核（grep 验证）

- `python/sglang/srt/managers/schedule_batch.py:3194` → `def merge_batch(self, other: ScheduleBatch):` ✅
- `python/sglang/srt/managers/schedule_batch.py:2739` → `def mix_with_running(self, running_batch: ScheduleBatch):` ✅
- `python/sglang/srt/managers/scheduler.py` 内 `grep -n "def mix_with_running\|def merge_batch"` → 无匹配（确认非定义所在）✅
- `scheduler.py:3089` / `:3440` 处确为对上述方法的调用，符合"调用点而非定义"。

## 范围声明

- 仅修改 `glossary.md` 一处锚点（第 103-104 行），其它内容、mermaid 代码块、OPEN 注释均未改动。
- 本次只处理本文件问题清单（问题 1，anchor_drift，low）。TOP5 中其余条目属于其它文档（reading-guide.md / e2e-observation.md / add-a-kernel-backend.md），不在本文件范围内，未触碰。
