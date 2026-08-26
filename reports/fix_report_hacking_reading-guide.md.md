# 锚点修复报告：reading-guide.md

- 目标文件：`/home/kimmo/develop/sglangReading/docs/hacking/reading-guide.md`
- SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）
- 修复日期：2026-08-26

## 修复清单（旧锚点 → 新锚点）

### 问题 1（anchor_drift, high）
- 符号：`launch_batch_sample_if_needed`
- 旧锚点（错误）：`python/sglang/srt/managers/scheduler.py:1814` 与 `:1814-1815`
  - 实测：:1814 = `if self.is_generation:`，:1815 = `self.launch_batch_sample_if_needed(batch_result, batch)` —— 仅为**调用点**，非函数定义。
- 新锚点（已 grep 复核）：`python/sglang/srt/managers/scheduler.py:3881`
  - 实测：:3881 = `def launch_batch_sample_if_needed(`（真实函数定义）
- 修改位置：
  - line 73：`（:1814-1815）` → `（:3881）`
  - line 133：`（:1814）` → `（:3881）`

### 问题 2（anchor_drift, medium）
- 符号：`init_model_worker`
- 旧锚点（错误）：`python/sglang/srt/managers/scheduler.py:901`
- 新锚点（已 grep 复核）：`python/sglang/srt/managers/scheduler.py:986`
  - 实测：:986 = `def init_model_worker(self):`（真实函数定义）
- 修改位置：
  - line 141：`（:901）` → `（:986）`

## 复核（grep 验证修正后锚点处确为引用符号）

```
$ grep -n "def launch_batch_sample_if_needed\|def init_model_worker" \
    /home/kimmo/develop/sglang/python/sglang/srt/managers/scheduler.py
986:    def init_model_worker(self):
3881:    def launch_batch_sample_if_needed(
```

- :3881 → `def launch_batch_sample_if_needed(` ✅ 与文档引用符号一致
- :986  → `def init_model_worker(self):` ✅ 与文档引用符号一致

## 说明
- 仅修改上述三处锚点行号（`:1814-1815`/` :1814` → `:3881`；`:901` → `:986`），未增删任何非锚点内容，mermaid 代码块与 `[OPEN]` 标记均保持原样。
- 未运行 mkdocs build（按沙箱约定）。
