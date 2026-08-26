# 锚点修复报告：add-a-kernel-backend.md

**SSOT**：/home/kimmo/develop/sglang（commit e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7）
**目标文件**：/home/kimmo/develop/sglangReading/docs/hacking/add-a-kernel-backend.md

## 修复明细

### 问题 1（anchor_drift, low）
- 旧锚点：`attention_backend_setup.py:L191-L222`（出现在文档第 149 行、第 199 行）
- 新锚点：`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:179-L222`
- 修正依据：在 SSOT 用 `grep -n "_build_resolved_backend" python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py` 实测，函数定义 `def _build_resolved_backend(` 位于 **第 179 行**，函数体延续至约 222 行。
- 修正类型：①补全缺前缀的裸文件名（`attention_backend_setup.py` → 补全为 `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py`）；②行号区间起点 191 改为真实定义行 179。

## 复核（grep 反复确认）

变更后文档中两处锚点均为：
`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:179-L222`（第 149 行、第 199 行）。

SSOT 复核：
```
$ grep -n "_build_resolved_backend" python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py
179:def _build_resolved_backend(
```
第 179 行确为文中引用的符号 `def _build_resolved_backend`，与修正后锚点一致。

## 未改动项
- 文档第 215 行引用 `attention_backend_setup.py:L191-L217`，属于**不同**锚点，不在本问题清单范围内，保持原样。
- 未新增/删除任何非锚点内容；mermaid 代码块完整；未引入新 TODO。
