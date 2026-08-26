# 修复报告：observability.md 源码锚点

- 目标文件：`/home/kimmo/develop/sglangReading/docs/deep-dive/observability.md`
- SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）
- 修复范围：仅源码锚点错误（anchor_drift）。未改动其它非锚点内容，未引入新 TODO，mermaid 代码块保持完整。

## 已修正锚点（旧 → 新）

### 问题 1（SEVERITY=low TYPE=anchor_drift）— L112
- 旧：`profiler_manager.py:220-265`
- 新：`python/sglang/srt/managers/scheduler_components/profiler_manager.py:197` 与 `:338`
- 理由：`.trace.json.gz` 实际字符串出现在 SSOT 的 `profiler_manager.py:197`（RPD 分支文件名）与 `:338`（主输出文件名拼接），而非 220-265（该区间实为 `activities` 处理段）。

### 问题 1 相关 — L113（MEM/RPD/CUDA_PROFILER 扩展锚点）
- 旧：`profiler_manager.py:254-263`
- 新：`python/sglang/srt/managers/scheduler_components/profiler_manager.py:252-L264`
- 理由：原区间略偏；按 SUGGESTED_FIX 与 SSOT 实测，`MEM`（`_record_memory_history`，L254-257）与 `CUDA_PROFILER`（L260-263）处理块实际落在 L252-L264（含 `profile_in_progress` 置位与两块扩展收尾）。

## 复核（grep 验证行号处确为文中引用符号）

SSOT 文件：`/home/kimmo/develop/sglang/python/sglang/srt/managers/scheduler_components/profiler_manager.py`

- L197：`"rpd-" + str(time.time()) + f"-TP-{self.ps.tp_rank}" + ".trace.json.gz",` ✅ 即 `.trace.json.gz` Chrome trace 输出文件名
- L338：`+ ".trace.json.gz"` ✅ 即 `.trace.json.gz` Chrome trace 输出文件名
- L252-L264：
  - L254 `if "MEM" in activities:` → L255 `torch.cuda.memory._record_memory_history(...)`
  - L260 `if "CUDA_PROFILER" in activities:` → L262 `torch.cuda.cudart().cudaProfilerStart()`
  ✅ 覆盖 MEM / CUDA_PROFILER 扩展处理段

## 未处理项（超出本次锚点修复范围）

### 问题 2（SEVERITY=low TYPE=shallow）— L118-130（Benchmark 工具链）
- 性质：内容深度（shallow），**非源码锚点错误**。
- 建议修复需新增调用链内容或将该段落降级标注为“索引性附录”。
- 未处理原因：违反本次约束“严禁新增/删除非锚点内容”与总体范围“只改源码锚点错误”。如需处理，应单独发起内容修订任务，不在本次锚点修复内。

## 其它同类锚点说明

同段落 L114/L115/L144 的 `profiler_manager.py:*` 锚点（如 `135-153`、`392-418`、`58-62`、`99`、`114-118`、`311`）未被本问题清单标记为错误，且 SSOT 实测均为真实存在行号，故未改动，以保持最小修改范围。
