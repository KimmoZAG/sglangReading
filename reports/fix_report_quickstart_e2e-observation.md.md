# 修复报告：quickstart/e2e-observation.md

## 目标文件
`/home/kimmo/develop/sglangReading/docs/quickstart/e2e-observation.md`

## SSOT 基准
`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）

## 修正的锚点（旧 → 新）

### 问题 1（anchor_drift）— `request_logger.py` 的 `_compute_metadata` 范围起点偏移
- 旧：`python/sglang/srt/utils/request_logger.py:200-L234`（文中第 170 行，`--log-requests-level=3` 规则处）
- 新：`python/sglang/srt/utils/request_logger.py:193-L234`
- 复核：`grep -n "def _compute_metadata" request_logger.py` → `193:    def _compute_metadata(`（函数体 193-235，结束行 234 落在该函数内）

### 问题 2（anchor_drift）— `metrics_collector.py` 三处延迟直方图 off-by-1
- 旧：`metrics_collector.py:1698`（time_to_first_token_seconds，文中第 126、187 行）
- 新：`metrics_collector.py:1699`
- 旧：`metrics_collector.py:1708`（inter_token_latency_seconds，文中第 127、187 行）
- 新：`metrics_collector.py:1709`
- 旧：`metrics_collector.py:1715`（e2e_request_latency_seconds，文中第 128、187 行）
- 新：`metrics_collector.py:1716`
- 复核：
  - `grep -n "time_to_first_token_seconds" metrics_collector.py` → `1699: name="sglang:time_to_first_token_seconds"`
  - `grep -n "inter_token_latency_seconds" metrics_collector.py` → `1709: name="sglang:inter_token_latency_seconds"`
  - `grep -n "e2e_request_latency_seconds" metrics_collector.py` → `1716: name="sglang:e2e_request_latency_seconds"`

## 末次 grep 复核（修正后）
- `request_logger.py:193` → `def _compute_metadata` ✔
- `metrics_collector.py:1699` → `name="sglang:time_to_first_token_seconds"` ✔
- `metrics_collector.py:1709` → `name="sglang:inter_token_latency_seconds"` ✔
- `metrics_collector.py:1716` → `name="sglang:e2e_request_latency_seconds"` ✔

## 范围说明
- 仅修改源码锚点字符串，未增删任何非锚点正文、未改动 mermaid 代码块、未引入新 TODO。
- 未运行 mkdocs build（遵循沙箱规避要求）。
