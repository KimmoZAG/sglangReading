# 修复报告：docs/architecture/overview.md 源码锚点错误

- 目标文件：`/home/kimmo/develop/sglangReading/docs/architecture/overview.md`
- SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）
- 修复方式：仅改正锚点（路径/行号/符号名），未改动其它内容，未编造。

## 改动清单（旧 → 新）

### 问题 1（SEVERITY=high, anchor_fake）
- 位置：第 168 行（"仅在 disaggregation … Rust server 模式下"一句）
- 旧锚点：`python/sglang/srt/entrypoints/engine.py:2003` 的 `self.recv_from_tokenizer = rust_server`
- 新锚点：`python/sglang/srt/managers/scheduler.py:2003` 的 `self.recv_from_tokenizer = rust_server`
- 复核：`engine.py` 仅 1846 行（:2003 越界）且全文无 `rust_server`/`recv_from_tokenizer` 命中；`scheduler.py:2003` 实测为 `        self.recv_from_tokenizer = rust_server` ✅

### 问题 2（SEVERITY=medium, anchor_drift）
- 位置：第 178 行（"坑与边界"第 6 条"启动阻塞点"）
- 旧符号名：`wait_for_ready`（锚点 `engine.py:1762-1791` 区间本身正确）
- 新符号名：`_wait_for_scheduler_ready`（锚点区间 `engine.py:1762-1791` 保持不变）
- 复核：`engine.py:1762` 实测为 `def _wait_for_scheduler_ready(`，其内 `engine.py:1774` 为 `if scheduler_pipe_readers[i].poll(timeout=5.0):`；而 `engine.py:939` 的 `wait_for_ready` 仅是调用 `_wait_for_scheduler_ready` 的薄包装，不含 `poll(timeout=5.0)` 阻塞逻辑。文中描述的是 poll 阻塞等待行为，故符号名应更正为 `_wait_for_scheduler_ready`，行号区间保留。✅

### 问题 3（SEVERITY=low, anchor_drift）
- 位置：第 20 行（"GPU 计算并不由独立的'模型进程'承担"一句）
- 旧锚点：`python/sglang/srt/managers/scheduler.py:986-917`（区间起止颠倒，986 > 917，无效）
- 新锚点：`python/sglang/srt/managers/scheduler.py:917-986`
- 复核：`scheduler.py:917` 实测为 `            self.tp_worker = TpModelWorker(**worker_kwargs)`；`scheduler.py:986` 实测为 `    def init_model_worker(self):`，为升序有效区间。✅

## 验收（再次 grep 复核）
| 修正后锚点 | 行内容 | 结果 |
|------------|--------|------|
| `scheduler.py:2003` | `self.recv_from_tokenizer = rust_server` | ✅ 命中符号 |
| `engine.py:1762` | `def _wait_for_scheduler_ready(` | ✅ 命中符号 |
| `engine.py:1774` | `if scheduler_pipe_readers[i].poll(timeout=5.0):` | ✅ poll 阻塞逻辑所在 |
| `scheduler.py:917` | `self.tp_worker = TpModelWorker(**worker_kwargs)` | ✅ 命中符号 |
| `scheduler.py:986` | `def init_model_worker(self):` | ✅ 命中符号 |

未改动 mermaid 代码块，未引入新 TODO，未新增/删除非锚点内容。
