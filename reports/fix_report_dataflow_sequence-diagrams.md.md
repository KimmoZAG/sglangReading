# 锚点修复报告：docs/dataflow/sequence-diagrams.md

- 目标文件：`/home/kimmo/develop/sglangReading/docs/dataflow/sequence-diagrams.md`
- SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）
- 评阅问题：问题 1（anchor_drift，涉及 `tp_worker.py:574` 与 `:609` 作为 `TpModelWorker.forward_batch_generation`）

## 结论：未做修改（SSOT 实测后锚点实为正确）

经 grep/Read 在 SSOT 实测复核，**文档现有锚点 `:574` / `:609` / `:651` 已正确指向 `TpModelWorker.forward_batch_generation` 及其 `ModelRunner.forward` / `ModelRunner.sample` 调用**，不存在评阅清单所描述的下沉（drift）。因此未对文档做任何改动。

### SSOT 实测证据（`python/sglang/srt/managers/tp_worker.py`）

```text
74:  class BaseTpWorker(ABC):
76:      def forward_batch_generation(self, forward_batch: ForwardBatch):   # 抽象声明（@abstractmethod, pass）
299: class TpModelWorker(BaseTpWorker):
574:     def forward_batch_generation(                                      # 具体实现 = TpModelWorker.forward_batch_generation
            ...
589:         forward_batch = ForwardBatch.init_new(...)                      # 文档 :589
609:         out = self.model_runner.forward(...)                            # 文档 :609（ModelRunner.forward）
641/651:   batch_result.next_token_ids = self.model_runner.sample(...)      # 文档 :651（ModelRunner.sample）
...
684:     def forward_batch_split_prefill(...)                               # 并非 forward_batch_generation
```

- `grep -n "^class " tp_worker.py` 仅返回 `BaseTpWorker`（74）与 `TpModelWorker`（299）两类。文件中**不存在**评阅 DETAIL 所声称的 "client/proxy 类"；`TpModelWorker` 类体从 299 延续到文件末尾，故 `:574` 确属 `TpModelWorker`。
- 评阅 DETAIL 称 "`TpModelWorker.forward_batch_generation` 实际定义于 `tp_worker.py:76`" 与 SSOT 不符：`:76` 是 `BaseTpWorker` 的**抽象**方法声明（`pass`），并非 GPU 前向入口，也不属于 `TpModelWorker`。
- 评阅提到 "该方法亦出现于 684" 亦与 SSOT 不符：`:684` 是 `forward_batch_split_prefill`，并非 `forward_batch_generation`。

### 文档中相关锚点（复核为正确，故保留）

| 文档位置 | 锚点 | SSOT 实测内容 | 判定 |
| --- | --- | --- | --- |
| 3.1 时序图 L75 | `[:3623/:574]` | `:574` = `TpModelWorker.forward_batch_generation` 定义 | 正确 |
| 3.1 时序图 L76 | `[:589/:609]` | `:609` = `self.model_runner.forward(...)` | 正确 |
| 3.2 时序图 L109 | `[:3623/:574]` | `:574` = `TpModelWorker.forward_batch_generation` 定义 | 正确 |
| 3.2 时序图 L110 | `[:609/:651]` | `:609` 前向 / `:651` 采样 | 正确 |
| 3.3 时序图 L155 | `[:3623/:574]` | `:574` = `TpModelWorker.forward_batch_generation` 定义 | 正确 |
| 证据速查表 L292 | `tp_worker.py:574` → `tp_worker.py:609` | `:574` 定义 / `:609` 前向调用 | 正确 |

### 为何不套用 SUGGESTED_FIX（574→76）

SUGGESTED_FIX 建议将主锚点改到 `:76`。但 SSOT 实测 `:76` 为 `BaseTpWorker.forward_batch_generation`（抽象 `pass`），既非 GPU 前向入口、也不属于 `TpModelWorker`。若改之，读者按 `:76` 将找不到 `TpModelWorker` 的该方法——即恰好制造了评阅所担心的"找不到"问题，且会违反验收要求"修正后的锚点行号处确为文中引用的符号"（`:76` 处符号为 `BaseTpWorker` 抽象方法，非 `TpModelWorker.forward_batch_generation`）。

依据任务规则"所有修正必须先用 grep/Read 在 SSOT 实测确认"与"不编造"，当 SSOT 实测证明现有锚点正确、而建议修复会把正确锚点改错时，正确做法是**不修改**。

## 验收复核（grep）

- `grep -n "def forward_batch_generation" tp_worker.py` → `76`（BaseTpWorker 抽象）、`574`（TpModelWorker 具体）。文档 `:574` 与 SSOT 一致。
- `grep -n "self.model_runner.forward" tp_worker.py` 在 `574` 方法体内命中 `:609` 与 `:673`；文档 `:609` 与 SSOT 一致。
- 文档未改动，其余所有非锚点内容保持原样，mermaid 代码块完整，无新增 TODO。
