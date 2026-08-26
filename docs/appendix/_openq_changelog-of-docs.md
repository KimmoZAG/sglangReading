# Open Questions — changelog-of-docs

### PROGRESS.md 在本快照中不存在，状态来源应以谁为准？
任务说明要求「列出已规划的文档清单与状态（对照 PROGRESS.md）」，但本快照（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）的 SSOT 根目录与文档站根目录均不存在 `PROGRESS.md`（已用 `ls` 验证：`/home/kimmo/develop/sglang/PROGRESS.md` 不存在；文档站 `sglangReading/docs/PROGRESS.md` 也不存在）。

**当前处理**：本文档 `changelog-of-docs.md` 以**文件系统实际清单**作为状态来源——依据各文档 `.md` 是否存在、以及是否存在对应 `docs/appendix/_openq_<文档名>.md` 来推断「草稿（含未解问题）/ 规划中」。

**可能的方向**：
1. 主会话是否在后续统一创建 `PROGRESS.md` 并作为权威状态源？若是，本文第 4 节应改为引用它并同步。
2. 是否应把「文档清单+状态」直接以本文第 4 节为权威（不再依赖 PROGRESS.md）？若如此，`PROGRESS.md` 仅是任务描述的误导项，可忽略。
3. 状态推断仅基于 `_openq_*.md` 有无，是否足够？是否需要额外读取各文档正文以区分「草稿 / 评审通过 / 完成」？当前为避免大范围读取，仅做二分推断，存在粒度不足的风险。
