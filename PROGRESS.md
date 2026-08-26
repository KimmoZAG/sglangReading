# PROGRESS.md —— SGLang 源码精读文档站

> 任务勾选表。每完成一个阶段/一篇文档，勾选并 git commit（信息用中文，形如 `docs: 完成调度器模块深潜`）。
> SSOT：`/home/kimmo/develop/sglang`（commit `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7`）。

## 阶段状态

- [x] 阶段 0：侦察（代码规模/语言分布/目录树/入口/构建/测试）
- [x] 阶段 1：骨架（mkdocs 工程 + GitHub Actions + 占位文件，`mkdocs build --strict` 通过）
- [ ] 阶段 2：主干打通（architecture/* 三篇 + dataflow/key-data-structures.md）
- [ ] 阶段 3：深潜（deep-dive/* 逐篇）
- [ ] 阶段 4：实操篇（quickstart / hacking）
- [ ] 阶段 5：收口（术语表/配置参考/交叉链接/断链检查/README）

## 全量任务勾选表

### 骨架与入口
- [x] `mkdocs.yml`（nav 显式列出、strict、mermaid、中文搜索分隔符）
- [x] `requirements-docs.txt`
- [x] `.github/workflows/docs.yml`（Pages 部署）
- [x] `docs/index.md`（版本快照、能力矩阵、阅读路线）
- [x] `docs/architecture/directory-map.md`（初版）

### 阶段 2：主干
- [ ] `docs/architecture/overview.md`
- [ ] `docs/architecture/request-lifecycle.md`
- [ ] `docs/dataflow/key-data-structures.md`

### 阶段 3：深潜（顺序 scheduler→memory-pool→radix-cache→model-runner→attention-backends→其余）
- [ ] `docs/deep-dive/scheduler.md`
- [ ] `docs/deep-dive/memory-pool.md`
- [ ] `docs/deep-dive/radix-cache.md`
- [ ] `docs/deep-dive/model-runner.md`
- [ ] `docs/deep-dive/attention-backends.md`
- [ ] `docs/deep-dive/frontend-language.md`
- [ ] `docs/deep-dive/server-entrypoint.md`
- [ ] `docs/deep-dive/tokenizer-detokenizer.md`
- [ ] `docs/deep-dive/model-impl.md`
- [ ] `docs/deep-dive/parallelism.md`
- [ ] `docs/deep-dive/quantization.md`
- [ ] `docs/deep-dive/constrained-decoding.md`
- [ ] `docs/deep-dive/speculative-decoding.md`
- [ ] `docs/deep-dive/sampling.md`
- [ ] `docs/deep-dive/lora-multimodal.md`
- [ ] `docs/deep-dive/disaggregation.md`
- [ ] `docs/deep-dive/observability.md`

### 阶段 4：实操
- [ ] `docs/quickstart/install.md`
- [ ] `docs/quickstart/minimal-example.md`
- [ ] `docs/quickstart/e2e-observation.md`
- [ ] `docs/hacking/dev-setup.md`
- [ ] `docs/hacking/add-a-model.md`
- [ ] `docs/hacking/add-a-kernel-backend.md`
- [ ] `docs/hacking/reading-guide.md`

### 阶段 5：收口
- [ ] `docs/appendix/glossary.md`
- [ ] `docs/appendix/config-reference.md`
- [ ] `docs/appendix/open-questions.md`
- [ ] `docs/appendix/changelog-of-docs.md`
- [ ] `README.md`（Pages 链接、徽章、Settings 说明）
- [ ] 交叉链接补全 + `mkdocs build --strict` 最终通过
- [ ] 断链检查脚本通过

## 执行说明
- 子任务经 `codebuddy` CLI 并行拉起（无会话持久化、--max-turns 限流），每个子任务独立改不同文件避免冲突。
- 子任务不负责 git commit（避免并发提交损坏仓库）；每波完成后由主会话统一 commit 并更新本表。
- 所有论断须带证据锚点；存疑写入 `docs/appendix/open-questions.md`。

## 迭代记录（self-iteration，CLI 驱动 + 机械校验兜底）

- R1 CLI 评阅（4 并行 reviewer）：整体 8–9.5/10；发现锚点类缺陷（路径错/行号漂移/区间颠倒/缺前缀/占位）。
- F1/F2 CLI 修复（14 文件）：子会话**过度声称成功**——部分 Edit 未真正落盘（如 overview 高优 `engine.py:2003` 未改）。
- R3 CLI 复审失败：codebuddy CLI 触发**频率限制**（2026-08-27 10:48 UTC+8 重置），CLI 自迭代循环暂停。
- 改为机械校验兜底：自研 `prompts/verify_anchors.py` 扫描全部 34 文档锚点（文件存在性 / 行号越界 / 区间颠倒 / 占位），
  修复其报出的 5 处真实缺陷：overview `engine.py:2003`→`scheduler.py:2003`、overview `986-917`→`917-986`、
  server-entrypoint/config-reference `arg_utils.py:L338`→`L337`、quantization `kv_cache.py:L86`→`L85`。
  复跑结果：**显式路径锚点 0 处越界/颠倒/占位**（34 文档）。
- 待 CLI 限额恢复后，重启「评阅→决定→修复→验收」循环，重点攻：裸文件名锚点的符号级准确性、深度(What/Why/How/坑)、mermaid 类名真实性。
