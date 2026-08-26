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
