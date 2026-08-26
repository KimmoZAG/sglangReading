# SGLang 源码精读（文档站）

一套「能让人从零到深入掌握 SGLang」的中文文档站，基于本地 sglang 源码（SSOT）逐行阅读整理，通过 GitHub Pages 发布。

> 所有结论均来自源码阅读，附证据锚点（如 `python/sglang/srt/managers/scheduler.py:L120-L180`）。
> 文档站本身由 [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) 构建。

## 文档对齐版本

| 项 | 值 |
| --- | --- |
| 源码路径 | `/home/kimmo/develop/sglang` |
| Git Commit | `e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7` |
| Commit 时间 | `2026-08-14 11:11:02 +0800` |
| `git describe` | `gateway-v0.3.1-7844-ge1c4db9621` |

## 本地预览

```bash
pip install -r requirements-docs.txt
mkdocs serve          # 本地 http://127.0.0.1:8000
# 或仅构建静态站：
mkdocs build --strict
```

> 注：在本机某些沙箱环境下 `mkdocs build` 的 site 目录清理会被拦截（与文档内容无关）。
> 完整的 `--strict` 构建在 CI（GitHub Actions，`ubuntu-latest`）中执行，请推送后查看 Actions 结果。

## 发布到 GitHub Pages

本仓库已包含 `.github/workflows/docs.yml`：push 到 `main` 时自动
`setup-python → pip install → mkdocs build --strict → upload-pages-artifact → deploy-pages`。

开启 Pages：

1. 仓库 **Settings → Pages → Source** 选择 **GitHub Actions**。
2. 推送 `main` 分支，等待 Actions 跑完。
3. 访问站点：`https://<你的用户名>.github.io/sglangReading/`。

### 项目页（project page）路径注意

本仓库以项目页形式托管（URL 含 `/sglangReading/` 子路径）。请确保 `mkdocs.yml` 中：

- `site_url: https://<用户名>.github.io/sglangReading/`（结尾带 `/` 与仓库名一致）
- 默认 `use_directory_urls: true`（mkdocs-material 会据此生成相对资源路径）

若部署后样式/资源 404，通常是 `site_url` 未带仓库名子路径，按上条修正即可。

## 目录结构

```
mkdocs.yml                 # 站点配置（nav / 主题 / 插件 / 中文搜索分隔符）
requirements-docs.txt      # 文档构建依赖
.github/workflows/docs.yml # Pages 自动发布流水线
docs/
  index.md                 # 首页：项目定位、能力矩阵、版本快照、阅读路线
  architecture/            # 全局架构、请求生命周期、目录地图
  deep-dive/               # 逐模块深潜（调度器/KV缓存/Radix/ModelRunner/…）
  dataflow/                # 核心数据结构、时序图集
  quickstart/              # 安装、最小示例、端到端观测
  hacking/                 # 开发环境、新增模型/后端、阅读路线
  appendix/                # 术语表、配置参考、未解问题、变更日志
prompts/                   # 生成各文档所用的自包含子任务 prompt（溯源）
```

## 构建状态

![pages-build](https://github.com/kimmo/sglangReading/actions/workflows/docs.yml/badge.svg)

## 许可

文档内容基于 sglang 源码阅读整理，遵循 sglang 上游许可；本仓库文档以 CC-BY-4.0 组织。
