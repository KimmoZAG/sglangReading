# 锚点修复报告：docs/hacking/dev-setup.md

- **SSOT**：/home/kimmo/develop/sglang（commit e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7）
- **问题类型**：anchor_drift（裸文件名缺目录前缀）
- **修复文件**：/home/kimmo/develop/sglangReading/docs/hacking/dev-setup.md

## 修改的锚点（旧 → 新）

所有改动均为补全裸文件名 `contribution_guide.mdx:` 的目录前缀，行号经 grep 复核与 SSOT 实际内容一致，未改动其它任何内容。

| 行号 | 旧锚点 | 新锚点 | SSOT 复核 |
|---|---|---|---|
| 34 | `contribution_guide.mdx:108-113` | `docs/docs/developer_guide/contribution_guide.mdx:108-113` | L110 "only top and trusted contributors have permission to trigger CI tests" ✓ |
| 62 | `contribution_guide.mdx:23-31` | `docs/docs/developer_guide/contribution_guide.mdx:23-31` | L23 "## Format code with pre-commit" ✓ |
| 79 | `contribution_guide.mdx:33` | `docs/docs/developer_guide/contribution_guide.mdx:33` | L33 "pre-commit run --all-files manually runs all configured checks…re-run it" ✓ |
| 79 | `contribution_guide.mdx:35` | `docs/docs/developer_guide/contribution_guide.mdx:35` | L35 "Link checking with lychee is enforced in CI…" ✓ |
| 197 | `contribution_guide.mdx:34` | `docs/docs/developer_guide/contribution_guide.mdx:34` | L34 "Do not commit directly to the main branch…" ✓ |
| 199 | `contribution_guide.mdx:159` | `docs/docs/developer_guide/contribution_guide.mdx:159` | L159 "Never use pickle.loads()…" ✓ |
| 201 | `contribution_guide.mdx:156-157` | `docs/docs/developer_guide/contribution_guide.mdx:156-157` | L156 "If a single test file run longer than 500 seconds…" / L157 "If a single job…runs longer than 30 mins" ✓ |

## 复核方法

- 在 SSOT 用 `ls` 确认文件位于 `docs/docs/developer_guide/contribution_guide.mdx`。
- 用 `grep -n` / `sed -n` 逐条复核行号处确为文中引用的符号/内容。
- 修复后再次 `grep` 确认文档中已无裸文件名 `contribution_guide.mdx:`（仅剩带完整前缀的引用）。

## 其它说明

- 行号本身均准确，仅缺目录前缀，故只补前缀、不改行号、不增删任何非锚点内容。
- mermaid 代码块、表格、环境变量清单均未改动。
