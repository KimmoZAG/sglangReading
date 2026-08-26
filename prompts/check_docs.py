#!/usr/bin/env python3
import os, glob

# 1) 代码围栏（```）配对检查
bad = []
for f in glob.glob("docs/**/*.md", recursive=True):
    s = open(f, encoding="utf-8").read()
    if s.count("```") % 2 != 0:
        bad.append(f)
print("UNBALANCED FENCES:", bad if bad else "none")

# 2) nav 中列出的页面是否都存在（解析 mkdocs.yml 的 nav 文件名）
import re
yml = open("mkdocs.yml", encoding="utf-8").read()
# 匹配 'xxx.md' 或 'xxx/yyy.md'
refs = re.findall(r"[\w./-]+\.md", yml)
missing = []
for r in sorted(set(refs)):
    if not os.path.exists(os.path.join("docs", r)):
        missing.append(r)
print("MISSING NAV FILES:", missing if missing else "none")

# 3) 是否还有 TODO 占位
todo = []
for f in glob.glob("docs/**/*.md", recursive=True):
    if "TODO: 待子任务填充" in open(f, encoding="utf-8").read():
        todo.append(f)
print("TODO REMAINS:", todo if todo else "none")

# 4) 每篇文档的字数(m)/锚点数/mermaid数 概览
print("\n=== per-doc stats ===")
for f in sorted(glob.glob("docs/**/*.md", recursive=True)):
    s = open(f, encoding="utf-8").read()
    anchors = len(re.findall(r"python/sglang/[A-Za-z0-9_./-]+\.py[:#]L?[0-9-]+", s))
    mer = s.count("```mermaid")
    print(f"{f:48s} chars={len(s):6d} anchors={anchors:4d} mermaid={mer}")
