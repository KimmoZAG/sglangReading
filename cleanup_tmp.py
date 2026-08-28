#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理 SGLang 文档站仓库的中间产物（仅删除生成/运行时产物，保留产品与质量脚本）。"""
import os, shutil, glob

ROOT = "/home/kimmo/develop/sglangReading"

KEEP_PROMPTS = {"check_docs.py", "verify_anchors.py"}

def rm(path):
    if not os.path.exists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
    print("REMOVED:", os.path.relpath(path, ROOT))

# 1) gitignored runtime/build dirs
for d in ["site", "logs", "pids"]:
    rm(os.path.join(ROOT, d))

# 2) 未跟踪的 prompts/ 生成物（已被 git rm 的是 tracked；这里清 disk 残留）
for f in ["launch_reviews.sh", "parse_r4.py", "verify_facts.py",
          "reviewR4_1.txt", "reviewR4_2.txt", "reviewR4_3.txt", "reviewR4_4.txt"]:
    rm(os.path.join(ROOT, "prompts", f))
for d in ["fixA", "fixB", "fixR4A", "fixR4B"]:
    rm(os.path.join(ROOT, "prompts", d))

# 3) 已 git rm 的空目录残留
for d in [os.path.join(ROOT, "reports"),
          os.path.join(ROOT, "prompts", "fixA"),
          os.path.join(ROOT, "prompts", "fixB")]:
    if os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d); print("RMDIR:", os.path.relpath(d, ROOT))

# 4) 任何遗漏的 .pid / 旧 task/review/fix 散落文件
for pat in ["*.pid"]:
    for p in glob.glob(os.path.join(ROOT, "**", pat), recursive=True):
        rm(p)
for pat in ["task_*.txt", "review_*.txt", "reviewR3_*.txt",
            "fix_report_*.md", "review_batch_*.md", "issues*.json",
            "anchor_problems.json"]:
    for p in glob.glob(os.path.join(ROOT, pat)):
        rm(p)

# 5) 列剩余 prompts/ 内容确认
print("=== prompts/ remaining ===")
for f in sorted(os.listdir(os.path.join(ROOT, "prompts"))):
    print("  ", f)
print("=== top-level ===")
for f in sorted(os.listdir(ROOT)):
    if not f.startswith("."):
        print("  ", f)
