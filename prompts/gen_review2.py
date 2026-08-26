#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 3：只对本次已改动（git diff）的文档重新评审，得到真实剩余缺陷。"""
import os, subprocess

SSOT = "/home/kimmo/develop/sglang"
OUT = "/home/kimmo/develop/sglangReading/docs"
COMMIT = "e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7"

changed = subprocess.check_output(
    ["git", "diff", "--name-only"], cwd="."
).decode().split()
docs = [c for c in changed if c.startswith("docs/") and c.endswith(".md")]
# 排除附录/报告等噪音
docs = [d for d in docs if "appendix/_openq" not in d and "/reports/" not in d]
docs = sorted(set(docs))
print("changed docs:", docs)

PREAMBLE = f"""你是严格的代码文档评阅人。任务【只评审、不修改】，产出结构化报告。
【唯一事实来源 SSOT】{SSOT}（commit {COMMIT}）。锚点真伪必须在此目录 grep/Read 核实，严禁凭记忆。
【评审对象】
"""

def write(n, files):
    body = "\n".join(f"{i+1}. {OUT}/{f}" for i, f in enumerate(files))
    body += f"""

【硬性评审流程】
1) Read 通读每篇；提取全部锚点，正则 `python/sglang/[A-Za-z0-9_./-]+\\.py[:#]L?\\d+(-\\d+)?`，也含省略前缀写法（如 `managers/scheduler.py:L120`、`metrics_collector.py:238`）。
2) 逐锚点真伪核验：
   - 解析路径：缺 `python/sglang/srt/` 前缀则按 `python/sglang/srt/<path>` 解析；缺 `python/sglang/` 前缀（如 `contribution_guide.mdx`）按 SSOT 内真实位置解析（用 find/grep 定位），找不到则标【路径无法解析】。
   - 找文中引用符号（类名/函数名/字段名），`grep -n "<符号>" <解析路径>` 确认命中行落在标注 [Lp,Lq]。
   - 判定：OK / 漂移(行号差) / 伪造(文件不存在或符号搜不到) / 占位(Lx/Ly/XXX/Lxxx+)。
3) 深度：是否回答 What/Why/How/坑？标出泛泛而谈段。
4) mermaid：节点名是否真实类/函数（grep `class <名>`/`def <名>` 验证），否则标【图类名虚构】。
【输出】只写文件（绝对路径）：{OUT}/../reports/reviewR3_{n}.md（用 Write，勿改 docs/ 与 SSOT）。
报告结构：
# 评阅报告 R3-{n}
## 总体结论（评分0-10 + 最严重问题一句话）
## 逐篇
### <相对路径>
- 评分: x/10
- 锚点: OK=N 漂移=M 伪造=P 占位=Q
- 深度问题: <引用>
- mermaid问题: <引用>
## ISSUES（###I 分隔，机器可解析）
###I
FILE: <相对路径>
SEVERITY: high|medium|low
TYPE: anchor_drift|anchor_fake|anchor_placeholder|path_unresolvable|shallow|mermaid_fake|missing_section
DETAIL: <错误锚点原文 + 你的判断>
SUGGESTED_FIX: <正确锚点区间/应补内容>
###I
...
## TOP5（最该优先修的5条）
"""
    p = f"prompts/reviewR3_{n}.txt"
    open(p, "w", encoding="utf-8").write(PREAMBLE + body)
    print("wrote", p)

if __name__ == "__main__":
    half = (len(docs) + 1) // 2
    write(1, docs[:half])
    write(2, docs[half:])
