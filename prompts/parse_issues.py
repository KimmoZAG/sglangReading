#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析 4 份评阅报告里的 ###I 块，按 FILE 聚合，生成每个文件的修复子任务 prompt（fixA/fixB 两批）。
每个 fixer 只改自己那一篇，互不冲突。
"""
import os, re, json, glob

SSOT = "/home/kimmo/develop/sglang"
OUT = "/home/kimmo/develop/sglangReading/docs"
COMMIT = "e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7"

KEYS = ["FILE", "SEVERITY", "TYPE", "DETAIL", "SUGGESTED_FIX"]

def parse_blocks(text):
    # 切分 ###I 块
    parts = re.split(r"^###I\s*$", text, flags=re.M)
    blocks = []
    for p in parts[1:]:
        d = {}
        cur = None
        for line in p.splitlines():
            m = re.match(r"^(FILE|SEVERITY|TYPE|DETAIL|SUGGESTED_FIX):\s?(.*)$", line)
            if m:
                cur = m.group(1)
                d[cur] = m.group(2)
            elif cur:
                d[cur] += "\n" + line
        if d.get("FILE"):
            blocks.append(d)
    return blocks

def main():
    blocks = []
    for f in sorted(glob.glob("reports/review_batch_*.md")):
        blocks += parse_blocks(open(f, encoding="utf-8").read())
    # 按 FILE 聚合
    grouped = {}
    for b in blocks:
        grouped.setdefault(b["FILE"].replace("docs/", ""), []).append(b)
    os.makedirs("prompts/fixA", exist_ok=True)
    os.makedirs("prompts/fixB", exist_ok=True)
    items = list(grouped.items())
    # 按问题数降序，均衡分到 A/B
    items.sort(key=lambda x: -len(x[1]))
    A, B = [], []
    sa = sb = 0
    for path, iss in items:
        if sa <= sb:
            A.append((path, iss)); sa += len(iss)
        else:
            B.append((path, iss)); sb += len(iss)
    def write_batch(batch, dname):
        for i, (path, iss) in enumerate(batch, 1):
            p = os.path.join(dname, f"fix_{i:02d}.txt")
            issues_txt = ""
            for j, b in enumerate(iss, 1):
                issues_txt += f"\n### 问题 {j}（SEVERITY={b.get('SEVERITY','?')} TYPE={b.get('TYPE','?')}）\n"
                issues_txt += f"- DETAIL: {b.get('DETAIL','').strip()}\n"
                issues_txt += f"- SUGGESTED_FIX: {b.get('SUGGESTED_FIX','').strip()}\n"
            prompt = f"""你是 SGLang 文档修复工程师。只改【一个】文档文件里的源码锚点错误，不改动其它内容，不编造。

【唯一事实来源 SSOT】{SSOT}（commit {COMMIT}）。所有修正必须先用 grep/Read 在 SSOT 实测确认。

【目标文件（绝对路径）】{OUT}/{path}

【本文件被评阅出的问题清单】
{issues_txt}

【修复步骤（严格执行）】
1) 用 Read 工具读完整篇 {OUT}/{path}。
2) 对清单里每一条：
   - 用 `grep -n "<符号>" {SSOT}/python/sglang/srt/<对应文件>` 找到该符号的真实定义行（若 SUGGESTED_FIX 已给正确锚点，仍要用 grep 复核行号是否真实）。
   - 在文档里定位那条错误的锚点字符串（从 DETAIL 引用的错误锚点原文），用 Edit 工具替换为【已 grep 复核】的正确锚点。
   - 规则：
     * 路径错误（如 engine.py 应为 scheduler.py）：改正路径与行号。
     * 区间颠倒（如 986-917）：改为升序 917-986。
     * 符号名不匹配（如 wait_for_ready 实为 _wait_for_scheduler_ready、match_prefix_for_req 实为 _match_prefix_and_lock）：改正符号名，行号保留建议值并复核。
     * 缺前缀的裸文件名（如 `server_args.py:L218`）：补成完整相对 SSOT 路径 `python/sglang/srt/server_args.py:L218`（若该模块确实在 srt 下；用 ls/grep 确认存在）。
     * 占位/非精确锚点（如 `Lxxx+`）：用 grep 算出真实行号后替换。
3) 严禁新增/删除非锚点内容；保持 mermaid 代码块完整、不引入新 TODO。
4) 不运行 mkdocs build（本机沙箱会误报）；只保证锚点真实。

【验收（完成后）】
- 报告你改了哪些锚点（旧→新）到文件：{OUT}/../reports/fix_report_{path.replace('/','_')}.md
- 再次 grep 复核：每条修正后的锚点行号处确为文中引用的符号。
"""
            open(p, "w", encoding="utf-8").write(prompt)
    write_batch(A, "prompts/fixA")
    write_batch(B, "prompts/fixB")
    # 汇总 JSON
    json.dump({k: len(v) for k, v in grouped.items()}, open("reports/issues.json", "w"), ensure_ascii=False, indent=2)
    print(f"files with issues: {len(grouped)} | A={len(A)} B={len(B)} | total issues={len(blocks)}")
    print("A:", [x[0] for x in A])
    print("B:", [x[0] for x in B])

if __name__ == "__main__":
    main()
