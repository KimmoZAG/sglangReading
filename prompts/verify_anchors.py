#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机械校验所有文档锚点：文件是否存在、行号是否越界、区间是否升序、是否占位。
不依赖 LLM，输出可定位的违规清单，供直接修复。"""
import os, re, glob

SSOT = "/home/kimmo/develop/sglang"
ROOT = "/home/kimmo/develop/sglangReading"

# 匹配锚点：路径以 .py/.mdx/.md 结尾，后跟 :Lp 或 :Lp-Lq 或 #Lp，允许 + 占位
# 路径限定为 ASCII 路径字符，避免吞入中文/括号前言
ANCHOR_RE = re.compile(
    r"`?([A-Za-z0-9_./-]+\.(?:py|mdx|md))(?::|#)L?(\d+)(?:-L?(\d+|\d+x|\?\?\?|XXX|xxx|Lx|Ly))?`?"
)

def _glob_first(basename):
    import glob as _g
    hits = _g.glob(os.path.join(SSOT, "**", basename), recursive=True)
    if not hits:
        return None
    # 优先 srt / lang 下的实现文件
    hits.sort(key=lambda p: (0 if "/srt/" in p else (1 if "/lang/" in p else 2), p))
    return hits[0]

def resolve(rel):
    # 返回绝对路径或 None
    rel = rel.strip("`").strip()
    if rel.startswith("python/sglang/"):
        c = os.path.join(SSOT, rel)
        return c if os.path.exists(c) else None
    if rel.startswith("python/") or rel.startswith("srt/"):
        bases = [SSOT,
                 os.path.join(SSOT, "python"),
                 os.path.join(SSOT, "python", "sglang"),
                 os.path.join(SSOT, "python", "sglang", "srt")]
        for base in bases:
            c = os.path.join(base, rel)
            if os.path.exists(c):
                return c
        return None
    if "/" in rel:
        # 优先级：仓库根精确 > srt/<rel> > sglang/<rel> > 全树 glob(取最长)
        cands = [
            os.path.join(SSOT, rel),
            os.path.join(SSOT, "python", "sglang", "srt", rel),
            os.path.join(SSOT, "python", "sglang", rel),
        ]
        for c in cands:
            if os.path.exists(c):
                return c
        import glob as _g
        hits = _g.glob(os.path.join(SSOT, "**", rel), recursive=True)
        if hits:
            hits.sort(key=lambda p: -sum(1 for _ in open(p, errors="ignore")))
            return hits[0]
        return None
    # 裸文件名：不可靠，交 LLM 复核，这里不解析
    return None

def main():
    problems = []
    docs = [f for f in glob.glob(f"{ROOT}/docs/**/*.md", recursive=True)
            if "/reports/" not in f and "_openq" not in f]
    for f in docs:
        text = open(f, encoding="utf-8").read()
        for m in ANCHOR_RE.finditer(text):
            raw = m.group(0)
            path = m.group(1)
            lp = m.group(2)
            lq = m.group(3)
            # 跳过不可靠/元信息锚点：裸文件名、__init__.py、含 ... 的占位、以及指向本仓库的元路径
            bn = os.path.basename(path)
            if "/" not in path:
                continue
            if bn == "__init__.py":
                continue
            if "..." in path:
                continue
            if path.startswith("sglangReading/") or path.startswith("docs/"):
                continue
            # 占位检测
            if lq and any(t in lq for t in ("x", "?", "XXX", "xxx", "Lx", "Ly")):
                problems.append((f, raw, "PLACEHOLDER", "非精确行号"))
                continue
            abs = resolve(path)
            if abs is None:
                problems.append((f, raw, "UNRESOLVABLE_PATH", "文件不存在于 SSOT"))
                continue
            total = sum(1 for _ in open(abs, encoding="utf-8", errors="ignore"))
            try:
                p = int(lp.lstrip("L"))
                q = int((lq or lp).lstrip("L"))
            except ValueError:
                problems.append((f, raw, "BAD_NUM", "行号解析失败"))
                continue
            if p > q:
                problems.append((f, raw, "REVERSED", f"起止颠倒 {p}>{q} (file {total}L)"))
            elif q > total:
                problems.append((f, raw, "OUT_OF_RANGE", f"行号 {q} 越界 (file {total}L)"))
            elif p > total:
                problems.append((f, raw, "OUT_OF_RANGE", f"行号 {p} 越界 (file {total}L)"))
    # 输出
    print(f"扫描文档: {len(docs)} | 问题锚点: {len(problems)}\n")
    by_type = {}
    for f, raw, typ, msg in problems:
        by_type.setdefault(typ, []).append((f, raw, msg))
    for typ in ["PLACEHOLDER","UNRESOLVABLE_PATH","REVERSED","OUT_OF_RANGE","BAD_NUM"]:
        items = by_type.get(typ, [])
        if not items: continue
        print(f"## {typ} ({len(items)})")
        for f, raw, msg in items:
            rel = os.path.relpath(f, ROOT)
            print(f"  {rel} :: {raw}  -> {msg}")
    # 写 JSON 供后续修复
    import json
    json.dump([{"file": os.path.relpath(f, ROOT), "anchor": raw, "type": typ, "msg": msg}
               for f, raw, typ, msg in problems],
              open(f"{ROOT}/reports/anchor_problems.json", "w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
