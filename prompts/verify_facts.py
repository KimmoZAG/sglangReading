#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量核验一组 (file, lineno) 锚点：打印该行内容，便于人工确认报告的 SUGGESTED_FIX 是否真实。"""
import os, sys

SSOT = "/home/kimmo/develop/sglang"

# (file_relative_to_SSOT_python, lineno)
CHECKS = [
    ("sglang/srt/managers/scheduler_components/metrics_reporter.py", 92),
    ("sglang/srt/managers/scheduler_components/pool_stats_observer.py", 142),
    ("sglang/srt/observability/metrics_collector.py", 201),
    ("sglang/srt/observability/metrics_collector.py", 215),
    ("sglang/srt/observability/metrics_collector.py", 1127),
    ("sglang/srt/managers/scheduler_components/profiler_manager.py", 85),
    ("sglang/srt/managers/scheduler_components/profiler_manager.py", 156),
    ("sglang/srt/observability/request_metrics_exporter.py", 72),
    ("sglang/srt/observability/request_metrics_exporter.py", 159),
    ("sglang/srt/sampling/custom_logit_processor.py", 15),
    ("sglang/srt/sampling/custom_logit_processor.py", 24),
    ("sglang/srt/sampling/sampling_params.py", 82),
    ("sglang/srt/layers/sampler.py", 246),
    ("sglang/srt/layers/sampler.py", 277),
    ("sglang/srt/layers/sampler.py", 563),
    ("sglang/srt/configs/model_config.py", 1422),
    ("sglang/srt/configs/model_config.py", 1440),
    ("sglang/srt/layers/quantization/nvfp4_online.py", 32),
    ("sglang/srt/managers/scheduler.py", 2715),
    ("sglang/srt/managers/scheduler.py", 3549),
    ("sglang/srt/managers/schedule_policy.py", 997),
    ("sglang/srt/managers/schedule_policy.py", 1190),
    ("sglang/srt/managers/schedule_policy.py", 1379),
    ("sglang/srt/managers/schedule_policy.py", 1410),
    ("sglang/srt/managers/schedule_policy.py", 1414),
    ("sglang/srt/managers/schedule_batch.py", 1297),
    ("sglang/srt/eplb/eplb_algorithms/deepseek.py", 86),
    ("sglang/srt/eplb/eplb_algorithms/deepseek.py", 171),
    ("sglang/srt/mem_cache/allocator/paged.py", 105),
    ("sglang/srt/debug_utils/dumper.py", 132),
    ("sglang/srt/debug_utils/dumper.py", 167),
    ("sglang/srt/debug_utils/dumper.py", 224),
    ("sglang/srt/debug_utils/dumper.py", 239),
    ("sglang/srt/layers/attention/base_attn_backend.py", 261),
    ("sglang/srt/layers/attention/flashinfer_backend.py", 1253),
    ("sglang/srt/layers/attention/flashinfer_backend.py", 1414),
    ("sglang/srt/test/test_utils.py", 620),
    ("sglang/srt/test/test_utils.py", 668),
    ("sglang/srt/test/test_utils.py", 713),
    ("sglang/srt/model_loader/auto_loader.py", 177),
]

def main():
    base = os.path.join(SSOT, "python")
    for rel, ln in CHECKS:
        p = os.path.join(base, rel)
        if not os.path.exists(p):
            print(f"[MISSING FILE] {rel}:{ln}")
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if 1 <= ln <= len(lines):
            print(f"[OK] {rel}:{ln}: {lines[ln-1].rstrip()}")
        else:
            print(f"[OOB] {rel}:{ln} (file has {len(lines)} lines)")

if __name__ == "__main__":
    main()
