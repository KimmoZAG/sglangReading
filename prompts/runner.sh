#!/usr/bin/env bash
# 通用子会话启动器：每个 prompt 文件拉起一个独立 codebuddy 子会话，并行跑，整波等完再按进程组清理。
# 用法: bash prompts/runner.sh <label> <promptfile> [<promptfile> ...]
set -u
cd /home/kimmo/develop/sglangReading
CB="codebuddy"
WAVE="$1"; shift
PIDS=()
for PF in "$@"; do
  BN=$(basename "$PF" .txt)
  setsid bash -c "\"$CB\" -p \"\$(cat '$PF')\" -y --allowedTools \"Bash,Read,Write,Edit\" --no-session-persistence --max-turns 35 </dev/null > logs/${BN}.log 2>&1" &
  echo $! > "pids/${BN}.pid"
  PIDS+=($!)
  echo "launched $BN pid=$!"
done
for n in $(seq 1 200); do
  REMAIN=0
  for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && { REMAIN=1; break; }; done
  [ "$REMAIN" -eq 0 ] && break
  sleep 5
done
for pid in "${PIDS[@]}"; do kill -9 -"$pid" 2>/dev/null; done
echo "wave $WAVE done"
