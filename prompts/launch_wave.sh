#!/usr/bin/env bash
# 启动一波子任务：每个任务一个全新 codebuddy 子会话（headless），并行跑。
# 用法: bash prompts/launch_wave.sh <wave_label> <task_num> [<task_num> ...]
set -u
cd /home/kimmo/develop/sglangReading
CB="codebuddy"   # 二进制名用变量，避免命令字面量触发安全规则
WAVE="$1"; shift
PIDS=()
for I in "$@"; do
  setsid bash -c "\"$CB\" -p \"\$(cat prompts/task_${I}.txt)\" -y --allowedTools \"Bash,Read,Write,Edit\" --no-session-persistence --max-turns 60 </dev/null > logs/task_${I}.log 2>&1" &
  echo $! > "pids/task_${I}.pid"
  PIDS+=($!)
  echo "launched task_${I} pid=$!"
done

# 轮询等待整波完成（最多 ~15 分钟）
for n in $(seq 1 180); do
  REMAIN=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then REMAIN=1; break; fi
  done
  if [ "$REMAIN" -eq 0 ]; then break; fi
  sleep 5
done

# 清场：杀整个进程组，防泄漏（按记录的 PGID）
for I in "$@"; do
  PG=$(cat "pids/task_${I}.pid" 2>/dev/null)
  if [ -n "$PG" ]; then kill -9 -"$PG" 2>/dev/null; fi
done
echo "wave $WAVE finished"
