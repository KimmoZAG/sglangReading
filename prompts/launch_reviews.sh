#!/usr/bin/env bash
# 启动 R4 评阅：4 个并行 codebuddy 子会话，覆盖全部 35 篇文档的符号级/深度/mermaid 审查。
set -u
cd /home/kimmo/develop/sglangReading
mkdir -p logs pids reports
CB="codebuddy"
for I in 1 2 3 4; do
  setsid bash -c "\"$CB\" -p \"\$(cat prompts/reviewR4_${I}.txt)\" -y --allowedTools \"Bash,Read,Write,Edit\" --no-session-persistence --max-turns 40 </dev/null > logs/reviewR4_${I}.log 2>&1" &
  echo $! > "pids/reviewR4_${I}.pid"
  echo "launched reviewR4_${I} pid=$(cat pids/reviewR4_${I}.pid)"
done
echo "all R4 reviews launched; poll reports/reviewR4_*.md"
