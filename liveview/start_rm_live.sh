#!/usr/bin/env bash
# 启动/重启 RM 3v3 训练片段观察窗（默认 8768）
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=${1:-8768}
LOG=${RM_LIVE_LOG:-/tmp/alphaRM_JAX_liveview.log}
OLD=$(ss -ltnp 2>/dev/null | grep ":$PORT" | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
if [ -n "${OLD:-}" ]; then
  echo ">> 停掉占 $PORT 的旧进程 pid=$OLD"
  kill "$OLD" 2>/dev/null || true
  sleep 1
fi
nohup python3 "$REPO_ROOT/liveview/serve_rm_live.py" --port "$PORT" --bind 0.0.0.0 \
  > "$LOG" 2>&1 &
sleep 2
cat "$LOG"
echo ">> 打开： http://localhost:$PORT/"
for p in "/" "/files" "/rm_live.json" "/status.json"; do
  printf "   %-16s %s\n" "$p" "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT$p")"
done
