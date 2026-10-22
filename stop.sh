#!/usr/bin/env bash
# 停止 domain-codegen 演示进程
PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for name in server frontend; do
  pidfile="$PLATFORM_DIR/logs/$name.pid"
  if [ -f "$pidfile" ]; then
    kill "$(cat "$pidfile")" 2>/dev/null && echo "已停止 $name" || true
    rm -f "$pidfile"
  fi
done
for port in 3200 8200; do
  fuser -k "${port}/tcp" 2>/dev/null || true
done
echo "domain-codegen 演示进程已全部停止。"
