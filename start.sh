#!/usr/bin/env bash
# ============================================================
# 领域知识增强的无人系统代码智能适配平台 · 集成演示一键启动
#
#   编排后端  http://localhost:8200  (FastAPI，统一调度两套知识库流水线)
#   统一前端  http://localhost:3200  (Next.js 生产构建)
#
# 引擎与数据自包含：vendor/herta（协议生成/评测引擎+知识库）、vendor/opalign（样例模型）、vendor/onnx（ONNX 知识库）
# 全程离线可用：无 LLM Key 时预置样例展示预生成结果；评测为本地真实计算。
# ============================================================
set -euo pipefail

PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PLATFORM_DIR/logs"
mkdir -p "$LOG_DIR"

CONDA_SH=""
for cand in "$HOME/anaconda3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
  [ -f "$cand" ] && CONDA_SH="$cand" && break
done
if [ -n "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate demo 2>/dev/null || conda activate base
fi

echo "==> 启动编排后端 (8200)..."
cd "$PLATFORM_DIR/server"
nohup uvicorn app:app --port 8200 > "$LOG_DIR/server.log" 2>&1 &
echo $! > "$LOG_DIR/server.pid"

wait_health () {
  local url="$1" name="$2" n=0
  until curl -sf "$url" > /dev/null 2>&1; do
    n=$((n+1)); [ $n -ge 30 ] && { echo "!! $name 健康检查超时，查看 $LOG_DIR"; exit 1; }
    sleep 1
  done
  echo "    $name 就绪 ✓"
}
wait_health "http://localhost:8200/api/health" "编排后端"

echo "==> 启动前端 (3200)..."
cd "$PLATFORM_DIR/app"
nohup npm start > "$LOG_DIR/frontend.log" 2>&1 &
echo $! > "$LOG_DIR/frontend.pid"
wait_health "http://localhost:3200/" "前端"

echo ""
echo "============================================================"
echo "  演示已就绪： http://localhost:3200"
echo "  停止演示：   $PLATFORM_DIR/stop.sh"
echo "============================================================"
