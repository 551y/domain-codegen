# ============================================================
# 领域知识增强的无人系统代码智能适配平台 · 集成演示一键启动（Windows）
#
#   编排后端  http://localhost:8200  (FastAPI，统一调度两套知识库流水线)
#   统一前端  http://localhost:3200  (Next.js 生产构建)
#
# 引擎与数据自包含：vendor/herta（协议生成/评测引擎+知识库）、vendor/opalign（样例模型）、vendor/onnx（ONNX 知识库）
# 全程离线可用：无 LLM Key 时预置样例展示预生成结果；评测为本地真实计算。
#
# 若提示“禁止运行脚本”：powershell -ExecutionPolicy Bypass -File .\start.ps1
# ============================================================

$PLATFORM_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$LOG_DIR = Join-Path $PLATFORM_DIR "logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

# ---- 检查 node ----
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "!! 未找到 node，请先安装 Node.js LTS: https://nodejs.org"
    exit 1
}

# ---- 激活 conda 环境（优先 demo，失败回退 base；找不到 conda 则用当前环境）----
$condaExe = $null
foreach ($cand in @(
    "$HOME\miniconda3\Scripts\conda.exe",
    "$HOME\anaconda3\Scripts\conda.exe",
    "C:\ProgramData\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\anaconda3\Scripts\conda.exe")) {
    if (Test-Path $cand) { $condaExe = $cand; break }
}
if (-not $condaExe) {
    $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($cmd) { $condaExe = $cmd.Source }
}
if ($condaExe) {
    (& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
    conda activate demo
    if ($LASTEXITCODE -ne 0) { conda activate base }
} else {
    Write-Host "!! 未找到 conda，将使用当前 Python 环境（需已安装依赖）"
}

function Wait-Health {
    param($Url, $Name)
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            Write-Host "    $Name 就绪 ✓"
            return
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    Write-Host "!! $Name 健康检查超时，查看 $LOG_DIR"
    exit 1
}

# ---- 后端（8200）----
Write-Host "==> 启动编排后端 (8200)..."
$serverDir = Join-Path $PLATFORM_DIR "server"
$proc = Start-Process -FilePath "uvicorn" -ArgumentList "app:app", "--port", "8200" `
    -WorkingDirectory $serverDir -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $LOG_DIR "server.log") `
    -RedirectStandardError (Join-Path $LOG_DIR "server.err.log")
$proc.Id | Out-File (Join-Path $LOG_DIR "server.pid") -Encoding ascii
Wait-Health "http://localhost:8200/api/health" "编排后端"

# ---- 前端（3200）：直接用 node 拉起 next，保证拿到可停止的真实 PID ----
Write-Host "==> 启动前端 (3200)..."
$appDir = Join-Path $PLATFORM_DIR "app"
$nextBin = Join-Path $appDir "node_modules\next\dist\bin\next"
if (-not (Test-Path $nextBin)) {
    Write-Host "!! 未找到前端依赖，请先执行: cd app; npm install --legacy-peer-deps; npm run build; cd .."
    exit 1
}
$proc = Start-Process -FilePath "node" -ArgumentList "`"$nextBin`"", "start", "-p", "3200" `
    -WorkingDirectory $appDir -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $LOG_DIR "frontend.log") `
    -RedirectStandardError (Join-Path $LOG_DIR "frontend.err.log")
$proc.Id | Out-File (Join-Path $LOG_DIR "frontend.pid") -Encoding ascii
Wait-Health "http://localhost:3200/" "前端"

Write-Host ""
Write-Host "============================================================"
Write-Host "  演示已就绪： http://localhost:3200"
Write-Host "  停止演示：   $PLATFORM_DIR\stop.ps1"
Write-Host "============================================================"
