# 停止 domain-codegen 演示进程（Windows）
$PLATFORM_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

foreach ($name in @("server", "frontend")) {
    $pidFile = Join-Path $PLATFORM_DIR "logs\$name.pid"
    if (Test-Path $pidFile) {
        $procId = [int](Get-Content $pidFile)
        if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
            Stop-Process -Id $procId -Force
            Write-Host "已停止 $name"
        }
        Remove-Item $pidFile -Force
    }
}

# 兜底：清理仍占用 8200/3200 端口的残留进程
foreach ($port in @(8200, 3200)) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
        $conns | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
            Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        }
    }
}
Write-Host "domain-codegen 演示进程已全部停止。"
