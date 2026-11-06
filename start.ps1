$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Write-Host 'DepthSight http://127.0.0.1:8000' -ForegroundColor Cyan
if (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
    & '.\.venv\Scripts\python.exe' app.py
} else {
    python app.py
}
