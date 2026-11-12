# Start training detached with file logging (avoids terminal pipe deadlocks).
param(
    [string]$Config = "configs\sepformer_s_tierb.yaml",
    [string]$Resume = "",
    [string]$InitCheckpoint = "",
    [string]$DataRoot = "D:\bss-audio-data",
    [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:BSS_DATA_ROOT = $DataRoot
$env:PYTHONUNBUFFERED = "1"
$env:WANDB_CONSOLE = "off"

$Python = "C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe"
if (-not $LogFile) {
    $runName = ([System.IO.Path]::GetFileNameWithoutExtension($Config))
    $LogFile = Join-Path $LogDir "train-$runName.log"
}

$Args = @("-u", "scripts\train.py", "--config", $Config)
if ($Resume) {
    $Args += @("--resume", $Resume)
}
if ($InitCheckpoint) {
    $Args += @("--init-checkpoint", $InitCheckpoint)
}

Get-Process python* -ErrorAction SilentlyContinue |
    Where-Object { (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine -like "*train.py*" } |
    ForEach-Object { Stop-Process -Id $_.Id -Force }

$proc = Start-Process -FilePath $Python `
    -ArgumentList $Args `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $LogFile `
    -PassThru `
    -WindowStyle Hidden

Write-Host "Training started PID $($proc.Id)"
Write-Host "Config: $Config"
Write-Host "Log: $LogFile"
Write-Host "Tail: Get-Content '$LogFile' -Wait -Tail 20"
