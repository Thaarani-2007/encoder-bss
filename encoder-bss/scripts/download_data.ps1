# Tier-Min dataset download for AC-SIM fast track (~42 GB)
# Uses parallel curl downloads + resume. Extraction via extract_data.ps1 (tar for large archives).
#
# Usage:
#   .\scripts\download_data.ps1 [-DataRoot "D:\bss-audio-data"] [-Parallel 4] [-SkipExtract]

param(
    [string]$DataRoot = "D:\bss-audio-data",
    [int]$Parallel = 4,
    [switch]$SkipExtract
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Ensure-Dir($path) {
    if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
}

Write-Host "=== BSS-Audio Tier-Min Download (parallel curl) ==="
Write-Host "Data root: $DataRoot  |  parallel jobs: $Parallel"

$dl = Join-Path $DataRoot "downloads"
Ensure-Dir $dl

$jobs = @(
    @{
        Name = "LibriSpeech-train-clean-100"
        Url  = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
        Dest = Join-Path $dl "train-clean-100.part"
        MinBytes = 6GB
    },
    @{
        Name = "WHAM-noise"
        Url  = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
        Dest = Join-Path $dl "wham_noise.zip"
        MinBytes = 17GB
    },
    @{
        Name = "ESC-50"
        Url  = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
        Dest = Join-Path $dl "ESC-50.zip"
        MinBytes = 100MB
    },
    @{
        Name = "MIT-IR-Survey"
        Url  = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
        Dest = Join-Path $dl "mit_ir_audio.zip"
        MinBytes = 10MB
    }
)

$procs = @()
foreach ($item in $jobs) {
    while (($procs | Where-Object { -not $_.HasExited }).Count -ge $Parallel) {
        Start-Sleep -Seconds 2
    }
    $dest = $item.Dest
    $url = $item.Url
    $min = $item.MinBytes
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -lt $min)) {
        $size = (Get-Item $dest).Length
        if ($size -lt 1MB) {
            Write-Host "Removing corrupt stub: $dest ($size bytes)"
            Remove-Item $dest -Force
        } else {
            Write-Host "Resuming partial: $dest ($([math]::Round($size / 1GB, 2)) GB)"
        }
    }
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -ge $min)) {
        Write-Host "SKIP download (complete): $($item.Name)"
        continue
    }
    Write-Host "Launching: $($item.Name)"
    $procs += Start-Process -FilePath "curl.exe" -ArgumentList @(
        "-L", "-C", "-", "--retry", "5", "--retry-delay", "3", "-o", $dest, $url
    ) -PassThru -NoNewWindow
}

if ($procs.Count -gt 0) {
    Write-Host "Waiting for $($procs.Count) parallel curl processes..."
    $procs | Wait-Process
    $failed = @()
    foreach ($p in $procs) {
        if ($p.ExitCode -ne 0) { $failed += $p.Id }
    }
    if ($failed.Count -gt 0) {
        Write-Warning "Some curl processes failed (ids: $($failed -join ', ')). Re-run to resume."
    }
}

if (-not $SkipExtract) {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $ScriptDir "extract_data.ps1") -DataRoot $DataRoot
}

Write-Host ""
Write-Host "=== Done ==="
Write-Host ('Set env: $env:BSS_DATA_ROOT = "' + $DataRoot + '"')
