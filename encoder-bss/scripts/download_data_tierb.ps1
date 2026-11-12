# Tier B dataset download for Phase 1b (~70 GB add-on)
# Usage:
#   .\scripts\download_data_tierb.ps1 [-DataRoot "D:\bss-audio-data"] [-Parallel 3] [-SkipExtract]
#
# Run in a dedicated PowerShell window — background tasks may kill curl transfers.

param(
    [string]$DataRoot = "D:\bss-audio-data",
    [int]$Parallel = 3,
    [switch]$SkipExtract
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Ensure-Dir($path) {
    if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
}

Write-Host "=== BSS-Audio Tier B Download (parallel curl) ==="
Write-Host "Data root: $DataRoot  |  parallel jobs: $Parallel"
Write-Host "Estimated download @ 25 Mbps: ~20-24 h for ~70 GB"

$dl = Join-Path $DataRoot "downloads"
Ensure-Dir $dl

$jobs = @(
    @{
        Name = "LibriSpeech-train-clean-360"
        Url  = "https://www.openslr.org/resources/12/train-clean-360.tar.gz"
        Dest = Join-Path $dl "train-clean-360.tar.gz"
        MinBytes = 20GB
    },
    @{
        Name = "VCTK-Corpus"
        Dest = Join-Path $dl "VCTK-Corpus-0.92.zip"
        MinBytes = 10GB
        Url  = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    },
    @{
        Name = "DAPS-full"
        Url  = "https://zenodo.org/record/4660670/files/daps.tar.gz?download=1"
        Dest = Join-Path $dl "daps.tar.gz"
        MinBytes = 14GB
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
    & powershell -ExecutionPolicy Bypass -File (Join-Path $ScriptDir "extract_data_tierb.ps1") -DataRoot $DataRoot
}

Write-Host ""
Write-Host "=== Tier B download done ==="
Write-Host "Next: .\scripts\verify_downloads_tierb.ps1 -DataRoot $DataRoot"
