# Wait for Tier B downloads, then extract, verify, eval, and build 200k cache.
# Usage:
#   .\scripts\postdownload_tierb.ps1 [-DataRoot "D:\bss-audio-data"] [-CacheWorkers 4] [-PollMinutes 5]

param(
    [string]$DataRoot = "D:\bss-audio-data",
    [int]$CacheWorkers = 4,
    [int]$PollMinutes = 5,
    [switch]$DeleteLibri100,
    [switch]$SkipWait
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "postdownload_tierb.log"

function Log([string]$Message) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Invoke-PythonLogged([string[]]$Args) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @Args 2>&1 | ForEach-Object { Log "$_" }
    $ErrorActionPreference = $prev
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed (exit $LASTEXITCODE): $($Args -join ' ')"
    }
}

function Get-ArchiveSize([string]$Path) {
    if (Test-Path $Path) { return (Get-Item $Path).Length }
    return 0
}

function Test-DownloadsReady {
    $dl = Join-Path $DataRoot "downloads"
    $libri = @(
        (Join-Path $dl "train-clean-360.tar.gz"),
        (Join-Path $dl "train-clean-360.part")
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    $vctk = Join-Path $dl "VCTK-Corpus-0.92.zip"
    $daps = Join-Path $dl "daps.tar.gz"

    $ready = @(
        @{ Name = "Libri-360"; Path = $libri; Min = 20GB },
        @{ Name = "VCTK"; Path = $vctk; Min = 10GB },
        @{ Name = "DAPS"; Path = $daps; Min = 14GB }
    )

    $status = @()
    $allOk = $true
    foreach ($item in $ready) {
        $size = Get-ArchiveSize $item.Path
        $ok = $size -ge $item.Min
        if (-not $ok) { $allOk = $false }
        $status += "{0}: {1:N2} GB / {2:N2} GB{3}" -f $item.Name, ($size / 1GB), ($item.Min / 1GB), ($(if ($ok) { " OK" } else { "" }))
    }

    $curlActive = @(Get-Process curl -ErrorAction SilentlyContinue).Count
    return @{
        Ready = ($allOk -and $curlActive -eq 0)
        Status = $status
        CurlActive = $curlActive
    }
}

Set-Location $Root
$env:BSS_DATA_ROOT = $DataRoot
$env:PYTHONUNBUFFERED = "1"

Log "=== Post-download Tier B pipeline started ==="
Log "Data root: $DataRoot"
Log "Log file: $LogFile"

if (-not $SkipWait) {
    Log "Waiting for VCTK + DAPS downloads to finish ..."
    while ($true) {
        $check = Test-DownloadsReady
        Log ("  " + ($check.Status -join " | ") + " | curl: $($check.CurlActive)")
        if ($check.Ready) {
            Log "All downloads complete."
            break
        }
        Start-Sleep -Seconds ($PollMinutes * 60)
    }
} else {
    Log "SkipWait set - proceeding immediately."
}

Log "=== Extract Tier B archives ==="
$extractArgs = @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "scripts\extract_data_tierb.ps1"), "-DataRoot", $DataRoot)
if ($DeleteLibri100) { $extractArgs += "-DeleteLibri100" }
& powershell @extractArgs 2>&1 | ForEach-Object { Log $_ }

Log "=== Verify Tier B downloads/extracts ==="
& powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\verify_downloads_tierb.ps1") -DataRoot $DataRoot 2>&1 |
    ForEach-Object { Log $_ }

Log "=== Generate Tier B eval sets ==="
Invoke-PythonLogged @(
    "scripts/generate_eval_sets.py",
    "--data-root", $DataRoot,
    "--output", (Join-Path $DataRoot "eval\acsim_tierb"),
    "--clips-per-split", "225",
    "--clip-seconds", "4.0"
)

Log "=== Generate 200k training cache ($CacheWorkers workers, ~18h) ==="
$cacheDir = Join-Path $DataRoot "cache\train_acsim_200k"
Invoke-PythonLogged @(
    "scripts/generate_train_cache.py",
    "--config", "configs/sepformer_s_tierb.yaml",
    "--data-root", $DataRoot,
    "--output", $cacheDir,
    "--num-clips", "200000",
    "--workers", "$CacheWorkers"
)

$count = (Get-ChildItem (Join-Path $cacheDir "clip_*.pt") -ErrorAction SilentlyContinue | Measure-Object).Count
Log "Cache clips: $count / 200000"

Log "=== Validate cache ==="
Invoke-PythonLogged @(
    "scripts/verify_cache.py",
    "--cache-dir", $cacheDir,
    "--expected-clips", "200000"
)

Log "=== Post-download Tier B pipeline complete ==="
