# Phase 1b orchestration — run steps in order (non-destructive except Libri-100 delete flag)
# Usage:
#   .\scripts\phase1b_run.ps1 -Step download|extract|eval|cache|validate|all

param(
    [ValidateSet("download", "extract", "eval", "cache", "validate", "preflight", "postdownload", "all")]
    [string]$Step = "preflight",
    [string]$DataRoot = "D:\bss-audio-data",
    [int]$CacheWorkers = 4,
    [switch]$DeleteLibri100
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe"
$env:BSS_DATA_ROOT = $DataRoot

function Run-Preflight {
    Write-Host "=== Phase 1b preflight ==="
    Set-Location $Root
    & $Python -m pytest tests/ -q --tb=line
    & $Python scripts/validate_sepformer_s.py
    & $Python scripts/smoke_test_gpu.py --config configs/sepformer_s_tierb.yaml
    Write-Host "Preflight OK"
}

function Run-Download {
    Write-Host "=== Tier B download (keep window open ~22h) ==="
    & powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\resume_downloads_tierb.ps1") -DataRoot $DataRoot
}

function Run-Extract {
    Write-Host "=== Tier B extract ==="
    if ($DeleteLibri100) {
        & powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\extract_data_tierb.ps1") -DataRoot $DataRoot -DeleteLibri100
    } else {
        & powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\extract_data_tierb.ps1") -DataRoot $DataRoot
    }
    & powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\verify_downloads_tierb.ps1") -DataRoot $DataRoot
}

function Run-Eval {
    Write-Host "=== Regenerate Tier B eval sets ==="
    & $Python scripts/generate_eval_sets.py `
        --data-root $DataRoot `
        --output (Join-Path $DataRoot "eval\acsim_tierb") `
        --clips-per-split 225 `
        --clip-seconds 4.0
}

function Run-Cache {
    Write-Host "=== Generate 200k training cache (~18h) ==="
    & $Python scripts/generate_train_cache.py `
        --config configs/sepformer_s_tierb.yaml `
        --data-root $DataRoot `
        --output (Join-Path $DataRoot "cache\train_acsim_200k") `
        --num-clips 200000 `
        --workers $CacheWorkers
}

function Run-Validate {
    Write-Host "=== Validate cache + eval ==="
    $cacheDir = Join-Path $DataRoot "cache\train_acsim_200k"
    $count = (Get-ChildItem (Join-Path $cacheDir "clip_*.pt") -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "Cache clips: $count / 200000"
    & $Python scripts/verify_cache.py --cache-dir (Join-Path $DataRoot "cache\train_acsim_200k") --expected-clips 200000
}

function Run-PostDownload {
    Write-Host "=== Wait for downloads, then extract + eval + 200k cache ==="
    $args = @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "scripts\postdownload_tierb.ps1"), "-DataRoot", $DataRoot, "-CacheWorkers", $CacheWorkers)
    if ($DeleteLibri100) { $args += "-DeleteLibri100" }
    & powershell @args
}

switch ($Step) {
    "preflight" { Run-Preflight }
    "download"  { Run-Download }
    "extract"   { Run-Extract }
    "eval"      { Run-Eval }
    "cache"     { Run-Cache }
    "validate"  { Run-Validate }
    "postdownload" { Run-PostDownload }
    "all"       {
        Run-Preflight
        Run-Download
        Run-Extract
        Run-Eval
        Run-Cache
        Run-Validate
    }
}
