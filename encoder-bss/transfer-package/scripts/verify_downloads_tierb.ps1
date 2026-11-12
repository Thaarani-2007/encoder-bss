# Verify Tier B download sizes and archive integrity
# Usage: .\scripts\verify_downloads_tierb.ps1 [-DataRoot "D:\bss-audio-data"]

param(
    [string]$DataRoot = "D:\bss-audio-data"
)

$ErrorActionPreference = "Continue"
$dl = Join-Path $DataRoot "downloads"
$speech = Join-Path $DataRoot "speech"

Write-Host "=== Tier B Download Verification ==="

function Check-Archive($path, $minBytes, $name) {
    if (-not (Test-Path $path)) {
        Write-Host "$name : MISSING"
        return $false
    }
    $size = (Get-Item $path).Length
    $sizeGb = [math]::Round($size / 1GB, 2)
    if ($size -lt $minBytes) {
        Write-Host "$name : INCOMPLETE ($sizeGb GB / min $([math]::Round($minBytes/1GB,2)) GB)"
        return $false
    }
    if ($path -match '\.tar\.gz$|\.part$') {
        tar -tzf $path 2>$null | Select-Object -First 1 | Out-Null
    } elseif ($path -match '\.zip$') {
        tar -tf $path 2>$null | Select-Object -First 1 | Out-Null
    }
    $integrity = if ($LASTEXITCODE -eq 0) { "INTEGRITY OK" } else { "CORRUPT - re-download" }
    Write-Host "$name : $sizeGb GB — $integrity"
    return $LASTEXITCODE -eq 0
}

function Resolve-Libri360Archive($dlDir) {
    foreach ($name in @("train-clean-360.part", "train-clean-360.tar.gz")) {
        $path = Join-Path $dlDir $name
        if (Test-Path $path) { return $path }
    }
    return Join-Path $dlDir "train-clean-360.part"
}

$ok = $true
$ok = (Check-Archive (Resolve-Libri360Archive $dl) 20GB "Libri-360") -and $ok
function Resolve-VctkArchive($dlDir) {
    foreach ($name in @("VCTK-Corpus-0.92.zip", "VCTK-Corpus.zip")) {
        $path = Join-Path $dlDir $name
        if (Test-Path $path) { return $path }
    }
    return Join-Path $dlDir "VCTK-Corpus-0.92.zip"
}

$ok = (Check-Archive (Resolve-VctkArchive $dl) 10GB "VCTK") -and $ok
$dapsPath = @(
    (Join-Path $dl "daps.tar.gz"),
    (Join-Path $dl "daps_produced.tar.gz")
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($dapsPath) {
    $ok = (Check-Archive $dapsPath 14GB "DAPS") -and $ok
} else {
    Write-Host "DAPS : OPTIONAL (not downloaded)"
}

Write-Host ""
Write-Host "=== Extracted counts ==="
$libri = Join-Path $speech "LibriSpeech"
if (Test-Path $libri) {
    $flacs = (Get-ChildItem $libri -Recurse -Filter *.flac -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "LibriSpeech FLAC: $flacs (expect ~104k with 360, or ~28k with 100 only)"
}
$vctk = Join-Path $speech "VCTK"
if (Test-Path $vctk) {
    $vctkWavs = (Get-ChildItem $vctk -Recurse -Filter *.wav -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "VCTK WAV: $vctkWavs"
}
$daps = Join-Path $speech "DAPS"
if (Test-Path $daps) {
    $dapsWavs = (Get-ChildItem $daps -Recurse -Filter *.wav -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "DAPS WAV: $dapsWavs"
}

if ($ok) { Write-Host "`nTier B archives: OK (or in progress if sizes below min)" }
else { Write-Host "`nTier B archives: INCOMPLETE — run .\scripts\resume_downloads_tierb.ps1" }
