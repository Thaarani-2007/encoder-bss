# Extract Tier B datasets (run after downloads complete)
# Usage: .\scripts\extract_data_tierb.ps1 [-DataRoot "D:\bss-audio-data"] [-Force] [-DeleteLibri100]

param(
    [string]$DataRoot = "D:\bss-audio-data",
    [switch]$Force,
    [switch]$DeleteLibri100
)

$ErrorActionPreference = "Stop"

$LIBRI360_EXPECTED = 20000000000
$VCTK_EXPECTED = 10000000000
$DAPS_EXPECTED = 14000000000

function Ensure-Dir($path) {
    if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
}

function Test-ArchiveIntegrity($archive) {
    Write-Host "Verifying: $archive"
    if ($archive -match '\.tar\.gz$|\.part$') {
        tar -tzf $archive 2>$null | Out-Null
    } elseif ($archive -match '\.zip$') {
        tar -tf $archive 2>$null | Out-Null
    } else {
        return $true
    }
    return $LASTEXITCODE -eq 0
}

function Should-Extract($destDir, [int]$minFiles, [switch]$Force) {
    if ($Force) { return $true }
    if (-not (Test-Path $destDir)) { return $true }
    $count = (Get-ChildItem $destDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    return $count -lt $minFiles
}

$dl = Join-Path $DataRoot "downloads"
$speech = Join-Path $DataRoot "speech"
Ensure-Dir $speech

Write-Host "=== BSS-Audio Tier B Extraction ==="

# --- LibriSpeech train-clean-360 ---
$libri360Archive = @(
    (Join-Path $dl "train-clean-360.part"),
    (Join-Path $dl "train-clean-360.tar.gz")
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $libri360Archive) {
    $libri360Archive = Join-Path $dl "train-clean-360.part"
}
$libriDest = Join-Path $speech "LibriSpeech"
if (-not (Test-Path $libri360Archive) -or (Get-Item $libri360Archive).Length -lt $LIBRI360_EXPECTED) {
    Write-Error "Libri-360 archive incomplete. Run .\scripts\resume_downloads_tierb.ps1"
}
if (-not (Test-ArchiveIntegrity $libri360Archive)) {
    Write-Error "Libri-360 archive corrupt: $libri360Archive"
}
$libri360Marker = Join-Path $libriDest "LibriSpeech/train-clean-360"
if (-not (Test-Path $libri360Marker)) {
    $libri360Marker = Join-Path $libriDest "train-clean-360"
}
if (Should-Extract $libri360Marker 50000 $Force) {
    Write-Host "Extracting LibriSpeech train-clean-360 ..."
    Ensure-Dir $libriDest
    tar -xzf $libri360Archive -C $libriDest
    if ($LASTEXITCODE -ne 0) { throw "Libri-360 extraction failed" }
    $flacs = (Get-ChildItem $libriDest -Recurse -Filter *.flac | Measure-Object).Count
    Write-Host "LibriSpeech total FLAC: $flacs"
} else {
    Write-Host "SKIP Libri-360 (already extracted)"
}

# --- VCTK ---
$vctkZip = @(
    (Join-Path $dl "VCTK-Corpus-0.92.zip"),
    (Join-Path $dl "VCTK-Corpus.zip")
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vctkZip) { $vctkZip = Join-Path $dl "VCTK-Corpus-0.92.zip" }
$vctkDest = Join-Path $speech "VCTK"
if (-not (Test-Path $vctkZip) -or (Get-Item $vctkZip).Length -lt $VCTK_EXPECTED) {
    Write-Warning "VCTK archive incomplete - skipping (run resume_downloads_tierb.ps1)"
} elseif (-not (Test-ArchiveIntegrity $vctkZip)) {
    Write-Warning "VCTK archive corrupt: $vctkZip"
} elseif (Should-Extract $vctkDest 10000 $Force) {
    Write-Host "Extracting VCTK (tar) ..."
    if (Test-Path $vctkDest) { Remove-Item $vctkDest -Recurse -Force }
    Ensure-Dir $vctkDest
    tar -xf $vctkZip -C $vctkDest
    if ($LASTEXITCODE -ne 0) { throw "VCTK extraction failed" }
    $wavs = (Get-ChildItem $vctkDest -Recurse -Filter *.wav -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($wavs -lt 1000) {
        $flacs = (Get-ChildItem $vctkDest -Recurse -Filter *.flac -ErrorAction SilentlyContinue | Measure-Object).Count
        $wavs += $flacs
    }
    Write-Host "VCTK audio files: $wavs"
} else {
    Write-Host "SKIP VCTK (already extracted)"
}

# --- DAPS (Zenodo full archive; keep produced speech only) ---
$dapsArchive = @(
    (Join-Path $dl "daps.tar.gz"),
    (Join-Path $dl "daps_produced.tar.gz")
) | Where-Object { (Test-Path $_) -and ((Get-Item $_).Length -ge $DAPS_EXPECTED) } | Select-Object -First 1
$dapsDest = Join-Path $speech "DAPS"
if ($dapsArchive) {
    if (Test-ArchiveIntegrity $dapsArchive) {
        if (Should-Extract $dapsDest 500 $Force) {
            Write-Host "Extracting DAPS (produced speech only) ..."
            $tmp = Join-Path $DataRoot "tmp_daps_extract"
            if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
            Ensure-Dir $tmp
            tar -xzf $dapsArchive -C $tmp
            if ($LASTEXITCODE -ne 0) { throw "DAPS extraction failed" }
            $producedDir = Get-ChildItem $tmp -Recurse -Directory -Filter "produced" -ErrorAction SilentlyContinue | Select-Object -First 1
            if (-not $producedDir) {
                Write-Warning "DAPS produced folder not found in archive"
            } else {
                if (Test-Path $dapsDest) { Remove-Item $dapsDest -Recurse -Force }
                Ensure-Dir $dapsDest
                Copy-Item (Join-Path $producedDir.FullName "*") $dapsDest -Recurse -Force
                $dapsWavs = (Get-ChildItem $dapsDest -Recurse -Filter *.wav | Measure-Object).Count
                Write-Host "DAPS produced WAV files: $dapsWavs"
            }
            Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            Write-Host "SKIP DAPS (already extracted)"
        }
    } else {
        Write-Warning "DAPS archive corrupt: $dapsArchive"
    }
} else {
    Write-Warning "DAPS archive not found or incomplete (optional; Zenodo ~16 GB)"
}

# --- Delete Libri-100 after Libri-360 verified ---
if ($DeleteLibri100) {
    $libri100 = Join-Path $libriDest "train-clean-100"
    if (Test-Path $libri100) {
        $count360 = (Get-ChildItem (Join-Path $libriDest "train-clean-360") -Recurse -Filter *.flac -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($count360 -ge 50000) {
            Write-Host "Deleting Libri-100 (superseded by Libri-360) ..."
            Remove-Item $libri100 -Recurse -Force
            $arch100 = Join-Path $dl "train-clean-100.part"
            if (Test-Path $arch100) { Remove-Item $arch100 -Force }
            $arch100gz = Join-Path $dl "train-clean-100.tar.gz"
            if (Test-Path $arch100gz) { Remove-Item $arch100gz -Force }
        } else {
            Write-Warning "Libri-360 not verified ($count360 FLAC) - not deleting Libri-100"
        }
    }
}

Write-Host ""
Write-Host "=== Tier B extraction complete ==="
