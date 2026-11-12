# Extract Tier-Min datasets (run after downloads complete)
# Usage: .\scripts\extract_data.ps1 [-DataRoot "D:\bss-audio-data"] [-Force]

param(
    [string]$DataRoot = "D:\bss-audio-data",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$LIBRI_EXPECTED = 6387309499
$WHAM_EXPECTED = 18194930748

function Ensure-Dir($path) {
    if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
}

function Get-BestLibriArchive($dl) {
    $candidates = @(
        (Join-Path $dl "train-clean-100.part"),
        (Join-Path $dl "train-clean-100.tar.gz")
    )
    $best = $null
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $size = (Get-Item $c).Length
            if ($size -ge $LIBRI_EXPECTED -and ($null -eq $best -or $size -gt $best.Size)) {
                $best = @{ Path = $c; Size = $size }
            }
        }
    }
    return $best
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
$noise = Join-Path $DataRoot "noise"
$rir = Join-Path $DataRoot "rir"

Ensure-Dir $speech
Ensure-Dir $noise
Ensure-Dir $rir

Write-Host "=== BSS-Audio Tier-Min Extraction ==="

# --- LibriSpeech ---
$libriDest = Join-Path $speech "LibriSpeech"
$libri = Get-BestLibriArchive $dl
if ($null -eq $libri) {
    Write-Error "No complete LibriSpeech archive found (need >= $LIBRI_EXPECTED bytes). Run .\scripts\resume_downloads.ps1"
}
if (-not (Test-ArchiveIntegrity $libri.Path)) {
    Write-Error "LibriSpeech archive is corrupt: $($libri.Path). Delete and re-download via .\scripts\resume_downloads.ps1"
}
if (Should-Extract $libriDest 28000 $Force) {
    Write-Host "Extracting LibriSpeech from $($libri.Path) ..."
    if (Test-Path $libriDest) { Remove-Item $libriDest -Recurse -Force }
    Ensure-Dir $libriDest
    tar -xzf $libri.Path -C $libriDest
    if ($LASTEXITCODE -ne 0) { throw "LibriSpeech tar extraction failed" }
    $flacs = (Get-ChildItem $libriDest -Recurse -Filter *.flac | Measure-Object).Count
    Write-Host "LibriSpeech: $flacs FLAC files"
} else {
    Write-Host "SKIP LibriSpeech (already extracted)"
}

# --- WHAM noise (use tar, NOT Expand-Archive — large zip corrupts in PS) ---
$whamZip = Join-Path $dl "wham_noise.zip"
$whamDest = Join-Path $noise "wham_noise"
if (-not (Test-Path $whamZip) -or (Get-Item $whamZip).Length -lt $WHAM_EXPECTED) {
    Write-Error "WHAM archive incomplete. Run .\scripts\resume_downloads.ps1"
}
if (-not (Test-ArchiveIntegrity $whamZip)) {
    Write-Error "WHAM archive is corrupt: $whamZip. Delete and re-download via .\scripts\resume_downloads.ps1"
}
if (Should-Extract $whamDest 1000 $Force) {
    Write-Host "Extracting WHAM noise (tar -xf) ..."
    if (Test-Path $whamDest) { Remove-Item $whamDest -Recurse -Force }
    Ensure-Dir $noise
    # Zip contains top-level wham_noise/ folder
    tar -xf $whamZip -C $noise
    if ($LASTEXITCODE -ne 0) { throw "WHAM tar extraction failed" }
    $wavs = (Get-ChildItem $whamDest -Recurse -Filter *.wav -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "WHAM: $wavs WAV files"
} else {
    Write-Host "SKIP WHAM (already extracted)"
}

# --- ESC-50 ---
$escZip = Join-Path $dl "ESC-50.zip"
$escDest = Join-Path $noise "ESC-50"
if (Test-Path $escZip) {
    if (Should-Extract $escDest 500 $Force) {
        Write-Host "Extracting ESC-50 ..."
        if (Test-Path $escDest) { Remove-Item $escDest -Recurse -Force }
        Ensure-Dir $escDest
        Expand-Archive -Path $escZip -DestinationPath $escDest -Force
        $audio = (Get-ChildItem $escDest -Recurse -Filter *.wav | Measure-Object).Count
        Write-Host "ESC-50: $audio WAV files"
    } else {
        Write-Host "SKIP ESC-50 (already extracted)"
    }
}

# --- MIT IR Survey ---
$mitZip = Join-Path $dl "mit_ir_audio.zip"
$mitDest = Join-Path $rir "mit_ir_survey"
if (Test-Path $mitZip) {
    if (Should-Extract (Join-Path $mitDest "Audio") 200 $Force) {
        Write-Host "Extracting MIT IR Survey ..."
        if (Test-Path $mitDest) { Remove-Item $mitDest -Recurse -Force }
        Ensure-Dir $mitDest
        Expand-Archive -Path $mitZip -DestinationPath $mitDest -Force
        $irs = (Get-ChildItem $mitDest -Recurse -Filter *.wav | Measure-Object).Count
        Write-Host "MIT IR: $irs WAV files"
    } else {
        Write-Host "SKIP MIT IR (already extracted)"
    }
}

Write-Host ""
Write-Host "=== Extraction complete ==="
Write-Host ('Set env: $env:BSS_DATA_ROOT = "' + $DataRoot + '"')
