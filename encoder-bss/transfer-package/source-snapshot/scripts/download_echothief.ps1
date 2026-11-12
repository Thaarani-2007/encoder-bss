# Optional: download EchoThief RIR subset for AC-SIM (manual link on echothief.com)
# Usage: .\scripts\download_echothief.ps1 [-DataRoot "D:\bss-audio-data"] [-Url "<direct zip url>"]

param(
    [string]$DataRoot = "D:\bss-audio-data",
    [string]$Url = ""
)

$ErrorActionPreference = "Stop"
$dest = Join-Path $DataRoot "rir\echothief"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

if (-not $Url) {
    Write-Host "EchoThief requires a direct download URL from https://www.echothief.com/downloads/"
    Write-Host "Re-run with: .\scripts\download_echothief.ps1 -Url '<zip url>'"
    Write-Host "MIT IR survey RIRs remain active under rir\mit_ir_survey until EchoThief is added."
    exit 0
}

$archive = Join-Path $DataRoot "downloads\echothief.zip"
Write-Host "Downloading EchoThief to $archive ..."
curl.exe -L -C - --retry 5 --retry-delay 3 -o $archive $Url
if ($LASTEXITCODE -ne 0) { throw "EchoThief download failed" }

Write-Host "Extracting to $dest ..."
Expand-Archive -Path $archive -DestinationPath $dest -Force
$wavs = (Get-ChildItem $dest -Recurse -Filter *.wav -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "EchoThief WAV files: $wavs"
