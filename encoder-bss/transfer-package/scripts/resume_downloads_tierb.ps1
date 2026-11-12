# Resume Tier B large downloads in a persistent window.
# Usage: .\scripts\resume_downloads_tierb.ps1 [-DataRoot "D:\bss-audio-data"]

param(
    [string]$DataRoot = "D:\bss-audio-data"
)

$ErrorActionPreference = "Stop"
$dl = Join-Path $DataRoot "downloads"

$jobs = @(
    @{
        Out = Join-Path $dl "train-clean-360.tar.gz"
        Url = "https://www.openslr.org/resources/12/train-clean-360.tar.gz"
        Min = 20GB
    },
    @{
        Out = Join-Path $dl "VCTK-Corpus-0.92.zip"
        Url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
        Min = 10GB
    },
    @{
        Out = Join-Path $dl "daps.tar.gz"
        Url = "https://zenodo.org/record/4660670/files/daps.tar.gz?download=1"
        Min = 14GB
    }
)

Write-Host "=== Resume Tier B downloads ==="
Write-Host "Data root: $DataRoot"
Write-Host "Keep this window open until all transfers finish."

foreach ($j in $jobs) {
    $size = if (Test-Path $j.Out) { (Get-Item $j.Out).Length } else { 0 }
    if ($size -ge $j.Min) {
        Write-Host "SKIP (complete): $($j.Out)"
        continue
    }
    Write-Host "Downloading: $($j.Out) ($([math]::Round($size/1GB,2)) GB so far)"
    curl.exe -L -C - --retry 5 --retry-delay 3 -o $j.Out $j.Url
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "curl failed for $($j.Out) - re-run to resume"
    }
}

Write-Host ""
Write-Host "All Tier B downloads finished. Run:"
Write-Host ('  .\scripts\extract_data_tierb.ps1 -DataRoot ' + $DataRoot)
Write-Host ('  .\scripts\verify_downloads_tierb.ps1 -DataRoot ' + $DataRoot)
