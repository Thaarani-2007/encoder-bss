# Resume large downloads in a persistent PowerShell window.
# Keep this window open until both files finish.
#
# Usage:
#   .\scripts\resume_downloads.ps1 [-DataRoot "D:\bss-audio-data"]

param(
    [string]$DataRoot = "D:\bss-audio-data"
)

$dl = Join-Path $DataRoot "downloads"
New-Item -ItemType Directory -Path $dl -Force | Out-Null

$jobs = @(
    @{
        Out  = Join-Path $dl "wham_noise.zip"
        Url  = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
        MinBytes = 17GB
    },
    @{
        Out  = Join-Path $dl "train-clean-100.part"
        Url  = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
        MinBytes = 6GB
    }
)

foreach ($j in $jobs) {
    $size = if (Test-Path $j.Out) { (Get-Item $j.Out).Length } else { 0 }
    if ($size -ge $j.MinBytes) {
        Write-Host "SKIP (complete): $($j.Out) ($([math]::Round($size/1GB,2)) GB)"
        continue
    }
    Write-Host "Downloading: $($j.Out) ($([math]::Round($size/1GB,2)) GB so far)"
    & curl.exe -L -C - --retry 10 --retry-delay 5 -o $j.Out $j.Url
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed: $($j.Out)" }
    else { Write-Host "Done: $($j.Out)" }
}

Write-Host "All downloads finished. Run .\scripts\extract_data.ps1 to extract."
