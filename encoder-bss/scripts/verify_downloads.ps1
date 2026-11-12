# Verify download sizes and archive integrity
# Usage: .\scripts\verify_downloads.ps1 [-DataRoot "D:\bss-audio-data"]

param(
    [string]$DataRoot = "D:\bss-audio-data"
)

$dl = Join-Path $DataRoot "downloads"
$expected = @{
    "train-clean-100.part" = 6387309499
    "wham_noise.zip"       = 18194930748
    "ESC-50.zip"           = 100000000
    "mit_ir_audio.zip"     = 10000000
}

Write-Host "=== Download Verification ==="
foreach ($name in $expected.Keys) {
    $path = Join-Path $dl $name
    if (-not (Test-Path $path)) {
        Write-Host "[MISSING] $name"
        continue
    }
    $size = (Get-Item $path).Length
    $exp = $expected[$name]
    $pct = [math]::Round(100 * $size / $exp, 1)
    $status = if ($size -ge $exp) { "SIZE OK" } else { "INCOMPLETE ($pct%)" }
    Write-Host "[$status] $name : $([math]::Round($size/1GB,2)) GB / $([math]::Round($exp/1GB,2)) GB"
    if ($size -ge $exp) {
        if ($name -match '\.part$|\.tar\.gz$') { tar -tzf $path 2>$null | Out-Null }
        elseif ($name -match '\.zip$') { tar -tf $path 2>$null | Out-Null }
        $integrity = if ($LASTEXITCODE -eq 0) { "INTEGRITY OK" } else { "CORRUPT - re-download" }
        Write-Host "         $integrity"
    }
}

Write-Host ""
Write-Host "Extracted file counts:"
$checks = @(
    @{ Path = Join-Path $DataRoot "speech\LibriSpeech"; Filter = "*.flac"; Min = 28000; Label = "LibriSpeech" },
    @{ Path = Join-Path $DataRoot "noise\wham_noise"; Filter = "*.wav"; Min = 1000; Label = "WHAM" },
    @{ Path = Join-Path $DataRoot "noise\ESC-50"; Filter = "*.wav"; Min = 500; Label = "ESC-50" },
    @{ Path = Join-Path $DataRoot "rir\mit_ir_survey"; Filter = "*.wav"; Min = 200; Label = "MIT IR" }
)
foreach ($c in $checks) {
    if (Test-Path $c.Path) {
        $n = (Get-ChildItem $c.Path -Recurse -Filter $c.Filter -ErrorAction SilentlyContinue | Measure-Object).Count
        $ok = if ($n -ge $c.Min) { "OK" } else { "INCOMPLETE" }
        Write-Host "[$ok] $($c.Label): $n files (need >= $($c.Min))"
    } else {
        Write-Host "[MISSING] $($c.Label): not extracted"
    }
}
