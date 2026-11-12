param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid
)

$ErrorActionPreference = "Stop"
Wait-Process -Id $WaitPid
$python = "C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe"
& $python -u "transfer-package\source-snapshot\scripts\verify_paired_cache.py" `
    --cache-dir "E:\bss-audio-data\cache\convtasnet_paired_200k" `
    --expected-clips 100000 `
    --expected-samples 80000 `
    --sample-limit 1000 `
    *> "E:\bss-audio-data\cache\paired_verification.log"
