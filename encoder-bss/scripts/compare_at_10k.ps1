# Compare v1@120k, v2@10k (when available), and recommend finetune path.
param(
    [string]$DataRoot = "D:\bss-audio-data",
    [string]$EvalRoot = "D:\bss-audio-data\eval\acsim_tierb"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe"
$env:BSS_DATA_ROOT = $DataRoot

$v1 = Join-Path $Root "checkpoints\sepformer_s_tierb\best.pt"
$v2_10k = Join-Path $Root "checkpoints\sepformer_s_tierb_v2\step_10000.pt"

$args = @(
    "-u", "scripts\compare_checkpoints.py",
    "--config", "configs\sepformer_s_tierb.yaml",
    "--eval-root", $EvalRoot,
    "--max-clips", "50",
    "v1_run1_120k=$v1"
)

if (Test-Path $v2_10k) {
    $args += "v2_scratch_10k=$v2_10k"
} else {
    Write-Host "v2 @ 10k not ready yet: $v2_10k"
}

Push-Location $Root
& $Python @args
Pop-Location

Write-Host ""
Write-Host "If v2@10k < v1@120k, start finetune:"
Write-Host "  .\scripts\start_train.ps1 -Config configs\sepformer_s_tierb_v4_finetune.yaml"
Write-Host "  (hybrid alt: configs\sepformer_s_tierb_v4b_finetune.yaml)"
