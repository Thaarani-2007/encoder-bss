param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid
)

$ErrorActionPreference = "Stop"
$log = "E:\bss-audio-data\cache\paired_cache_continuation.log"
Start-Transcript -Path $log -Append

try {
    Wait-Process -Id $WaitPid

    $python = "C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe"
    $root = "E:\bss-audio-data"
    $manifest = "$root\manifests\speech_split_16k.json"
    $config = "transfer-package\source-snapshot\configs\convtasnet_clean_teacher.yaml"
    $script = "transfer-package\source-snapshot\scripts\generate_paired_cache.py"

    & $python -m py_compile $script
    & $python -u $script `
        --config $config `
        --data-root $root `
        --speech-manifest $manifest `
        --output "$root\cache\bench_paired_20" `
        --num-clips 20 `
        --workers 8 `
        --seed 5100000

    $check = @'
import torch
from pathlib import Path
p = Path(r"E:\bss-audio-data\cache\bench_paired_20")
files = sorted(p.glob("clip_*.pt"))
assert len(files) == 20, len(files)
d = torch.load(files[0], map_location="cpu", weights_only=False)
required = {"clean_mixture", "noisy_mixture", "mixture", "target"}
assert required.issubset(d), sorted(d)
assert torch.equal(d["mixture"], d["noisy_mixture"])
assert d["clean_mixture"].shape == d["noisy_mixture"].shape
assert d["target"].shape[0] == 2
print("paired smoke: OK")
'@
    & $python -c $check

    & $python -u $script `
        --config $config `
        --data-root $root `
        --speech-manifest $manifest `
        --output "$root\cache\convtasnet_paired_200k" `
        --num-clips 200000 `
        --workers 8 `
        --seed 5100000
}
finally {
    Stop-Transcript
}
