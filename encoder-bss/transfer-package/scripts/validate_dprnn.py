#!/usr/bin/env python3

"""Validate DPRNN-S, legacy DPRNN-M, and Asteroid-aligned DPRNNTasNet."""



from __future__ import annotations



import sys

from pathlib import Path



import torch



ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))



from bss_audio.config import Config

from bss_audio.train.loop import build_model



EXPECTED_PARAMS = {

    "dprnn_s": 3_602_176,

    "dprnn_m": 14_347_776,

}





def _validate_config(config_path: Path, *, expected_name: str | None = None) -> int:

    cfg = Config.from_yaml(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    name = str(cfg.get("model", "name", default="dprnn_s")).lower()



    sr = cfg.sample_rate

    t = int(sr * float(cfg.get("data", "clip_seconds", default=4.0)))

    mixture = torch.randn(1, t, device=device)



    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):

        out = model(mixture)



    print(f"\nConfig: {config_path.name}")

    print(f"Model: {name}")

    print(f"Params: {n_params:,} ({n_params / 1e6:.3f}M)")

    print(f"Input:  {tuple(mixture.shape)}")

    print(f"Output: {tuple(out.shape)}")

    assert out.shape == (1, 2, t), f"Unexpected output shape {out.shape}"



    if expected_name:

        assert name == expected_name



    legacy_expected = EXPECTED_PARAMS.get(name)

    if legacy_expected is not None and name == "dprnn_s":
        assert n_params == legacy_expected, f"Expected {legacy_expected} params, got {n_params}"



    if device.type == "cuda":

        vram_gb = torch.cuda.max_memory_allocated() / 1e9

        print(f"VRAM after forward: {vram_gb:.2f} GB")

        if vram_gb > 12.0:

            print("WARNING: VRAM above 12 GB — may be tight on 16 GB GPU")



    print(f"validate_dprnn ({name}): OK")

    return n_params





def main() -> None:

    _validate_config(ROOT / "configs" / "dprnn_s_tierb_sanity.yaml")

    smoke_m = ROOT / "configs" / "dprnn_m_tierb_50k.yaml"

    if smoke_m.exists():

        _validate_config(smoke_m)

    v2 = ROOT / "configs" / "dprnn_m_tierb_v2.yaml"

    if v2.exists():

        n = _validate_config(v2, expected_name="dprnn_tasnet")

        print(f"DPRNNTasNet param count for v2 config: {n:,}")





if __name__ == "__main__":

    main()

