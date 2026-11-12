#!/usr/bin/env python3
"""Export local WAV demos with the same per-track levels as WandB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.config import Config
from bss_audio.eval.harness import load_eval_clips
from bss_audio.logging.audio_demos import peak_normalize_playback
from bss_audio.losses.metrics import si_sdri
from bss_audio.train.loop import build_model


def _save(path: Path, wav: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav.astype(np.float32), sr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/convtasnet_noisy_student_pit_only.yaml")
    parser.add_argument("--eval-root", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split", type=str, default="D-All")
    parser.add_argument("--clip-indices", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--output", type=str, default="demos/separation")
    parser.add_argument(
        "--playback-peak",
        type=float,
        default=0.95,
        help="Per-track peak cap (same rule as WandB demos; default 0.95)",
    )
    parser.add_argument("--save-raw", action="store_true", help="Also write unnormalized waveforms under raw/")
    args = parser.parse_args()

    import os

    if args.data_root:
        os.environ["BSS_DATA_ROOT"] = args.data_root

    cfg = Config.from_yaml(ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    eval_root = Path(
        args.eval_root
        or cfg.get(
            "eval",
            "root",
            default=str(Path(os.environ.get("BSS_DATA_ROOT", "E:/bss-audio-data")) / "eval" / "acsim_tierb"),
        )
    )
    split_dir = eval_root / args.split
    clips = load_eval_clips(split_dir)
    sr = int(cfg.sample_rate)
    out_root = Path(args.output) / ckpt_path.parent.name / f"step_{int(ckpt.get('step', 0))}"
    pn = lambda w: peak_normalize_playback(w, peak=args.playback_peak)

    manifest: list[dict] = []
    with torch.no_grad():
        for idx in args.clip_indices:
            if idx >= len(clips):
                continue
            clip = clips[idx]
            mixture = clip["mixture"].unsqueeze(0).to(device)
            target = clip["target"].unsqueeze(0).to(device)
            estimate = model(mixture)

            mix_np = mixture[0].cpu().numpy()
            t0 = target[0, 0].cpu().numpy()
            t1 = target[0, 1].cpu().numpy()
            e0 = estimate[0, 0].cpu().numpy()
            e1 = estimate[0, 1].cpu().numpy()
            speech_only = t0 + t1
            residual = mix_np - speech_only
            si = si_sdri(estimate[0], target[0], mixture[0])

            clip_dir = out_root / f"clip_{idx:02d}"
            play = clip_dir / "playback_wandb_levels"
            _save(play / "mixture_noisy.wav", pn(mix_np), sr)
            _save(play / "mixture_speech_only.wav", pn(speech_only), sr)
            _save(play / "target_speaker1.wav", pn(t0), sr)
            _save(play / "target_speaker2.wav", pn(t1), sr)
            _save(play / "estimate_speaker1.wav", pn(e0), sr)
            _save(play / "estimate_speaker2.wav", pn(e1), sr)
            _save(play / "noise_residual.wav", pn(residual), sr)

            if args.save_raw:
                raw = clip_dir / "raw"
                _save(raw / "mixture_noisy.wav", mix_np, sr)
                _save(raw / "mixture_speech_only.wav", speech_only, sr)
                _save(raw / "target_speaker1.wav", t0, sr)
                _save(raw / "target_speaker2.wav", t1, sr)
                _save(raw / "estimate_speaker1.wav", e0, sr)
                _save(raw / "estimate_speaker2.wav", e1, sr)
                _save(raw / "noise_residual.wav", residual, sr)

            rms_r = float(np.sqrt(np.mean(residual**2)))
            rms_m = float(np.sqrt(np.mean(mix_np**2)))
            manifest.append(
                {
                    "clip_index": idx,
                    "si_sdri_db": float(si),
                    "has_additive_noise": bool(rms_r > 1e-5),
                    "residual_to_mixture_rms_db": float(
                        20 * np.log10(rms_r / max(rms_m, 1e-12))
                    )
                    if rms_r > 1e-10
                    else None,
                }
            )

    readme = out_root / "README.txt"
    readme.write_text(
        "playback_wandb_levels/ — same per-track peak rule as WandB (cap at 0.95, no boost)\n"
        "  mixture_noisy.wav         — eval mixture (speech + static/event noise when present)\n"
        "  mixture_speech_only.wav — target1 + target2 (no additive noise)\n"
        "  target_speaker*.wav       — per-speaker targets (reverb/overlap)\n"
        "  estimate_speaker*.wav     — model outputs\n"
        "  noise_residual.wav        — mixture_noisy - speech_only\n\n"
        "Optional raw/ — unnormalized model outputs (estimates can sound very loud)\n",
        encoding="utf-8",
    )
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote demos to {out_root.resolve()}")
    for row in manifest:
        noise = row.get("residual_to_mixture_rms_db")
        noise_s = f"{noise:.1f} dB" if noise is not None else "none"
        print(f"  clip {row['clip_index']}: SI-SDRi={row['si_sdri_db']:.2f} dB, additive noise ~ {noise_s}")


if __name__ == "__main__":
    main()
