#!/usr/bin/env python3
"""Train AC-SIM-ML separator."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from bss_audio.config import Config
from bss_audio.train.loop import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "dprnn_m_tierb_v3_finetune.yaml"),
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Load model weights only (finetune). Fresh optimizer and step 0.",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Override BSS_DATA_ROOT")
    args = parser.parse_args()

    if args.data_root:
        os.environ["BSS_DATA_ROOT"] = args.data_root

    cfg = Config.from_yaml(args.config)
    resume = Path(args.resume) if args.resume else None
    init_checkpoint = Path(args.init_checkpoint) if args.init_checkpoint else None
    if resume and resume.exists() and args.config is None:
        ckpt = __import__("torch").load(resume, map_location="cpu", weights_only=False)
        if "config" in ckpt:
            cfg = Config(raw=ckpt["config"])
    train(cfg, resume=resume, init_checkpoint=init_checkpoint)


if __name__ == "__main__":
    main()
