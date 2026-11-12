"""Weights & Biases integration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb

# Avoid wandb stdout capture deadlocking tqdm when training is piped/backgrounded.
os.environ.setdefault("WANDB_CONSOLE", "off")


def init_wandb(
    *,
    project: str,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    entity: str | None = None,
    tags: list[str] | None = None,
    enabled: bool = True,
    run_id: str | None = None,
    offline: bool = False,
) -> wandb.sdk.wandb_run.Run | None:
    if not enabled:
        return None
    has_key = bool(os.environ.get("WANDB_API_KEY") or wandb.api.api_key)
    if not has_key and not offline and os.environ.get("WANDB_MODE") != "offline":
        print(
            "WARNING: WandB enabled but no API key. Set WANDB_API_KEY, run `wandb login`, "
            "or set wandb.offline: true / WANDB_MODE=offline in config."
        )
        return None
    init_kwargs: dict[str, Any] = {
        "project": project,
        "name": name,
        "config": config,
        "entity": entity,
        "tags": tags,
        "reinit": True,
    }
    if offline or os.environ.get("WANDB_MODE") == "offline":
        init_kwargs["mode"] = "offline"
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "must"
    run = wandb.init(**init_kwargs)
    if run is not None:
        # Train scalars: per log_every (e.g. every 50 steps).
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        # Eval scalars + audio: only at eval_every (e.g. 10k, 20k, …) via eval/step.
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        wandb.define_metric("metric/*", step_metric="eval/step")
        wandb.define_metric("audio/*", step_metric="eval/step")
        wandb.define_metric("checkpoint/*", step_metric="eval/step")
        wandb.define_metric("scheduler/*", step_metric="eval/step")
        wandb.define_metric("system/*", step_metric="train/step")
    return run


def log_metrics(metrics: dict[str, Any], step: int, run: wandb.sdk.wandb_run.Run | None) -> None:
    if run is not None:
        wandb.log(metrics, step=step)
        sec = metrics.get("train/sec_per_step")
        if sec is not None:
            run.summary["sec_per_step"] = round(float(sec), 3)


def log_eval_bundle(
    run: wandb.sdk.wandb_run.Run | None,
    *,
    training_step: int,
    metrics: dict[str, Any],
    audio_payload: dict[str, Any] | None = None,
) -> None:
    """Log eval metrics + optional audio demos on the eval/step axis (10k, 20k, …)."""
    if run is None:
        return
    bundle: dict[str, Any] = {"eval/step": training_step, **metrics}
    if audio_payload:
        bundle.update(audio_payload)
    wandb.log(bundle, step=training_step)


def log_audio_demos(
    payload: dict[str, Any],
    step: int,
    run: wandb.sdk.wandb_run.Run | None,
) -> None:
    """Log wandb.Audio demo dict (from demos_to_wandb_payload)."""
    if run is not None and payload:
        wandb.log(payload, step=step)


def log_checkpoint_artifact(
    run: wandb.sdk.wandb_run.Run | None,
    path: Path,
    *,
    alias: str = "best",
    metadata: dict[str, Any] | None = None,
) -> None:
    if run is None or not path.exists():
        return
    artifact = wandb.Artifact(
        name=f"model-checkpoint-{alias}",
        type="model",
        metadata=metadata or {},
    )
    artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=[alias])


def finish(run: wandb.sdk.wandb_run.Run | None, summary: dict[str, Any] | None = None) -> None:
    if run is not None:
        if summary:
            for key, value in summary.items():
                if value is not None and value == value:
                    run.summary[key] = value
        run.finish()
