"""Training loop for AC-SIM-ML separation."""

from __future__ import annotations

import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bss_audio.config import Config
from bss_audio.data.acsim.factory import build_train_dataset, dataloader_num_workers
from bss_audio.eval.collapse import summarize_collapse
from bss_audio.eval.harness import evaluate, load_eval_clips, primary_score, results_to_primary_metrics, results_to_wandb_metrics
from bss_audio.logging.audio_demos import collect_eval_audio_demos, demos_to_wandb_payload
from bss_audio.logging.wandb_logger import (
    finish,
    init_wandb,
    log_checkpoint_artifact,
    log_eval_bundle,
    log_metrics,
)
from bss_audio.losses.acsim_ml import (
    PITSISNRDistillLoss,
    align_embedding_time,
    build_loss_fn,
)
from bss_audio.losses.metrics import (
    mixture_baseline_si_sdr_db,
    pit_mean_si_sdr_db,
    si_sdri,
)
from bss_audio.models.conv_tasnet_embedding import (
    CleanConvTasNet,
    NoisyConvTasNetStudent,
)
from bss_audio.models.dprnn import DPRNNSeparator, DPRNNTasNet
from bss_audio.models.sepformer_s import SepformerSSeparator


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: Config) -> nn.Module:
    m = cfg.get("model", default={})
    name = str(m.get("name", "dprnn_s")).lower()
    n_src = int(m.get("n_src", 2))

    if name in ("convtasnet_clean", "conv_tasnet_clean", "clean_convtasnet"):
        return CleanConvTasNet(
            n_src=n_src,
            encoder_dim=int(m.get("encoder_dim", 512)),
            kernel_size=int(m.get("kernel_size", 16)),
            stride=int(m.get("stride", 8)),
            bottleneck_dim=int(m.get("bottleneck_dim", 128)),
            hidden_dim=int(m.get("hidden_dim", 512)),
            n_blocks=int(m.get("n_blocks", 8)),
            n_repeats=int(m.get("n_repeats", 3)),
            mask_activation=str(m.get("mask_activation", "sigmoid")),
        )

    if name in ("convtasnet_noisy", "conv_tasnet_noisy", "noisy_convtasnet"):
        return NoisyConvTasNetStudent(
            n_src=n_src,
            encoder_dim=int(m.get("encoder_dim", 512)),
            kernel_size=int(m.get("kernel_size", 16)),
            stride=int(m.get("stride", 8)),
            bottleneck_dim=int(m.get("bottleneck_dim", 128)),
            hidden_dim=int(m.get("hidden_dim", 512)),
            n_blocks=int(m.get("n_blocks", 8)),
            n_repeats=int(m.get("n_repeats", 3)),
            mask_activation=str(m.get("mask_activation", "sigmoid")),
            student_hidden_dim=int(m.get("student_hidden_dim", 192)),
            student_encoder_blocks=int(m.get("student_encoder_blocks", 2)),
        )

    if name in ("sepformer_s", "sepformer-s", "sepformer"):
        return SepformerSSeparator(
            n_src=n_src,
            encoder_dim=int(m.get("encoder_dim", 128)),
            n_heads=int(m.get("n_heads", 4)),
            d_ffn=int(m.get("d_ffn", 512)),
            n_blocks=int(m.get("n_blocks", 6)),
            n_stages=int(m.get("n_stages", 2)),
            kernel_size=int(m.get("kernel_size", 16)),
            stride=int(m.get("stride", 8)),
            chunk_size=int(m.get("chunk_size", 250)),
            dropout=float(m.get("dropout", 0.0)),
            gradient_checkpointing=bool(m.get("gradient_checkpointing", True)),
        )

    if name in ("dprnn_tasnet", "dprnn_m", "dprnn-m"):
        chunk_size = int(m.get("chunk_size", 250))
        hop = m.get("hop_size")
        return DPRNNTasNet(
            n_src=n_src,
            encoder_dim=int(m.get("encoder_dim", 256)),
            bottleneck_dim=int(m.get("bottleneck_dim", 128)),
            hidden_dim=int(m.get("hidden_dim", 128)),
            kernel_size=int(m.get("kernel_size", 16)),
            stride=int(m.get("stride", 8)),
            chunk_size=chunk_size,
            hop_size=int(hop) if hop is not None else None,
            n_layers=int(m.get("n_layers", 6)),
            mask_activation=str(m.get("mask_activation", "softmax")),
        )

    # dprnn_s — legacy scaffold (checkpoint compatibility).
    return DPRNNSeparator(
        n_src=n_src,
        encoder_dim=int(m.get("encoder_dim", 128)),
        hidden_dim=int(m.get("hidden_dim", 128)),
        chunk_size=int(m.get("chunk_size", 250)),
        n_layers=int(m.get("n_layers", 6)),
    )


class _RatePlateauState:
    """Rate-of-gain (derivative) LR scheduler on D-All eval.

    Halves LR when the per-step improvement rate stays below ``rate_floor`` or
    drops by more than ``rate_ratio`` vs the previous eval interval, for
    ``patience`` consecutive evals. Resets the streak on a clear uptick
    (positive gain only). Optionally treats regression below ``best_score``
    by more than ``regression_delta`` as slowing.
    """

    def __init__(self, cfg: Config) -> None:
        sched = cfg.get("scheduler", default={}) or {}
        self.patience = int(sched.get("patience", 2))
        self.factor = float(sched.get("factor", 0.5))
        self.rate_floor = float(sched.get("rate_floor", 2.0e-7))
        self.rate_ratio = float(sched.get("rate_ratio", 0.4))
        self.min_lr = float(sched.get("min_lr", 1.0e-5))
        self.improve_ratio = float(sched.get("improve_ratio", 1.1))
        regression_delta = sched.get("regression_delta")
        self.regression_delta = (
            float(regression_delta) if regression_delta is not None else None
        )
        self.prev_eval_step: int | None = None
        self.prev_eval_score: float | None = None
        self.prev_rate: float | None = None
        self.last_gain: float | None = None
        self.plateau_streak = 0

    def update(
        self,
        step: int,
        score: float,
        plateau_lr: float,
        *,
        best_score: float | None = None,
    ) -> float:
        if self.prev_eval_step is not None:
            gain = score - self.prev_eval_score  # type: ignore[operator]
            dstep = step - self.prev_eval_step
            rate = gain / max(1, dstep)
            self.last_gain = gain

            slowing = rate < self.rate_floor
            if self.prev_rate is not None:
                slowing = slowing or (rate < self.prev_rate * self.rate_ratio)
            if (
                self.regression_delta is not None
                and best_score is not None
                and score < best_score - self.regression_delta
            ):
                slowing = True

            if (
                gain > 0
                and self.prev_rate is not None
                and rate >= self.prev_rate * self.improve_ratio
            ):
                self.plateau_streak = 0
            elif slowing:
                self.plateau_streak += 1
            else:
                self.plateau_streak = 0

            if self.plateau_streak >= self.patience:
                plateau_lr = max(plateau_lr * self.factor, self.min_lr)
                self.plateau_streak = 0

            self.prev_rate = rate
        else:
            self.prev_rate = None
            self.last_gain = None

        self.prev_eval_step = step
        self.prev_eval_score = score
        return plateau_lr


def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    o = cfg.get("optimizer", default={})
    opt_name = str(o.get("name", "adamw")).lower()
    lr = float(o.get("lr", 1.5e-4))
    betas = tuple(o.get("betas", [0.9, 0.99]))
    weight_decay = float(o.get("weight_decay", 0.0))

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)


def lr_schedule(step: int, cfg: Config, *, plateau_lr: float | None = None) -> float:
    sched = cfg.get("scheduler", default={})
    mode = str(sched.get("mode", "cosine")).lower()
    o = cfg.get("optimizer", default={})
    base_lr = float(plateau_lr if plateau_lr is not None else o.get("lr", 1.5e-4))
    warmup = int(o.get("warmup_steps", 10000))
    max_steps = int(cfg.get("training", "max_steps", default=80000))

    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)

    if mode in ("plateau", "rate_plateau"):
        return base_lr

    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


class _EarlyStopState:
    def __init__(self, cfg: Config) -> None:
        es = cfg.get("training", "early_stop", default={}) or {}
        self.enabled = bool(es.get("enabled", False))
        self.patience = int(es.get("patience", 2))
        self.min_delta = float(es.get("min_delta", 0.3))
        self.stop_on_regression = bool(es.get("stop_on_regression", True))
        self.regression_delta = float(es.get("regression_delta", 1.0))
        self.min_steps = int(es.get("min_steps", 50000))
        self.no_improve = 0
        self.should_stop = False

    def update(self, step: int, score: float, best_score: float, is_new_best: bool = False) -> None:
        if not self.enabled or step < self.min_steps:
            return
        if is_new_best or score >= best_score - 1e-6:
            self.no_improve = 0
            return
        if self.stop_on_regression and score < best_score - self.regression_delta:
            self.no_improve += 1
        elif score <= best_score + self.min_delta:
            self.no_improve += 1
        if self.no_improve >= self.patience:
            self.should_stop = True


def load_init_weights(model: nn.Module, path: Path, *, strict: bool = True) -> dict:
    """Load model weights only (finetune / warm-start). Does not touch optimizer or step."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"Init checkpoint missing keys: {missing}")
    if unexpected:
        print(f"Init checkpoint unexpected keys: {unexpected}")
    meta = {
        "path": str(path),
        "source_step": int(ckpt.get("step", -1)),
        "source_best_eval_score": float(ckpt.get("best_eval_score", float("nan"))),
    }
    print(
        f"Loaded init weights from {path} "
        f"(source step {meta['source_step']}, "
        f"source best D-All {meta['source_best_eval_score']:.3f} dB)"
    )
    return meta


def build_clean_teacher(cfg: Config) -> CleanConvTasNet:
    """Construct a clean Conv-TasNet teacher (defaults to ``teacher_model`` block)."""
    m = cfg.get("teacher_model", default=None) or cfg.get("model", default={})
    return CleanConvTasNet(
        n_src=int(m.get("n_src", 2)),
        encoder_dim=int(m.get("encoder_dim", 512)),
        kernel_size=int(m.get("kernel_size", 16)),
        stride=int(m.get("stride", 8)),
        bottleneck_dim=int(m.get("bottleneck_dim", 128)),
        hidden_dim=int(m.get("hidden_dim", 512)),
        n_blocks=int(m.get("n_blocks", 8)),
        n_repeats=int(m.get("n_repeats", 3)),
        mask_activation=str(m.get("mask_activation", "sigmoid")),
    )


def load_student_masker_decoder_from_teacher(
    student: nn.Module,
    teacher_checkpoint: Path,
) -> list[str]:
    """Copy masker/decoder weights from a clean teacher checkpoint (skip encoder)."""
    ckpt = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    student_sd = student.state_dict()
    loaded: list[str] = []
    for key, value in state.items():
        if key.startswith("encoder."):
            continue
        if key in student_sd and student_sd[key].shape == value.shape:
            student_sd[key] = value
            loaded.append(key)
    student.load_state_dict(student_sd, strict=False)
    return loaded


def setup_frozen_teacher(cfg: Config, device: torch.device) -> CleanConvTasNet:
    teacher_ckpt = cfg.get("training", "teacher_checkpoint", default=None)
    if not teacher_ckpt:
        raise ValueError("training.teacher_checkpoint is required for pit_sisnr_distill")
    path = Path(str(teacher_ckpt))
    if not path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    teacher = build_clean_teacher(cfg).to(device)
    load_init_weights(teacher, path, strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    step: int,
    cfg: Config,
    *,
    best_eval_score: float = float("-inf"),
    eval_metrics: dict | None = None,
    last_rest_time: float | None = None,
    wandb_run_id: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.raw,
            "best_eval_score": best_eval_score,
            "eval_metrics": eval_metrics or {},
            "last_rest_time": last_rest_time,
            "wandb_run_id": wandb_run_id,
        },
        path,
    )


def _maybe_gpu_rest(
    cfg: Config,
    device: torch.device,
    last_rest_time: float,
    run,
    step: int,
) -> float:
    if not cfg.get("training", "gpu_rest_enabled", default=False):
        return last_rest_time
    if device.type != "cuda":
        return last_rest_time
    interval = float(cfg.get("training", "gpu_rest_interval_sec", default=7200))
    duration = float(cfg.get("training", "gpu_rest_duration_sec", default=900))
    now = time.time()
    if now - last_rest_time < interval:
        return last_rest_time
    torch.cuda.synchronize()
    elapsed = now - last_rest_time
    print(f"\n[GPU rest] Active for {elapsed / 60:.1f} min — pausing {duration / 60:.1f} min")
    log_metrics({"system/gpu_rest_elapsed_min": elapsed / 60}, step, run)
    time.sleep(duration)
    log_metrics({"system/gpu_rest_end": 1}, step, run)
    return time.time()


def train(
    cfg: Config,
    resume: Path | None = None,
    init_checkpoint: Path | None = None,
) -> None:
    set_seed(int(cfg.get("training", "seed", default=42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg.get("training", "precision", default="bf16") == "bf16" and device.type == "cuda"

    model = build_model(cfg).to(device)
    loss_fn = build_loss_fn(cfg)
    use_distill = isinstance(loss_fn, PITSISNRDistillLoss)
    teacher: CleanConvTasNet | None = None
    if use_distill:
        teacher = setup_frozen_teacher(cfg, device)
    optimizer = build_optimizer(model, cfg)

    dataset = build_train_dataset(cfg)
    num_workers = dataloader_num_workers(cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("training", "batch_size", default=1)),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    wb = cfg.get("wandb", default={})
    wandb_run_id: str | None = None
    ckpt = None
    if resume and resume.exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        wandb_run_id = ckpt.get("wandb_run_id")

    run = init_wandb(
        project=wb.get("project", "bss-audio-acsim"),
        name=cfg.get("run_name"),
        config=cfg.raw,
        entity=wb.get("entity"),
        tags=wb.get("tags"),
        enabled=bool(wb.get("enabled", True)),
        run_id=wandb_run_id,
        offline=bool(wb.get("offline", False)),
    )
    if run is not None and wandb_run_id is None:
        wandb_run_id = run.id
        if getattr(run, "url", None):
            print(f"WandB run: {run.url}")

    start_step = 0
    best_eval_score = float("-inf")
    last_eval_metrics: dict = {}
    last_rest_time = time.time()
    plateau_lr: float | None = None
    sched_cfg = cfg.get("scheduler", default={}) or {}
    sched_mode = str(sched_cfg.get("mode", "cosine")).lower()
    if sched_mode in ("plateau", "rate_plateau"):
        plateau_lr = float(cfg.get("optimizer", "lr", default=1.5e-4))
    plateau_patience = int(sched_cfg.get("patience", 3))
    plateau_min_lr = float(sched_cfg.get("min_lr", 1.0e-6))
    plateau_no_improve = 0
    rate_plateau = _RatePlateauState(cfg) if sched_mode == "rate_plateau" else None
    early_stop = _EarlyStopState(cfg)

    init_meta: dict | None = None
    if init_checkpoint is None:
        cfg_init = cfg.get("training", "init_checkpoint", default=None)
        if cfg_init:
            init_checkpoint = Path(str(cfg_init))
    model_name = str(cfg.get("model", "name", default="")).lower()
    init_masker_decoder = bool(
        cfg.get("training", "init_masker_decoder_from_teacher", default=True)
    )
    if init_checkpoint and init_checkpoint.exists() and ckpt is None:
        if use_distill and init_masker_decoder:
            loaded = load_student_masker_decoder_from_teacher(model, init_checkpoint)
            init_meta = {"path": str(init_checkpoint), "masker_decoder_keys": len(loaded)}
            print(
                f"Initialized student masker/decoder from {init_checkpoint} "
                f"({len(loaded)} tensors)"
            )
        elif "noisy" in model_name or "convtasnet" in model_name:
            if init_masker_decoder:
                loaded = load_student_masker_decoder_from_teacher(model, init_checkpoint)
                init_meta = {"path": str(init_checkpoint), "masker_decoder_keys": len(loaded)}
                print(
                    f"Partial init (masker/decoder) from {init_checkpoint} "
                    f"({len(loaded)} tensors)"
                )
            else:
                init_meta = load_init_weights(model, init_checkpoint, strict=False)
        else:
            init_meta = load_init_weights(model, init_checkpoint)
        if run is not None and init_meta:
            log_metrics(
                {
                    "init/source_step": init_meta.get("source_step", -1),
                    "init/source_best_d_all": init_meta.get(
                        "source_best_eval_score", float("nan")
                    ),
                },
                0,
                run,
            )
    elif use_distill and init_masker_decoder:
        teacher_ckpt = cfg.get("training", "teacher_checkpoint", default=None)
        if teacher_ckpt:
            path = Path(str(teacher_ckpt))
            if path.exists():
                loaded = load_student_masker_decoder_from_teacher(model, path)
                print(
                    f"Initialized student masker/decoder from teacher {path} "
                    f"({len(loaded)} tensors)"
                )

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0)) + 1
        best_eval_score = float(ckpt.get("best_eval_score", float("-inf")))
        last_eval_metrics = ckpt.get("eval_metrics", {})
        # Reset rest timer on resume — stale last_rest_time triggers immediate 15 min sleep.
        last_rest_time = time.time()
        print(f"Resumed from {resume} at step {start_step}, best_eval_score={best_eval_score:.3f}")

    max_steps = int(cfg.get("training", "max_steps", default=80000))
    grad_accum = int(cfg.get("training", "grad_accum_steps", default=8))
    log_every = int(cfg.get("training", "log_every", default=50))
    eval_every = int(cfg.get("training", "eval_every", default=2000))
    save_every = int(cfg.get("training", "save_every", default=5000))
    grad_clip = float(cfg.get("training", "grad_clip", default=5.0))
    eval_root = Path(cfg.get("eval", "root", default=str(cfg.data_root / "eval" / "acsim_fast")))
    eval_max_clips = cfg.get("eval", "max_clips_per_split", default=None)
    if eval_max_clips is not None:
        eval_max_clips = int(eval_max_clips)
    eval_metric_profile = str(cfg.get("eval", "metric_profile", default="train"))
    wb_audio = cfg.get("wandb", "audio", default={}) or {}
    log_audio_enabled = bool(wb_audio.get("enabled", True))
    audio_split = str(wb_audio.get("split", "D-All"))
    audio_max_clips = int(wb_audio.get("max_clips", 3))
    audio_clip_indices = wb_audio.get("clip_indices")
    if audio_clip_indices is not None:
        audio_clip_indices = [int(i) for i in audio_clip_indices]

    ckpt_dir = Path("checkpoints") / cfg.get("run_name", default="run")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    collapse_gate_step = int(cfg.get("training", "collapse_gate_step", default=50000))
    collapse_gate_pct = float(cfg.get("training", "collapse_gate_pct", default=30.0))
    collapse_eval_clips = int(cfg.get("training", "collapse_eval_clips", default=20))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    use_tqdm = sys.stdout.isatty()
    pbar = tqdm(
        range(start_step, max_steps),
        initial=start_step,
        total=max_steps,
        disable=not use_tqdm,
    )
    step_start = time.time()
    last_grad_norm = 0.0
    last_grad_norm_clipped = 0.0
    last_step = start_step
    loss_window: dict[str, float] = {}
    loss_window_n = 0
    step_time_acc = 0.0
    step_time_n = 0
    noisy_input_window = 0.0
    noisy_input_window_n = 0
    train_wall_start = time.time()
    prev_eval_d_all: float | None = None
    first_eval_d_all: float | None = None
    last_inference_ms: float | None = None

    for step in pbar:
        last_step = step
        for g in optimizer.param_groups:
            g["lr"] = lr_schedule(step, cfg, plateau_lr=plateau_lr)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)

        used_noisy = batch.get("used_noisy_input")
        if used_noisy is not None:
            if isinstance(used_noisy, torch.Tensor):
                noisy_input_window += float(used_noisy.float().mean().item())
            else:
                noisy_input_window += float(used_noisy)
            noisy_input_window_n += 1

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            if use_distill:
                if "clean_mixture" not in batch:
                    raise KeyError(
                        "Distillation requires clean_mixture in batch; "
                        "set data.require_paired=true and use paired cache."
                    )
                clean_mixture = batch["clean_mixture"].to(device)
                with torch.no_grad():
                    teacher_z = teacher.encode(clean_mixture)
                estimate, student_z = model(mixture, return_embedding=True)
                loss, parts = loss_fn(
                    estimate.float(),
                    target.float(),
                    mixture.float(),
                    student_embedding=student_z.float(),
                    teacher_embedding=teacher_z.float(),
                )
            else:
                estimate = model(mixture)
                try:
                    loss, parts = loss_fn(estimate.float(), target.float(), mixture.float())
                except TypeError:
                    loss, parts = loss_fn(estimate.float(), target.float())
            loss = loss / grad_accum

        loss.backward()

        if (step + 1) % grad_accum == 0:
            last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
            last_grad_norm_clipped = min(last_grad_norm, grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            last_rest_time = _maybe_gpu_rest(cfg, device, last_rest_time, run, step)
            for key, val in parts.items():
                loss_window[key] = loss_window.get(key, 0.0) + float(val)
            loss_window_n += 1

        step_time = time.time() - step_start
        step_start = time.time()
        step_time_acc += step_time
        step_time_n += 1

        if step % log_every == 0:
            avg_step_time = step_time_acc / step_time_n if step_time_n else step_time
            step_time_acc = 0.0
            step_time_n = 0
            # Window-averaged loss (SepReformer-style epoch mean over log_every optimizer steps).
            elapsed_h = (time.time() - train_wall_start) / 3600.0
            metrics = {
                "train/step": step,
                "training/global_step": step,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": last_grad_norm,
                "train/grad_norm_clipped": last_grad_norm_clipped,
                "train/sec_per_step": avg_step_time,
                "train/elapsed_hours": elapsed_h,
                "train/steps_per_hour": step / max(elapsed_h, 1e-9),
            }
            with torch.no_grad():
                est_f = estimate.float()
                metrics["train/batch_si_sdri_dB"] = si_sdri(est_f, target, mixture)
                metrics["train/batch_si_sdr_est_dB"] = pit_mean_si_sdr_db(est_f, target)
                metrics["train/batch_si_sdr_mixture_dB"] = mixture_baseline_si_sdr_db(
                    target, mixture
                )
                metrics["train/batch_si_sdri_gain_dB"] = (
                    metrics["train/batch_si_sdr_est_dB"]
                    - metrics["train/batch_si_sdr_mixture_dB"]
                )
            if loss_window_n > 0:
                for key, total in loss_window.items():
                    short = key.replace("loss/", "train/loss_")
                    metrics[f"{short}_avg"] = total / loss_window_n
                loss_window = {}
                loss_window_n = 0
            if noisy_input_window_n > 0:
                metrics["train/noisy_input_fraction"] = (
                    noisy_input_window / noisy_input_window_n
                )
                noisy_input_window = 0.0
                noisy_input_window_n = 0
            if use_distill and teacher is not None and step % log_every == 0:
                with torch.no_grad():
                    clean_mixture = batch.get("clean_mixture")
                    if clean_mixture is not None:
                        clean_mixture = clean_mixture.to(device)
                        t_z = teacher.encode(clean_mixture)
                        _, s_z = model(mixture, return_embedding=True)
                        t_z, s_z = align_embedding_time(s_z.float(), t_z.float())
                        var_s = s_z.var(dim=1).mean().item()
                        var_t = t_z.var(dim=1).mean().item()
                        near_zero = (s_z.abs().mean(dim=1) < 1e-4).float().mean().item()
                        metrics["train/emb_var_student"] = var_s
                        metrics["train/emb_var_teacher"] = var_t
                        metrics["train/emb_near_zero_frac"] = near_zero
            if device.type == "cuda":
                metrics["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                metrics["gpu/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            log_metrics(metrics, step, run)
            log_line = (
                f"step {step}/{max_steps} lr={metrics['train/lr']:.2e} "
                f"gnorm={last_grad_norm:.2f} ({avg_step_time:.2f}s/step)"
            )
            if use_tqdm:
                pbar.set_postfix(
                    lr=f"{metrics['train/lr']:.2e}",
                    gnorm=f"{last_grad_norm:.2f}",
                    stp=f"{avg_step_time:.2f}s",
                )
                pbar.write(log_line)
            else:
                print(log_line, flush=True)

        if step > 0 and step % eval_every == 0 and eval_root.exists():
            eval_start = time.time()
            eval_results = evaluate(
                model,
                eval_root,
                device,
                sample_rate=cfg.sample_rate,
                max_clips_per_split=eval_max_clips,
                metric_profile=eval_metric_profile,
            )
            eval_elapsed_min = (time.time() - eval_start) / 60
            eval_elapsed_sec = time.time() - eval_start
            score = primary_score(eval_results)
            if first_eval_d_all is None and score > float("-inf"):
                first_eval_d_all = score
            d_all_metrics = eval_results.get("D-All", {})
            if "inference_ms_per_clip" in d_all_metrics:
                last_inference_ms = float(d_all_metrics["inference_ms_per_clip"])
            eval_wandb: dict[str, float] = {
                **results_to_wandb_metrics(eval_results),
                **results_to_primary_metrics(eval_results),
                "eval/elapsed_min": eval_elapsed_min,
                "eval/elapsed_sec": eval_elapsed_sec,
                "train/elapsed_hours": (time.time() - train_wall_start) / 3600.0,
                "checkpoint/best_score": max(best_eval_score, score),
            }
            if prev_eval_d_all is not None:
                eval_wandb["eval/D-All_improvement_since_last_eval_dB"] = score - prev_eval_d_all
            if first_eval_d_all is not None:
                eval_wandb["eval/D-All_improvement_since_first_eval_dB"] = (
                    score - first_eval_d_all
                )
            prev_eval_d_all = score
            d_all_dir = eval_root / "D-All"
            if d_all_dir.exists():
                collapse_clips = load_eval_clips(d_all_dir)
                collapse_stats = summarize_collapse(
                    model, collapse_clips, device, max_clips=collapse_eval_clips
                )
                eval_wandb.update(collapse_stats)
                pbar.write(
                    f"  Collapse: {collapse_stats.get('eval/collapse_pct', 0):.0f}% "
                    f"({int(collapse_stats.get('eval/collapse_count', 0))}/"
                    f"{collapse_eval_clips}), "
                    f"energy_ratio={collapse_stats.get('eval/energy_ratio', 0):.1f}x"
                )
                if (
                    step >= collapse_gate_step
                    and collapse_stats.get("eval/collapse_pct", 0) > collapse_gate_pct
                ):
                    pbar.write(
                        f"Collapse gate FAILED at step {step}: "
                        f"{collapse_stats['eval/collapse_pct']:.0f}% > {collapse_gate_pct:.0f}%"
                    )
                    early_stop.should_stop = True
            is_new_best = score > best_eval_score
            eval_wandb["checkpoint/is_new_best"] = float(is_new_best)

            audio_payload: dict | None = None
            if log_audio_enabled:
                try:
                    demos = collect_eval_audio_demos(
                        model,
                        eval_root,
                        device,
                        split=audio_split,
                        clip_indices=audio_clip_indices,
                        max_clips=audio_max_clips,
                        sample_rate=cfg.sample_rate,
                    )
                    audio_payload = demos_to_wandb_payload(demos)
                except Exception as exc:
                    pbar.write(f"  WandB audio demo log failed: {exc}")

            log_eval_bundle(
                run,
                training_step=step,
                metrics=eval_wandb,
                audio_payload=audio_payload,
            )
            last_eval_metrics = eval_results
            s_all = eval_results.get("S-All", {}).get("silence_sdri", float("nan"))
            pbar.write(
                f"Eval step {step}: D-All SI-SDRi={score:.2f} dB, "
                f"S-All={s_all:.2f} dB ({eval_elapsed_min:.1f} min)"
            )

            if is_new_best:
                best_eval_score = score
                plateau_no_improve = 0
                best_path = ckpt_dir / "best.pt"
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    step,
                    cfg,
                    best_eval_score=best_eval_score,
                    eval_metrics=eval_results,
                    last_rest_time=last_rest_time,
                    wandb_run_id=wandb_run_id,
                )
                log_checkpoint_artifact(
                    run,
                    best_path,
                    alias="best",
                    metadata={"step": step, "D-All_si_sdri": score},
                )
                pbar.write(f"  New best checkpoint: D-All SI-SDRi={score:.2f} dB")
            else:
                plateau_no_improve += 1
                if (
                    plateau_lr is not None
                    and plateau_no_improve >= plateau_patience
                    and sched_mode == "plateau"
                ):
                    plateau_lr *= float(sched_cfg.get("factor", 0.5))
                    plateau_lr = max(plateau_lr, plateau_min_lr)
                    plateau_no_improve = 0
                    pbar.write(f"  Plateau LR halved -> {plateau_lr:.2e}")

            if rate_plateau is not None and plateau_lr is not None:
                old_lr = plateau_lr
                plateau_lr = rate_plateau.update(
                    step, score, plateau_lr, best_score=best_eval_score
                )
                if plateau_lr < old_lr:
                    pbar.write(f"  Rate plateau LR halved -> {plateau_lr:.2e}")
                if run is not None and rate_plateau.prev_rate is not None:
                    rp_metrics: dict[str, float] = {
                        "scheduler/rate": rate_plateau.prev_rate,
                        "scheduler/plateau_streak": rate_plateau.plateau_streak,
                        "scheduler/lr": plateau_lr,
                    }
                    if rate_plateau.last_gain is not None:
                        rp_metrics["scheduler/d_all_gain"] = rate_plateau.last_gain
                    log_eval_bundle(
                        run,
                        training_step=step,
                        metrics=rp_metrics,
                    )

            early_stop.update(step, score, best_eval_score, is_new_best=is_new_best)
            if early_stop.should_stop:
                pbar.write(
                    f"Early stop at step {step}: D-All={score:.2f} dB, best={best_eval_score:.2f} dB"
                )
                break

            save_checkpoint(
                ckpt_dir / "latest.pt",
                model,
                optimizer,
                step,
                cfg,
                best_eval_score=best_eval_score,
                eval_metrics=eval_results,
                last_rest_time=last_rest_time,
                wandb_run_id=wandb_run_id,
            )

            model.train()

        if early_stop.should_stop:
            break

        if step > 0 and step % save_every == 0:
            save_checkpoint(
                ckpt_dir / f"step_{step}.pt",
                model,
                optimizer,
                step,
                cfg,
                best_eval_score=best_eval_score,
                eval_metrics=last_eval_metrics,
                last_rest_time=last_rest_time,
                wandb_run_id=wandb_run_id,
            )
            save_checkpoint(
                ckpt_dir / "latest.pt",
                model,
                optimizer,
                step,
                cfg,
                best_eval_score=best_eval_score,
                eval_metrics=last_eval_metrics,
                last_rest_time=last_rest_time,
                wandb_run_id=wandb_run_id,
            )

    final_kwargs = dict(
        best_eval_score=best_eval_score,
        eval_metrics=last_eval_metrics,
        last_rest_time=last_rest_time,
        wandb_run_id=wandb_run_id,
    )
    save_checkpoint(ckpt_dir / "final.pt", model, optimizer, last_step, cfg, **final_kwargs)
    save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, last_step, cfg, **final_kwargs)
    best_path = ckpt_dir / "best.pt"
    if not best_path.exists() and (ckpt_dir / "latest.pt").exists():
        shutil.copy2(ckpt_dir / "latest.pt", best_path)
        print(f"No eval best checkpoint; copied latest.pt -> best.pt")
    log_checkpoint_artifact(run, ckpt_dir / "final.pt", alias="final", metadata={"step": last_step})
    train_hours = (time.time() - train_wall_start) / 3600.0
    finish(
        run,
        summary={
            "best_D-All_SI-SDRi_dB": best_eval_score if best_eval_score > float("-inf") else None,
            "final_step": last_step,
            "train_wall_hours": round(train_hours, 4),
            "inference_ms_per_clip": last_inference_ms,
            "wandb_run_id": wandb_run_id,
        },
    )
    best_msg = (
        f"{best_eval_score:.2f} dB"
        if best_eval_score > float("-inf")
        else "n/a (no eval)"
    )
    print(
        f"Training complete. Checkpoints in {ckpt_dir} "
        f"(wall {train_hours:.2f} h, best D-All SI-SDRi {best_msg})"
    )
