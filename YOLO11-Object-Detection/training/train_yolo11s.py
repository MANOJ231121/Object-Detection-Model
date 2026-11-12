#!/usr/bin/env python3
"""
train_yolo11s.py
================

Train Ultralytics YOLO11s on the 35-class dataset used in the original
project.

The training configuration is reused from the original YOLO11n experiment
``full_yolo11n_640_v3`` (see the source project's
``runs/full_yolo11n_640_v3/args.yaml``) and only the model changes. Class IDs
and names come from ``data.yaml``. The original dataset never assigned
semantic class names, so the names are placeholders (``class_00`` ...
``class_34``).

Examples
--------
Train YOLO11s with the original settings on the first GPU::

    python training/train_yolo11s.py

Train on CPU with a smaller batch::

    python training/train_yolo11s.py --device cpu --batch 8

Experiment with another Ultralytics detection model without editing this
file::

    python training/train_yolo11s.py --model yolo11m.pt --name yolo11m_640

Notes
-----
* Pretrained weights are downloaded automatically by Ultralytics on the first
  run. Pass ``--model path/to/weights.pt`` to start from a local checkpoint.
* Relative paths are resolved from the repository root, so the script works
  no matter which directory you launch it from.
* Runs are written to ``runs/train/<name>``. Existing runs are never
  overwritten unless you explicitly pass ``--exist-ok``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths (everything is relative to the repository root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "train"

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
# Reused from the original YOLO11n experiment (full_yolo11n_640_v3).
# Edit these values to create a new baseline, or override the common ones
# from the command line. Comments explain the important parameters.

TRAIN_CONFIG: dict[str, object] = {
    # --- Core schedule -----------------------------------------------------
    "epochs": 60,           # maximum number of passes over the training set
    "imgsz": 640,           # square input resolution used for train + val
    "batch": 16,            # images per batch (lower this if you run out of VRAM)
    "patience": 15,         # early-stop after N epochs without val improvement
    "workers": 2,           # dataloader worker processes (Windows-friendly)
    "cache": False,         # cache decoded images in RAM/disk (needs lots of RAM)
    "fraction": 1.0,        # use 100% of the training split
    # --- Reproducibility ---------------------------------------------------
    "seed": 0,              # fixed seed for Python/NumPy/torch/Ultralytics
    "deterministic": True,  # ask torch to use deterministic algorithms
    # --- Optimiser ---------------------------------------------------------
    "optimizer": "auto",    # let Ultralytics choose AdamW/SGD for the run
    "lr0": 0.01,            # initial learning rate
    "lrf": 0.01,            # final LR = lr0 * lrf
    "momentum": 0.937,      # SGD momentum / Adam beta1
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    # --- Loss weights ------------------------------------------------------
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    # --- Augmentation ------------------------------------------------------
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.4,
    "auto_augment": "randaugment",
    "close_mosaic": 10,     # disable mosaic for the final N epochs
    # --- Runtime / output --------------------------------------------------
    "amp": True,            # automatic mixed precision (GPU speed/memory win)
    "plots": True,          # save training curves and sample batches
    "val": True,            # run validation every epoch
    "save": True,
    "single_cls": False,    # keep all 35 classes separate
    "rect": False,          # rectangular training is off (matches original)
    "cos_lr": False,        # cosine LR schedule is off (matches original)
    "exist_ok": False,      # never overwrite an existing run directory
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train YOLO11s on the project dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help=(
            "Pretrained checkpoint or Ultralytics model name. Any detection "
            "model works, e.g. yolo11n.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt."
        ),
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Path to the dataset YAML.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAIN_CONFIG["epochs"],
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=TRAIN_CONFIG["imgsz"],
        help="Training and validation image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=TRAIN_CONFIG["batch"],
        help="Batch size. Lower it if you run out of GPU memory.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=TRAIN_CONFIG["patience"],
        help="Early-stopping patience in epochs.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help=(
            "Compute device: '0' for the first GPU, '0,1' for two GPUs, "
            "'cpu' for CPU, or 'mps' for Apple Silicon."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=TRAIN_CONFIG["workers"],
        help="Dataloader worker processes. Use 0 if you hit Windows issues.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TRAIN_CONFIG["seed"],
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--project",
        default=str(DEFAULT_PROJECT),
        help="Directory that receives run folders.",
    )
    parser.add_argument(
        "--name",
        default="yolo11s_640",
        help=(
            "Run name. Ultralytics appends a numeric suffix instead of "
            "overwriting an existing run."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume training. Pass the checkpoint via --model, for example "
            "--model runs/train/yolo11s_640/weights/last.pt --resume."
        ),
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images for faster epochs (needs enough RAM/disk).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=TRAIN_CONFIG["fraction"],
        help="Fraction of the training split to use (0.0-1.0).",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing run directory (off by default).",
    )
    return parser


def check_dataset(data_path: Path) -> None:
    """Fail early with a helpful message if the dataset is not in place."""
    if not data_path.is_file():
        raise SystemExit(
            f"ERROR: dataset config not found: {data_path}\n"
            "Download and extract the dataset first (see README.md), then "
            "run this script again."
        )

    try:
        import yaml

        config = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - a warning is enough here
        print(f"WARNING: could not parse {data_path}: {exc}", file=sys.stderr)
        return

    nc = config.get("nc")
    names = config.get("names") or {}
    if nc is not None and len(names) != int(nc):
        raise SystemExit(
            f"ERROR: {data_path.name} declares nc={nc} but defines "
            f"{len(names)} class names."
        )

    root = Path(config.get("path") or ".")
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()

    for split in ("train", "val"):
        value = config.get(split)
        if not isinstance(value, str):
            continue
        split_dir = (root / value).resolve()
        if not split_dir.exists():
            print(
                f"WARNING: {split} split not found: {split_dir}\n"
                "         Ultralytics will fail if this path stays missing.",
                file=sys.stderr,
            )


def main() -> int:
    args = build_parser().parse_args()

    # Resolve relative paths from the repository root regardless of the
    # current working directory. This keeps data.yaml portable.
    os.chdir(PROJECT_ROOT)

    data_path = Path(args.data).expanduser().resolve()
    check_dataset(data_path)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.", file=sys.stderr)
        print(
            "Install the project dependencies with: "
            "pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    project_dir = Path(args.project).expanduser().resolve()

    overrides = dict(TRAIN_CONFIG)
    overrides.update(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        fraction=args.fraction,
        cache=args.cache,
        amp=not args.no_amp,
        resume=args.resume,
        project=str(project_dir),
        name=args.name,
        exist_ok=args.exist_ok,
    )

    print("=" * 72)
    print(f"YOLO training: {args.model}")
    print("=" * 72)
    print(f"Model      : {args.model}")
    print(f"Data       : {data_path}")
    print(f"Epochs     : {args.epochs}")
    print(f"Image size : {args.imgsz}")
    print(f"Batch      : {args.batch}")
    print(f"Patience   : {args.patience}")
    print(f"Device     : {args.device}")
    print(f"Seed       : {args.seed}")
    print(f"Run        : {project_dir / args.name}")
    print("=" * 72)

    model = YOLO(args.model)
    model.train(data=str(data_path), **overrides)

    save_dir = Path(
        getattr(model.trainer, "save_dir", project_dir / args.name)
    )
    best = save_dir / "weights" / "best.pt"

    print()
    print(f"Training finished. Results: {save_dir}")
    if best.is_file():
        print(f"Best weights: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
