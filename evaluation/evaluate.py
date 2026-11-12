#!/usr/bin/env python3
"""
evaluate.py
===========

Evaluate one or more trained YOLO checkpoints on the validation and/or test
splits, print the headline metrics, and save the results.

Metrics reported
----------------
* Precision (P)
* Recall (R)
* mAP50
* mAP50-95
* Per-class P / R / F1 / AP50 / AP50-95

Examples
--------
Evaluate the newest checkpoint on both val and test::

    python evaluation/evaluate.py

Evaluate a specific checkpoint on the test split only::

    python evaluation/evaluate.py \
        --weights runs/train/yolo11n_640/weights/best.pt --split test

Compare YOLO11n and YOLO11s checkpoints::

    python evaluation/evaluate.py \
        --weights runs/train/yolo11n_640/weights/best.pt \
                  runs/train/yolo11s_640/weights/best.pt

Notes
-----
* Relative paths are resolved from the repository root.
* Results are written to ``evaluation/results/<run-name>/<split>/``.
* A combined ``comparison.csv`` / ``comparison.json`` is written when more
  than one checkpoint is evaluated.
* Class names are read from ``data.yaml`` and are placeholders unless you
  rename them there.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO checkpoints on val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        default=None,
        help=(
            "One or more .pt checkpoint paths. If omitted, the newest "
            "runs/**/weights/best.pt is used."
        ),
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Path to the dataset YAML.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=("val", "test"),
        default=["val", "test"],
        help="Dataset split(s) to evaluate.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Validation image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Validation batch size.",
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
        default=2,
        help="Dataloader worker processes.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Directory for saved metrics and plots.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save confusion matrices and PR/F1 curves.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print Ultralytics' per-class output.",
    )
    return parser


def discover_weights() -> list[Path]:
    """Return the newest ``best.pt`` under ``runs/`` (if any)."""
    runs_dir = PROJECT_ROOT / "runs"
    if not runs_dir.is_dir():
        return []
    candidates = sorted(
        runs_dir.rglob("weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates


def load_names(data_path: Path) -> dict[int, str]:
    """Read class names from the dataset YAML."""
    try:
        import yaml

        config = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - names are display-only
        return {}

    names = config.get("names") or {}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}

    result: dict[int, str] = {}
    for key, value in names.items():
        try:
            result[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return result


def unique_label(weights: Path, used: set[str]) -> str:
    """Build a stable, unique output folder name for a checkpoint."""
    label = weights.parent.parent.name or weights.stem
    candidate = label
    index = 2
    while candidate in used:
        candidate = f"{label}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def count_split_images(data_path: Path, split: str) -> int:
    """Count images in a split without relying on Ultralytics internals."""
    try:
        import yaml

        config = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - the count is informational
        return 0

    root = Path(config.get("path") or ".")
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()

    value = config.get(split)
    if not isinstance(value, str):
        return 0

    target = (root / value).resolve()
    if target.is_dir():
        return sum(
            1
            for path in target.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if target.is_file() and target.suffix.lower() == ".txt":
        return sum(
            1
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return 0


def per_class_rows(metrics, names: dict[int, str]) -> list[dict]:
    """Build per-class metric rows for every class in the dataset."""
    box = metrics.box
    index_to_pos = {
        int(class_id): position
        for position, class_id in enumerate(box.ap_class_index)
    }

    class_ids = set(names)
    class_ids.update(range(len(metrics.nt_per_class)))

    rows: list[dict] = []
    for class_id in sorted(class_ids):
        instances = (
            int(metrics.nt_per_class[class_id])
            if class_id < len(metrics.nt_per_class)
            else 0
        )
        position = index_to_pos.get(class_id)
        if position is None:
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, f"class_{class_id:02d}"),
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "ap50": "",
                    "ap50_95": "",
                    "instances": instances,
                    "evaluated": False,
                }
            )
            continue

        precision = float(box.p[position])
        recall = float(box.r[position])
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        rows.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, f"class_{class_id:02d}"),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "ap50": round(float(box.ap50[position]), 6),
                "ap50_95": round(float(box.ap[position]), 6),
                "instances": instances,
                "evaluated": True,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(
    weights: Path,
    label: str,
    args: argparse.Namespace,
    data_path: Path,
    names: dict[int, str],
    output_root: Path,
) -> list[dict]:
    """Evaluate one checkpoint on every requested split."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    summaries: list[dict] = []

    for split in args.split:
        print(f"\n=== {label} | {split} ===", flush=True)
        metrics = model.val(
            data=str(data_path),
            split=split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(output_root / label),
            name=split,
            exist_ok=True,
            plots=args.plots,
            save_json=False,
            verbose=args.verbose,
        )

        rows = per_class_rows(metrics, names)
        split_dir = output_root / label / split
        split_dir.mkdir(parents=True, exist_ok=True)
        write_csv(split_dir / "per_class.csv", rows)

        summary = {
            "model": label,
            "weights": str(weights),
            "data": str(data_path),
            "split": split,
            "images": count_split_images(data_path, split),
            "instances": int(sum(metrics.nt_per_class)),
            "precision": round(float(metrics.box.mp), 6),
            "recall": round(float(metrics.box.mr), 6),
            "map50": round(float(metrics.box.map50), 6),
            "map50_95": round(float(metrics.box.map), 6),
            "fitness": round(float(metrics.fitness), 6),
            "results_dict": {
                key: (float(value) if hasattr(value, "item") else value)
                for key, value in metrics.results_dict.items()
            },
        }
        (split_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(summary)

        print(
            f"{split}: P={summary['precision']:.4f} "
            f"R={summary['recall']:.4f} "
            f"mAP50={summary['map50']:.4f} "
            f"mAP50-95={summary['map50_95']:.4f}",
            flush=True,
        )

    return summaries


def write_comparison(output_root: Path, summaries: list[dict]) -> None:
    if not summaries:
        return

    fields = [
        "model",
        "split",
        "images",
        "instances",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "fitness",
        "weights",
    ]
    with (output_root / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    (output_root / "comparison.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


def print_comparison(summaries: list[dict]) -> None:
    if not summaries:
        return

    print("\n" + "=" * 88)
    print(
        f"{'model':<26}{'split':<7}{'P':>9}{'R':>9}"
        f"{'mAP50':>9}{'mAP50-95':>11}{'images':>9}"
    )
    print("-" * 88)
    for row in summaries:
        print(
            f"{row['model']:<26}{row['split']:<7}"
            f"{row['precision']:>9.4f}{row['recall']:>9.4f}"
            f"{row['map50']:>9.4f}{row['map50_95']:>11.4f}"
            f"{row['images']:>9}"
        )
    print("=" * 88)


def main() -> int:
    args = build_parser().parse_args()

    # Resolve relative paths from the repository root regardless of the
    # current working directory. This keeps data.yaml portable.
    os.chdir(PROJECT_ROOT)

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        print(f"ERROR: dataset config not found: {data_path}", file=sys.stderr)
        print(
            "Download and extract the dataset first (see README.md).",
            file=sys.stderr,
        )
        return 1

    if args.weights:
        weights_list = [
            Path(value).expanduser().resolve() for value in args.weights
        ]
    else:
        weights_list = discover_weights()

    if not weights_list:
        print("ERROR: no checkpoint found.", file=sys.stderr)
        print(
            "Train a model first, or pass --weights path/to/best.pt.",
            file=sys.stderr,
        )
        return 1

    missing = [path for path in weights_list if not path.is_file()]
    if missing:
        for path in missing:
            print(f"ERROR: checkpoint not found: {path}", file=sys.stderr)
        return 1

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ERROR: ultralytics is not installed.", file=sys.stderr)
        print(
            "Install the project dependencies with: "
            "pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    names = load_names(data_path)
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    used_labels: set[str] = set()

    for weights in weights_list:
        label = unique_label(weights, used_labels)
        summaries.extend(
            evaluate_checkpoint(
                weights=weights,
                label=label,
                args=args,
                data_path=data_path,
                names=names,
                output_root=output_root,
            )
        )

    write_comparison(output_root, summaries)
    print_comparison(summaries)
    print(f"\nResults saved to: {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
