#!/usr/bin/env python3
"""
predict.py
==========

Run a trained YOLO checkpoint on an image, a folder of images, a video, a
webcam, a glob, or a stream URL. Annotated results are saved by default.

Examples
--------
Single image (also display the annotated result)::

    python prediction/predict.py \
        --weights runs/train/yolo11n_640/weights/best.pt \
        --source samples/street.jpg --show

Folder of images with a higher confidence threshold::

    python prediction/predict.py \
        --weights runs/train/yolo11n_640/weights/best.pt \
        --source samples/images --conf 0.40

Video::

    python prediction/predict.py \
        --weights runs/train/yolo11n_640/weights/best.pt \
        --source samples/video.mp4 --show

Webcam (camera index 0)::

    python prediction/predict.py \
        --weights runs/train/yolo11n_640/weights/best.pt \
        --source 0

Notes
-----
* Relative paths are resolved from the repository root.
* Results are written to ``prediction/outputs/<name>/``.
* ``--show`` opens a window. It defaults to on for a webcam and off for files,
  so headless machines do not fail.
* Class names come from the checkpoint. They are placeholders unless you pass
  ``--names`` or rename them in ``data.yaml`` before training.
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
DEFAULT_OUTPUT = PROJECT_ROOT / "prediction" / "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a trained YOLO model on images, video, or a webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Path to a trained .pt checkpoint. If omitted, the newest "
            "runs/**/weights/best.pt is used."
        ),
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Input: image file, folder, video file, glob, stream URL, or a "
            "webcam index such as '0' (or 'webcam')."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold; lower values show more boxes.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
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
        "--project",
        default=str(DEFAULT_OUTPUT),
        help="Directory that receives prediction folders.",
    )
    parser.add_argument(
        "--name",
        default="predict",
        help=(
            "Output run name. Ultralytics appends a numeric suffix instead "
            "of overwriting an existing folder."
        ),
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing output folder (off by default).",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save annotated images/video.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Display results in a window. Defaults to on for a webcam and "
            "off for files."
        ),
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save YOLO-format .txt labels next to the results.",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Include confidence values in the saved .txt labels.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Write detections.csv with one row per detected box.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        help="Only keep these class IDs (for example: --classes 0 22).",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum number of detections per image/frame.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding-box line width in saved images.",
    )
    parser.add_argument(
        "--names",
        default=None,
        help=(
            "Optional JSON file mapping class IDs to display names, for "
            "example {\"0\": \"person\", \"22\": \"backpack\"}."
        ),
    )
    return parser


def parse_source(raw: str) -> tuple[object, bool]:
    """Return ``(source, is_webcam)`` for an Ultralytics ``source`` value."""
    text = raw.strip()
    if text.lower() in {"webcam", "camera"}:
        return 0, True
    if text.isdigit():
        return int(text), True
    if "://" in text:
        return text, False

    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path), False


def discover_weights() -> Path | None:
    """Return the newest ``best.pt`` under ``runs/`` (if any)."""
    runs_dir = PROJECT_ROOT / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        runs_dir.rglob("weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_names(path: Path) -> dict[int, str]:
    """Read a JSON mapping of class ID to display name."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict) and "names" in raw:
        raw = raw["names"]

    if isinstance(raw, list):
        return {index: str(name) for index, name in enumerate(raw)}

    result: dict[int, str] = {}
    for key, value in raw.items():
        if str(key).startswith("_") or not isinstance(value, str):
            continue
        try:
            result[int(key)] = value
        except ValueError:
            continue
    return result


def main() -> int:
    args = build_parser().parse_args()

    # Resolve relative paths from the repository root regardless of the
    # current working directory.
    os.chdir(PROJECT_ROOT)

    source, is_webcam = parse_source(args.source)

    if args.weights:
        weights = Path(args.weights).expanduser().resolve()
    else:
        weights = discover_weights()

    if weights is None or not weights.is_file():
        print("ERROR: no trained checkpoint found.", file=sys.stderr)
        print(
            "Train a model first, or pass --weights path/to/best.pt.",
            file=sys.stderr,
        )
        return 1

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

    model = YOLO(str(weights))

    if args.names:
        names_path = Path(args.names).expanduser().resolve()
        if not names_path.is_file():
            print(f"ERROR: names file not found: {names_path}", file=sys.stderr)
            return 1
        custom_names = load_names(names_path)
        if custom_names:
            try:
                merged = dict(model.model.names)
                merged.update(custom_names)
                model.model.names = merged
            except Exception:  # noqa: BLE001 - display names are optional
                print(
                    "WARNING: could not apply custom class names.",
                    file=sys.stderr,
                )

    show = args.show if args.show is not None else is_webcam
    project_dir = Path(args.project).expanduser().resolve()

    print("=" * 72)
    print("YOLO prediction")
    print("=" * 72)
    print(f"Model  : {weights}")
    print(f"Source : {source}")
    print(f"Image  : {args.imgsz}")
    print(f"Conf   : {args.conf}")
    print(f"IoU    : {args.iou}")
    print(f"Device : {args.device}")
    print(f"Show   : {show}")
    print(f"Output : {project_dir / args.name}")
    print("=" * 72)

    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        show=show,
        save_txt=args.save_txt,
        save_conf=args.save_conf and args.save_txt,
        classes=args.classes,
        max_det=args.max_det,
        line_width=args.line_width,
        project=str(project_dir),
        name=args.name,
        exist_ok=args.exist_ok,
        stream=True,
        verbose=True,
    )

    rows: list[dict] = []
    frame_index = 0

    for result in results:
        frame_index += 1
        if not args.save_csv:
            continue

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        source_name = Path(str(result.path)).name
        names = getattr(result, "names", {}) or {}
        for box, class_id, confidence in zip(
            boxes.xyxy.tolist(),
            boxes.cls.tolist(),
            boxes.conf.tolist(),
        ):
            rows.append(
                {
                    "frame": frame_index,
                    "source": source_name,
                    "class_id": int(class_id),
                    "class_name": names.get(int(class_id), ""),
                    "confidence": round(float(confidence), 4),
                    "x1": round(float(box[0]), 1),
                    "y1": round(float(box[1]), 1),
                    "x2": round(float(box[2]), 1),
                    "y2": round(float(box[3]), 1),
                }
            )

    save_dir = Path(
        getattr(model.predictor, "save_dir", project_dir / args.name)
    )

    if args.save_csv:
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / "detections.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "frame",
                    "source",
                    "class_id",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDetections CSV: {csv_path}")

    print(f"Frames/images processed: {frame_index}")
    print(f"Boxes recorded: {len(rows)}")
    print(f"Results saved to: {save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
