# YOLO11 Object Detection

A clean, reproducible Ultralytics YOLO11 training pipeline for a **35-class
object-detection dataset**. The project provides independent training scripts
for YOLO11n and YOLO11s, a reusable evaluation script with model comparison,
and a prediction script that works on images, folders, video, and webcams.

The dataset is intentionally **not** committed to this repository. Download
it separately (see below), extract it next to `data.yaml`, and the scripts will
find it automatically.
 If you want Original Dataset you can download it from - https://drive.google.com/file/d/1NFMPXsI8RGnXomLqamtXp0iCXGoQZZks/view?usp=sharing

---

## Table of contents

- [Features](#features)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Experimenting with other YOLO models](#experimenting-with-other-yolo-models)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [GPU and CPU execution](#gpu-and-cpu-execution)
- [Example command sheet](#example-command-sheet)
- [Reproducibility notes](#reproducibility-notes)
- [Class names](#class-names)
- [Baseline results](#baseline-results)
- [Troubleshooting](#troubleshooting)
- [Provenance and assumptions](#provenance-and-assumptions)

---

## Features

- **Two independent training scripts** — `train_yolo11n.py` and
  `train_yolo11s.py`, each self-contained and easy to modify.
- **Original training configuration preserved** — epochs, image size, batch
  size, patience, optimiser, augmentation, and seed match the original
  YOLO11n experiment.
- **Any Ultralytics detection model** can be used through `--model`
  (`yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`, or a local checkpoint).
- **Reproducible by default** — fixed seed, deterministic mode, fixed dataset
  splits, and run names that never overwrite previous results.
- **Portable paths** — `data.yaml` uses relative paths; all scripts resolve
  paths from the repository root.
- **Evaluation with comparison** — evaluate one or many checkpoints on `val`
  and/or `test`, save JSON/CSV metrics, and print a comparison table.
- **Flexible prediction** — image, folder, video, webcam, glob, or stream URL,
  with configurable confidence, IoU, image size, device, and class filters.
- **Small dependency footprint** — Ultralytics plus PyTorch and PyYAML.

---

## Project structure

```text
YOLO11-Object-Detection/
├── training/
│   ├── train_yolo11n.py      # YOLO11n training entry point
│   └── train_yolo11s.py      # YOLO11s training entry point
├── evaluation/
│   └── evaluate.py           # val/test evaluation + model comparison
├── prediction/
│   └── predict.py            # image / folder / video / webcam inference
├── data.yaml                 # 35 classes, relative dataset paths
├── requirements.txt
├── .gitignore
└── README.md

# Generated at runtime (ignored by Git):
runs/train/<run-name>/        # training outputs and weights
evaluation/results/<run>/     # evaluation metrics and plots
prediction/outputs/<name>/    # annotated predictions
dataset/                      # your extracted dataset (not committed)
```

---

## Dataset

### Download

The dataset is distributed separately because it is too large for Git.

> **Download link:** https://drive.google.com/file/d/1NFMPXsI8RGnXomLqamtXp0iCXGoQZZks/view?usp=sharing

### Expected layout

Extract the dataset ZIP so that the project root contains a `dataset/` folder
with exactly this structure:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

`data.yaml` already points at this layout:

```yaml
path: dataset
train: images/train
val: images/val
test: images/test
nc: 35
```

### Extraction

**Windows (PowerShell):**

```powershell
# Run from the project root (the folder containing data.yaml)
Expand-Archive -Path path\to\dataset.zip -DestinationPath .
```

**Linux / macOS:**

```bash
# Run from the project root (the folder containing data.yaml)
unzip path/to/dataset.zip -d .
```

After extraction, verify the layout:

```powershell
# Windows
Get-ChildItem dataset\images
Get-ChildItem dataset\labels
```

```bash
# Linux / macOS
ls dataset/images
ls dataset/labels
```

### Dataset statistics from the original project

| Split | Images | Boxes  |
| ----- | ------ | ------ |
| train | 11,526 | 16,316 |
| val   | 2,982  | 4,408  |
| test  | 795    | 1,220  |
| total | 15,303 | 21,944 |

The dataset is **not** uploaded to GitHub. The `.gitignore` excludes
`dataset/`, image/label folders, archives, caches, and training outputs.

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For **GPU training**, install the PyTorch build that matches your CUDA driver
first, then install the remaining requirements:

```bash
# Example for CUDA 12.x — check https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> The original project environment used Python 3.11, Ultralytics 8.4.208,
> PyTorch 2.11.0, and OpenCV 5.0.0. The scripts here were syntax-checked and
> their command-line interfaces validated in that environment. Other recent
> versions should work, but pin them if you need exact reproducibility.

### 3. Verify the environment

```bash
python -c "import ultralytics, torch, yaml; print('ultralytics', ultralytics.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

The first training or prediction run downloads the requested pretrained
weights (`yolo11n.pt`, `yolo11s.pt`, ...) automatically. This requires an
internet connection. To work fully offline, download the weights manually and
pass them with `--model path/to/yolo11n.pt`.

---

## Training

Both training scripts reuse the configuration of the original YOLO11n
experiment (`full_yolo11n_640_v3`). The only difference between the two files
is the default model and run name.

### Training configuration

| Parameter         | Value          | Meaning                                      |
| ----------------- | -------------- | -------------------------------------------- |
| `epochs`          | 60             | Maximum training epochs                      |
| `imgsz`           | 640            | Input image size                             |
| `batch`           | 16             | Images per batch                             |
| `patience`        | 15             | Early stopping patience                      |
| `optimizer`       | `auto`         | Ultralytics chooses AdamW/SGD                |
| `lr0`             | 0.01           | Initial learning rate                        |
| `lrf`             | 0.01           | Final learning rate factor                   |
| `momentum`        | 0.937          | SGD momentum / Adam beta1                    |
| `weight_decay`    | 0.0005         | L2 regularisation                            |
| `warmup_epochs`   | 3.0            | Learning-rate warmup                         |
| `seed`            | 0              | Random seed                                  |
| `deterministic`   | `True`         | Deterministic torch algorithms               |
| `close_mosaic`    | 10             | Disable mosaic for the last 10 epochs        |
| `amp`             | `True`         | Automatic mixed precision                    |
| `workers`         | 2              | Dataloader workers                           |
| `cache`           | `False`        | Do not cache the dataset in RAM              |
| `device`          | `0`            | First GPU                                    |
| `fraction`        | 1.0            | Use the full training split                  |
| `mosaic`          | 1.0            | Mosaic augmentation probability              |
| `fliplr`          | 0.5            | Horizontal flip probability                  |
| `scale`           | 0.5            | Random scale augmentation                    |
| `translate`       | 0.1            | Random translation                           |
| `erasing`         | 0.4            | Random erasing probability                   |
| `auto_augment`    | `randaugment`  | Automatic augmentation policy                |

All values are defined in the `TRAIN_CONFIG` dictionary at the top of each
training script. The most common ones can also be overridden from the command
line.

### Train YOLO11n

```bash
python training/train_yolo11n.py
```

Equivalent explicit command:

```bash
python training/train_yolo11n.py \
  --model yolo11n.pt \
  --data data.yaml \
  --epochs 60 \
  --imgsz 640 \
  --batch 16 \
  --patience 15 \
  --device 0 \
  --seed 0 \
  --name yolo11n_640
```

Outputs are written to:

```text
runs/train/yolo11n_640/
├── weights/
│   ├── best.pt
│   └── last.pt
├── args.yaml
├── results.csv
├── results.png
├── confusion_matrix.png
└── ...
```

### Train YOLO11s

```bash
python training/train_yolo11s.py
```

Equivalent explicit command:

```bash
python training/train_yolo11s.py \
  --model yolo11s.pt \
  --data data.yaml \
  --epochs 60 \
  --imgsz 640 \
  --batch 16 \
  --patience 15 \
  --device 0 \
  --seed 0 \
  --name yolo11s_640
```

Outputs are written to `runs/train/yolo11s_640/`.

### Resume a run

```bash
python training/train_yolo11n.py \
  --model runs/train/yolo11n_640/weights/last.pt \
  --resume
```

### Useful overrides

```bash
# CPU training
python training/train_yolo11n.py --device cpu

# Smaller batch if you run out of GPU memory
python training/train_yolo11n.py --batch 8

# Quick smoke test (not a real training run)
python training/train_yolo11n.py --epochs 2 --imgsz 320 --name smoke

# Cache images for faster epochs (needs enough RAM/disk)
python training/train_yolo11n.py --cache
```

> **Run names never overwrite previous results.** `exist_ok` is off by default,
> so Ultralytics creates `yolo11n_6402`, `yolo11n_6403`, ... if the name is
> already taken. Pass `--exist-ok` only if you intentionally want to reuse a
> folder.

---

## Experimenting with other YOLO models

The training pipeline is not tied to YOLO11n or YOLO11s. Pass any Ultralytics
detection checkpoint or model name with `--model`:

```bash
# YOLO11m
python training/train_yolo11n.py --model yolo11m.pt --name yolo11m_640

# YOLO11l
python training/train_yolo11n.py --model yolo11l.pt --name yolo11l_640

# YOLO11x
python training/train_yolo11n.py --model yolo11x.pt --name yolo11x_640

# A local checkpoint
python training/train_yolo11n.py \
  --model path/to/custom_weights.pt \
  --name custom_experiment
```

You can also copy a training script and change the `--model` / `--name`
defaults in its argument parser. No other code changes are required.

> Larger models need more GPU memory. Reduce `--batch` (for example to 8 or 4)
> or `--imgsz` (for example to 512) if you get a CUDA out-of-memory error.

---

## Evaluation

`evaluation/evaluate.py` loads one or more checkpoints, evaluates them on the
validation and/or test split, prints **Precision**, **Recall**, **mAP50**, and
**mAP50-95**, and saves the results.

### Evaluate the newest checkpoint on both splits

```bash
python evaluation/evaluate.py
```

### Evaluate a specific checkpoint

```bash
python evaluation/evaluate.py \
  --weights runs/train/yolo11n_640/weights/best.pt
```

### Evaluate only the test split

```bash
python evaluation/evaluate.py \
  --weights runs/train/yolo11n_640/weights/best.pt \
  --split test
```

### Compare multiple trained models

```bash
python evaluation/evaluate.py \
  --weights runs/train/yolo11n_640/weights/best.pt \
            runs/train/yolo11s_640/weights/best.pt
```

The comparison table is printed to the terminal and saved to
`evaluation/results/comparison.csv` and `evaluation/results/comparison.json`.

### Output layout

```text
evaluation/results/
├── yolo11n_640/
│   ├── val/
│   │   ├── summary.json
│   │   ├── per_class.csv
│   │   ├── confusion_matrix.png
│   │   ├── BoxPR_curve.png
│   │   └── ...
│   └── test/
│       └── ...
├── comparison.csv
└── comparison.json
```

### Evaluation options

| Option            | Default              | Description                              |
| ----------------- | -------------------- | ---------------------------------------- |
| `--weights`       | newest `best.pt`     | One or more checkpoint paths             |
| `--data`          | `data.yaml`          | Dataset configuration                    |
| `--split`         | `val test`           | Any of `val`, `test`                     |
| `--imgsz`         | 640                  | Validation image size                    |
| `--batch`         | 16                   | Validation batch size                    |
| `--device`        | `0`                  | GPU index, `cpu`, or `mps`               |
| `--workers`       | 2                    | Dataloader workers                       |
| `--output`        | `evaluation/results` | Directory for metrics and plots          |
| `--plots`         | on                   | Save confusion matrices and PR/F1 curves |
| `--verbose`       | on                   | Print Ultralytics per-class output       |

---

## Prediction

`prediction/predict.py` loads a trained checkpoint and runs inference on an
image, folder, video, webcam, glob, or stream URL. Annotated results are saved
by default.

### Single image

```bash
python prediction/predict.py \
  --weights runs/train/yolo11n_640/weights/best.pt \
  --source path/to/image.jpg \
  --show
```

### Folder of images

```bash
python prediction/predict.py \
  --weights runs/train/yolo11n_640/weights/best.pt \
  --source path/to/images \
  --conf 0.40
```

### Video

```bash
python prediction/predict.py \
  --weights runs/train/yolo11n_640/weights/best.pt \
  --source path/to/video.mp4 \
  --show
```

### Webcam

```bash
python prediction/predict.py \
  --weights runs/train/yolo11n_640/weights/best.pt \
  --source 0
```

`--source 0` (or `--source webcam`) opens the default camera. `--show` is
enabled automatically for webcams, so a window opens. Press `q` in the window
to stop.

### Save a CSV of detections

```bash
python prediction/predict.py \
  --weights runs/train/yolo11n_640/weights/best.pt \
  --source path/to/images \
  --save-csv
```

The CSV contains one row per detected box: `frame`, `source`, `class_id`,
`class_name`, `confidence`, `x1`, `y1`, `x2`, `y2`.

### Output layout

```text
prediction/outputs/predict/
├── image.jpg                 # annotated image
├── detections.csv            # only with --save-csv
└── ...
```

### Prediction options

| Option          | Default               | Description                                       |
| --------------- | --------------------- | ------------------------------------------------- |
| `--weights`     | newest `best.pt`      | Trained checkpoint                                |
| `--source`      | required              | Image, folder, video, webcam index, glob, or URL  |
| `--imgsz`       | 640                   | Inference image size                              |
| `--conf`        | 0.25                  | Confidence threshold                              |
| `--iou`         | 0.45                  | NMS IoU threshold                                 |
| `--device`      | `0`                   | GPU index, `cpu`, or `mps`                        |
| `--project`     | `prediction/outputs`  | Output directory                                  |
| `--name`        | `predict`             | Output run name                                   |
| `--save`        | on                    | Save annotated images/video                       |
| `--show`        | webcam: on, else off  | Display results in a window                       |
| `--save-txt`    | off                   | Save YOLO-format `.txt` labels                    |
| `--save-csv`    | off                   | Save `detections.csv`                             |
| `--classes`     | all                   | Keep only selected class IDs                      |
| `--max-det`     | 300                   | Maximum detections per image/frame                |
| `--line-width`  | 2                     | Bounding-box line width                           |
| `--names`       | none                  | Optional JSON mapping of class IDs to display names |

Example custom names file (`names.json`):

```json
{
  "0": "person",
  "22": "backpack"
}
```

```bash
python prediction/predict.py \
  --weights runs/train/yolo11n_640/weights/best.pt \
  --source path/to/images \
  --names names.json
```

> Custom names only change the text drawn on the boxes. They do not change the
> trained weights or the numeric class IDs.

---

## GPU and CPU execution

All three scripts accept `--device`.

| Value     | Meaning                                      |
| --------- | -------------------------------------------- |
| `0`       | First CUDA GPU (default)                     |
| `1`       | Second CUDA GPU                              |
| `0,1`     | Use two GPUs                                 |
| `cpu`     | CPU only                                     |
| `mps`     | Apple Silicon GPU (macOS)                    |

Examples:

```bash
# GPU training
python training/train_yolo11n.py --device 0

# CPU training
python training/train_yolo11n.py --device cpu --batch 8

# GPU evaluation
python evaluation/evaluate.py --device 0

# CPU evaluation
python evaluation/evaluate.py --device cpu --batch 8

# GPU prediction
python prediction/predict.py --weights runs/train/yolo11n_640/weights/best.pt \
  --source samples --device 0

# CPU prediction
python prediction/predict.py --weights runs/train/yolo11n_640/weights/best.pt \
  --source samples --device cpu
```

CPU training works but is much slower. For a quick CPU smoke test, use a small
number of epochs and a smaller image size:

```bash
python training/train_yolo11n.py --device cpu --epochs 2 --imgsz 320 --batch 4 --name smoke_cpu
```

---

## Example command sheet

```bash
# 1. Install
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Extract the dataset so that dataset/images/{train,val,test} exist

# 3. Train YOLO11n
python training/train_yolo11n.py

# 4. Train YOLO11s
python training/train_yolo11s.py

# 5. Evaluate on val + test
python evaluation/evaluate.py \
  --weights runs/train/yolo11n_640/weights/best.pt \
            runs/train/yolo11s_640/weights/best.pt

# 6. Predict on an image
python prediction/predict.py \
  --weights runs/train/yolo11n_640/weights/best.pt \
  --source path/to/image.jpg --show

# 7. Predict on a webcam
python prediction/predict.py \
  --weights runs/train/yolo11n_640/weights/best.pt \
  --source 0
```

---

## Reproducibility notes

- **Fixed seed:** `seed=0` is passed to Ultralytics, which seeds Python,
  NumPy, and PyTorch.
- **Deterministic mode:** `deterministic=True` asks PyTorch to use
  deterministic algorithms.
- **Fixed splits:** `data.yaml` always points at the same
  `images/{train,val,test}` folders, so the split does not change between runs.
- **No silent overwrites:** run names are unique by default. Reusing a name
  creates a new numbered folder unless `--exist-ok` is passed.
- **Configuration in one place:** every training parameter is listed in
  `TRAIN_CONFIG` at the top of each training script.
- **Recorded run settings:** Ultralytics writes `args.yaml` and `results.csv`
  into each run folder. Keep them with the weights to document an experiment.
- **Version pinning:** for exact reproduction, pin `ultralytics`, `torch`, and
  `torchvision` to the versions you used and record your GPU model and driver.
- **GPU caveat:** some CUDA operations are not bit-for-bit deterministic even
  with `deterministic=True`. Small metric differences between runs on different
  hardware are expected.
- **Original environment:** Python 3.11, Ultralytics 8.4.208, PyTorch 2.11.0
  (CUDA 12.8), OpenCV 5.0.0.

---

## Class names

The dataset contains **35 numeric class IDs (0–34)**. The original dataset
shipped without semantic class names, so `data.yaml` uses placeholders:

```yaml
0: class_00
1: class_01
...
34: class_34
```

**No semantic class name is confirmed.** The original project contains
evidence-based hints that class `0` is likely a whole person/pedestrian and
class `22` is likely a backpack/bag, but those hints were never confirmed and
are **not** used as labels here. If you confirm the meanings, rename the values
in `data.yaml` while keeping the numeric IDs unchanged:

```yaml
0: person
22: backpack
```

Renaming only affects display text. It does not change trained weights.

---

## Baseline results

These metrics come from the original YOLO11n experiment
(`full_yolo11n_640_v3`, 640 px, batch 16, up to 60 epochs, patience 15) and are
provided as a reference point. Retraining on a different machine may produce
slightly different numbers.

| Split | Images | Instances | Precision | Recall | mAP50  | mAP50-95 |
| ----- | ------ | --------- | --------- | ------ | ------ | -------- |
| val   | 2,982  | 4,408     | 0.8463    | 0.7745 | 0.8343 | 0.6323   |
| test  | 795    | 1,220     | 0.8425    | 0.7940 | 0.8470 | 0.6465   |

---

## Troubleshooting

### `dataset config not found` or `train split not found`

- Make sure `data.yaml` is in the project root.
- Make sure the extracted dataset is at `dataset/images/train`,
  `dataset/images/val`, and `dataset/images/test`.
- Run the scripts from the project root, or use the provided entry points
  (they resolve paths from the repository root automatically).

### CUDA out of memory

Lower the batch size or image size:

```bash
python training/train_yolo11n.py --batch 8
python training/train_yolo11n.py --batch 4 --imgsz 512
```

### No GPU available

Use `--device cpu` for training, evaluation, and prediction.

### Windows dataloader errors

Set the worker count to `0`:

```bash
python training/train_yolo11n.py --workers 0
```

### Pretrained weights cannot be downloaded

Ultralytics downloads `yolo11n.pt` / `yolo11s.pt` on first use. If you are
offline, download the weights on another machine and pass them explicitly:

```bash
python training/train_yolo11n.py --model path/to/yolo11n.pt
```

### Prediction window does not open

`--show` requires a desktop session. On a headless server, omit `--show` and
use the saved files in `prediction/outputs/`.

### `--source 0` opens the wrong camera

Try another index, for example `--source 1` or `--source 2`.

---

## Provenance and assumptions

- The dataset configuration (`nc: 35`, class IDs `0`–`34`, split names, and
  placeholder class names) was copied from the original project's
  `model_training/data.yaml`.
- The training configuration was reconstructed from the original project's
  final YOLO11n run settings (`runs/full_yolo11n_640_v3/args.yaml`):
  `epochs=60`, `imgsz=640`, `batch=16`, `patience=15`, `optimizer=auto`,
  `seed=0`, `deterministic=True`, `close_mosaic=10`, `amp=True`,
  `workers=2`, `cache=False`, `device=0`, `fraction=1.0`, and the same
  augmentation and loss weights.
- The original project did not contain a standalone Python training script;
  training was launched through the Ultralytics CLI. This project turns that
  recorded configuration into reproducible Python entry points.
- The original run recorded a non-standard `cls_remap: true` argument. It is
  not part of standard Ultralytics and is intentionally omitted here.
- Trained weights, ONNX exports, training runs, evaluation outputs, and the
  dataset are **not** copied into this repository. They are generated locally
  and excluded by `.gitignore`.
- The class names remain placeholders because the original dataset did not
  provide confirmed semantic names.
