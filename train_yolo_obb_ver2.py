import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

# Keep Ultralytics cache/config in project-local path to avoid permission issues.
os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

# Simple run: python train_yolo_obb_ver2.py
DATASET_YAML = Path("processed_obb_random") / "dataset.yaml"
MODEL_SOURCE = "yolo11m-obb.pt"  # .pt or .yaml
PROJECT_DIR = Path("runs_yolo_obb")
RUN_NAME = "exp08"

EPOCHS = 500
IMGSZ = 1024
BATCH = 2
WORKERS = 2
LR0 = 0.0010
COS_LR = True
WEIGHT_DECAY = 0.00035

DEVICE = 0
SEED = 42
PATIENCE = 40
PRETRAINED = True

# Weak augmentation profile
DEGREES = 1.0
TRANSLATE = 0.02
SCALE = 0.10
MOSAIC = 0.15
CLOSE_MOSAIC = 10
FLIPLR = 0.3
HSV_H = 0.005
HSV_S = 0.2
HSV_V = 0.2
ERASING = 0.0


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_existing_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in columns:
            return key
    return None


def read_epoch_metrics(results_csv: Path) -> Dict[str, List[Optional[float]]]:
    rows: List[dict] = []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    if not rows:
        return {"epoch": [], "train_loss": [], "val_loss": [], "accuracy_like": []}

    columns = list(rows[0].keys())
    epoch_key = _pick_existing_column(columns, ["epoch"])
    train_loss_keys = [k for k in ["train/box_loss", "train/cls_loss", "train/dfl_loss"] if k in columns]
    val_loss_keys = [k for k in ["val/box_loss", "val/cls_loss", "val/dfl_loss"] if k in columns]
    acc_key = _pick_existing_column(
        columns,
        [
            "metrics/mAP50(B)",
            "metrics/mAP50(O)",
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95(O)",
            "metrics/precision(B)",
            "metrics/precision(O)",
        ],
    )

    epochs: List[Optional[float]] = []
    train_losses: List[Optional[float]] = []
    val_losses: List[Optional[float]] = []
    acc_values: List[Optional[float]] = []

    for row in rows:
        epochs.append(_parse_float(row.get(epoch_key, "")) if epoch_key else None)

        tvals = [_parse_float(row.get(k, "")) for k in train_loss_keys]
        tvals = [v for v in tvals if v is not None]
        train_losses.append(sum(tvals) if tvals else None)

        vvals = [_parse_float(row.get(k, "")) for k in val_loss_keys]
        vvals = [v for v in vvals if v is not None]
        val_losses.append(sum(vvals) if vvals else None)

        acc_values.append(_parse_float(row.get(acc_key, "")) if acc_key else None)

    return {
        "epoch": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "accuracy_like": acc_values,
        "accuracy_metric_name": acc_key or "N/A",
    }


def save_visualization(metrics: Dict[str, List[Optional[float]]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [int(e) for e in metrics["epoch"] if e is not None]
    train_loss = [v for v in metrics["train_loss"] if v is not None]
    val_loss = [v for v in metrics["val_loss"] if v is not None]
    acc = [v for v in metrics["accuracy_like"] if v is not None]

    min_len = min(len(epochs), len(train_loss), len(val_loss))
    if min_len > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs[:min_len], train_loss[:min_len], label="train_loss")
        plt.plot(epochs[:min_len], val_loss[:min_len], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("YOLO OBB Training Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png", dpi=160)
        plt.close()

    min_len_acc = min(len(epochs), len(acc))
    if min_len_acc > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(
            epochs[:min_len_acc],
            acc[:min_len_acc],
            label=metrics.get("accuracy_metric_name", "accuracy_like"),
        )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("YOLO OBB Accuracy-like Metric")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_curve.png", dpi=160)
        plt.close()


def main() -> None:
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATASET_YAML}\nRun `python processing_obb.py` first."
        )

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_SOURCE)
    results = model.train(
        data=str(DATASET_YAML),
        task="obb",
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        seed=SEED,
        lr0=LR0,
        cos_lr=COS_LR,
        weight_decay=WEIGHT_DECAY,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        mosaic=MOSAIC,
        close_mosaic=CLOSE_MOSAIC,
        fliplr=FLIPLR,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        erasing=ERASING,
        patience=PATIENCE,
        pretrained=PRETRAINED,
        verbose=True,
    )

    run_dir = Path(results.save_dir)
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    epoch_metrics = read_epoch_metrics(results_csv)
    (run_dir / "epoch_metrics.json").write_text(
        json.dumps(epoch_metrics, indent=2), encoding="utf-8"
    )

    (run_dir / "train_config.json").write_text(
        json.dumps(
            {
                "model": MODEL_SOURCE,
                "epochs": EPOCHS,
                "imgsz": IMGSZ,
                "batch": BATCH,
                "workers": WORKERS,
                "lr0": LR0,
                "cos_lr": COS_LR,
                "weight_decay": WEIGHT_DECAY,
                "seed": SEED,
                "patience": PATIENCE,
                "pretrained": PRETRAINED,
                "degrees": DEGREES,
                "translate": TRANSLATE,
                "scale": SCALE,
                "mosaic": MOSAIC,
                "close_mosaic": CLOSE_MOSAIC,
                "fliplr": FLIPLR,
                "hsv_h": HSV_H,
                "hsv_s": HSV_S,
                "hsv_v": HSV_V,
                "erasing": ERASING,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    save_visualization(epoch_metrics, run_dir)

    print(f"Training complete: {run_dir}")
    print(f"Saved metrics: {run_dir / 'epoch_metrics.json'}")
    print(f"Saved plots: {run_dir / 'loss_curve.png'}, {run_dir / 'accuracy_curve.png'}")


if __name__ == "__main__":
    main()
