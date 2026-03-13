import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# Ultralytics user config path 권한 이슈 방지를 위해 작업 폴더 내부로 고정한다.
os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

from ultralytics import YOLO


# Simple run: python train_yolo_obb.py
DATASET_YAML = Path("processed_obb") / "dataset.yaml"
# Default is YOLO11n OBB
MODEL_SOURCE = "yolo11s-obb.pt"  # .pt or .yaml 
PROJECT_DIR = Path("runs_yolo_obb")
RUN_NAME = "exp04"  # 학습 결과가 저장될 폴더 이름, 필요에 따라 변경

EPOCHS = 250 # 충분히 긴 학습을 위해 150으로 설정, 필요에 따라 조정 가능
IMGSZ = 1024
BATCH = 4
WORKERS = 4
LR0 = 0.0015
COS_LR = True
WEIGHT_DECAY=0.0005

DEVICE = 0  # use "cpu" for CPU 
SEED = 42
PATIENCE = 50
PRETRAINED = True # True로 설정하면 모델이 사전 학습된 가중치를 사용하여 초기화된다. False로 설정하면 모델이 무작위 가중치로 초기화된다.

DEGREES = 3
TRANSLATE = 0.04
SCALE = 0.25
MOSAIC = 0.5
CLOSE_MOSAIC = 20
MIXUP = 0.0


def _parse_float(value: str) -> Optional[float]:
    """
    문자열 값을 float로 안전하게 변환한다.

    입력:
    - value: CSV에서 읽은 문자열 값

    처리:
    - float 변환을 시도한다.
    - 비어 있거나 숫자 변환이 불가능하면 None을 반환한다.

    반환:
    - 변환 성공 시 float
    - 실패 시 None
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_existing_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """
    후보 컬럼명 목록 중 실제 CSV에 존재하는 첫 컬럼명을 찾는다.

    입력:
    - columns: 현재 CSV 헤더 목록
    - candidates: 우선순위가 반영된 후보 컬럼명 목록

    처리:
    - candidates를 앞에서부터 순회하면서 columns 포함 여부를 확인한다.

    반환:
    - 존재하는 첫 컬럼명(str)
    - 없으면 None
    """
    for key in candidates:
        if key in columns:
            return key
    return None


def read_epoch_metrics(results_csv: Path) -> Dict[str, List[Optional[float]]]:
    """
    Ultralytics 학습 로그(results.csv)에서 에폭별 지표를 추출한다.

    입력:
    - results_csv: 학습 완료 후 생성된 결과 CSV 파일 경로

    처리:
    - CSV를 읽어 행 데이터를 메모리로 로드한다.
    - 컬럼명이 버전마다 다를 수 있어, 후보 컬럼 목록에서 실제 존재 컬럼을 동적으로 선택한다.
      예) mAP 관련 컬럼이 (B)/(O) 등으로 달라지는 경우 대응
    - train/val loss는 box/cls/dfl loss를 합산해 단일 loss 곡선으로 구성한다.
    - 탐지 태스크에는 분류 accuracy가 없으므로 accuracy 대체 지표로 mAP50 계열을 우선 사용한다.

    반환:
    - epoch: 에폭 번호 리스트
    - train_loss: 에폭별 train loss 리스트
    - val_loss: 에폭별 val loss 리스트
    - accuracy_like: mAP50 기반 정확도 대체 지표 리스트
    - accuracy_metric_name: 실제로 선택된 컬럼명
    """
    rows: List[dict] = []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"epoch": [], "train_loss": [], "val_loss": [], "accuracy_like": []}

    columns = list(rows[0].keys())
    epoch_key = _pick_existing_column(columns, ["epoch"])
    train_loss_keys = [
        k
        for k in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
        if k in columns
    ]
    val_loss_keys = [
        k
        for k in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]
        if k in columns
    ]
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
    """
    에폭별 손실/정확도 대체 지표를 그래프로 저장한다.

    입력:
    - metrics: read_epoch_metrics()로 추출한 지표 딕셔너리
    - output_dir: 그래프 저장 폴더

    처리:
    - matplotlib가 설치되어 있으면 시각화를 수행한다.
    - loss 그래프: train_loss, val_loss를 같은 축에 출력
    - accuracy 그래프: accuracy_like(mAP50 계열) 출력
    - 각 그래프를 PNG 파일로 저장

    출력 파일:
    - loss_curve.png
    - accuracy_curve.png

    참고:
    - matplotlib가 없으면 예외를 발생시키지 않고 조용히 반환한다.
    """
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
        plt.plot(epochs[:min_len_acc], acc[:min_len_acc], label=metrics.get("accuracy_metric_name", "accuracy_like"))
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("YOLO OBB Accuracy-like Metric")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_curve.png", dpi=160)
        plt.close()


def main() -> None:
    """
    YOLO OBB 학습 및 에폭별 지표 저장/시각화의 전체 실행 진입점.

    동작 순서:
    1. 데이터셋 yaml 존재 여부 확인
    2. YOLO OBB 모델 로드
    3. 학습 실행(model.train)
    4. 학습 결과 폴더의 results.csv 로드
    5. 에폭별 지표 JSON 저장
    6. loss/accuracy-like 그래프 PNG 저장

    실행 결과:
    - 학습 산출물: runs_yolo_obb/<RUN_NAME>/
    - 추가 지표 파일: epoch_metrics.json
    - 시각화 파일: loss_curve.png, accuracy_curve.png
    """
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
        weight_decay = WEIGHT_DECAY,
        degrees = DEGREES,
        mixup = MIXUP,
        translate = TRANSLATE,
        scale = SCALE,
        mosaic = MOSAIC,
        close_mosaic = CLOSE_MOSAIC,
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
    #학습 설정 저장(실험 재현)
    (run_dir / "train_config.json").write_text(
    json.dumps({
        "model": MODEL_SOURCE,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "lr0": LR0,
        "cos_lr": COS_LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "degrees": DEGREES,
        "mixup": MIXUP,
        "translate": TRANSLATE,
        "scale": SCALE,
        "mosaic": MOSAIC,
        "close_mosaic": CLOSE_MOSAIC
    }, indent=2),
    encoding="utf-8"
)
    save_visualization(epoch_metrics, run_dir)

    print(f"Training complete: {run_dir}")
    print(f"Saved metrics: {run_dir / 'epoch_metrics.json'}")
    print(f"Saved plots: {run_dir / 'loss_curve.png'}, {run_dir / 'accuracy_curve.png'}")


if __name__ == "__main__":
    main()
