import json
import os
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO


# Ultralytics config path (permission-safe)
os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))


# Simple run: python eval_test_yolo_obb.py
WEIGHTS = Path("runs") / "obb" / "runs_yolo_obb" / "exp08" / "weights" / "best.pt"
DATASET_YAML = Path("processed_obb_random") / "dataset.yaml"
TEST_IMAGES_DIR = Path("processed_obb_random") / "images" / "test"

PROJECT_DIR = Path("runs_eval_obb")
RUN_NAME = "test_exp08"

DEVICE = 0  # use "cpu" for CPU
IMGSZ = 1024
CONF = 0.3
IOU = 0.7
MAX_VIS_IMAGES = 100  # 시각화용 이미지 개수 제한(너무 크면 실행 시간이 길어짐)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metrics_to_dict(metrics_obj: Any) -> Dict[str, float | None]:
    """
    Ultralytics 평가 객체에서 핵심 지표만 안전하게 추출한다.
    """
    out: Dict[str, float | None] = {}

    # OBB도 보통 box 네임스페이스를 통해 precision/recall/mAP 값을 제공한다.
    box = getattr(metrics_obj, "box", None)
    if box is not None:
        out["precision"] = _safe_float(getattr(box, "mp", None))
        out["recall"] = _safe_float(getattr(box, "mr", None))
        out["mAP50"] = _safe_float(getattr(box, "map50", None))
        out["mAP50_95"] = _safe_float(getattr(box, "map", None))
    else:
        out["precision"] = None
        out["recall"] = None
        out["mAP50"] = None
        out["mAP50_95"] = None

    out["fitness"] = _safe_float(getattr(metrics_obj, "fitness", None))
    return out


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"가중치 파일이 없습니다: {WEIGHTS}")
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"데이터셋 YAML이 없습니다: {DATASET_YAML}")
    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(f"테스트 이미지 폴더가 없습니다: {TEST_IMAGES_DIR}")

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))

    # 1) 정량 평가 (test split)
    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        task="obb",
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        plots=True,
        verbose=True,
    )

    save_dir = Path(metrics.save_dir)
    summary = {
        "weights": str(WEIGHTS.resolve()),
        "dataset_yaml": str(DATASET_YAML.resolve()),
        "test_images_dir": str(TEST_IMAGES_DIR.resolve()),
        "run_dir": str(save_dir.resolve()),
        "metrics": metrics_to_dict(metrics),
    }
    (save_dir / "test_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 2) 정성 시각화 (test 이미지 일부만 예측)
    test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not test_images:
        raise FileNotFoundError(f"테스트 이미지가 없습니다: {TEST_IMAGES_DIR}")
    vis_images = [str(p) for p in test_images[:MAX_VIS_IMAGES]]

    model.predict(
        source=vis_images,
        task="obb",
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        save=True,
        project=str(save_dir),
        name="test_predictions",
        verbose=True,
    )

    print("테스트 평가 완료")
    print(f"평가 폴더: {save_dir}")
    print(f"정량 요약: {save_dir / 'test_metrics_summary.json'}")
    print(f"정성 시각화: {save_dir / 'test_predictions'}")


if __name__ == "__main__":
    main()
