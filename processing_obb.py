import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple
from PIL import Image


# 단순 실행: python processing_obb.py
ARCHIVE_DIR = Path("archive")
OUTPUT_DIR = Path("processed_obb")
COPY_AS_JPG = True
JPEG_QUALITY = 95
MIN_SIDE_PX = 2.0


def read_ids(txt_path: Path) -> List[str]:
    """
    ImageSets 분할 파일(train/val/test)에서 이미지 ID 리스트를 읽는다.
    """
    return [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clamp(v: float, lo: float, hi: float) -> float:
    """
    실수 값을 [lo, hi] 구간으로 제한한다.
    """
    return max(lo, min(hi, v))


def robndbox_to_points(cx: float, cy: float, w: float, h: float, angle: float) -> List[Tuple[float, float]]:
    """
    회전 박스(cx, cy, w, h, angle)를 4개 꼭짓점으로 변환한다.

    반환 순서:
    - 좌상, 우상, 우하, 좌하(회전 후 대응점)
    """
    hw = w / 2.0
    hh = h / 2.0
    local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    points: List[Tuple[float, float]] = []
    for x, y in local:
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        points.append((rx, ry))
    return points


def bndbox_to_points(xmin: float, ymin: float, xmax: float, ymax: float) -> List[Tuple[float, float]]:
    """
    robndbox가 없을 때를 위한 fallback:
    축정렬 bndbox를 4개 꼭짓점으로 변환한다.
    """
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def parse_xml_obb_lines(xml_path: Path, img_w: int, img_h: int) -> List[str]:
    """
    XML 어노테이션을 YOLO-OBB 라벨 라인으로 변환한다.

    출력 라인 형식:
    - class_id x1 y1 x2 y2 x3 y3 x4 y4
    - 좌표는 [0, 1] 정규화
    - ship 단일 클래스이므로 class_id=0
    """
    root = ET.parse(xml_path).getroot()
    lines: List[str] = []

    for obj in root.findall("object"):
        points: List[Tuple[float, float]]
        robnd = obj.find("robndbox")
        if robnd is not None:
            try:
                cx = float(robnd.findtext("cx", default="0"))
                cy = float(robnd.findtext("cy", default="0"))
                w = float(robnd.findtext("w", default="0"))
                h = float(robnd.findtext("h", default="0"))
                angle = float(robnd.findtext("angle", default="0"))
            except ValueError:
                continue
            if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
                continue
            points = robndbox_to_points(cx, cy, w, h, angle)
        else:
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                xmin = float(bnd.findtext("xmin", default="0"))
                ymin = float(bnd.findtext("ymin", default="0"))
                xmax = float(bnd.findtext("xmax", default="0"))
                ymax = float(bnd.findtext("ymax", default="0"))
            except ValueError:
                continue
            if (xmax - xmin) < MIN_SIDE_PX or (ymax - ymin) < MIN_SIDE_PX:
                continue
            points = bndbox_to_points(xmin, ymin, xmax, ymax)

        normalized = []
        for x, y in points:
            nx = clamp(x / img_w, 0.0, 1.0)
            ny = clamp(y / img_h, 0.0, 1.0)
            normalized.extend([nx, ny])

        lines.append("0 " + " ".join(f"{v:.6f}" for v in normalized))

    return lines


def export_image(src_img_path: Path, dst_img_path: Path) -> Tuple[int, int]:
    """
    원본 이미지를 출력 폴더로 복사/변환하고 (width, height)를 반환한다.
    """
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_img_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        if COPY_AS_JPG:
            img.save(dst_img_path, quality=JPEG_QUALITY)
        else:
            img.save(dst_img_path)
    return width, height


def process_split(split_name: str, ids: List[str]) -> dict:
    """
    split(train/val/test) 단위로 OBB 이미지/라벨을 생성한다.

    출력 구조:
    - processed_obb/images/{split}/*.jpg
    - processed_obb/labels/{split}/*.txt
    """
    src_images_dir = ARCHIVE_DIR / "AllImages"
    src_ann_dir = ARCHIVE_DIR / "Annotations"
    out_images_dir = OUTPUT_DIR / "images" / split_name
    out_labels_dir = OUTPUT_DIR / "labels" / split_name
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "input_ids": len(ids),
        "exported_images": 0,
        "label_files": 0,
        "objects": 0,
        "skipped_missing": 0,
        "skipped_invalid": 0,
    }

    for image_id in ids:
        src_img = src_images_dir / f"{image_id}.bmp"
        src_xml = src_ann_dir / f"{image_id}.xml"
        if not src_img.exists() or not src_xml.exists():
            stats["skipped_missing"] += 1
            continue

        ext = ".jpg" if COPY_AS_JPG else src_img.suffix
        dst_img = out_images_dir / f"{image_id}{ext}"
        dst_label = out_labels_dir / f"{image_id}.txt"

        try:
            width, height = export_image(src_img, dst_img)
            lines = parse_xml_obb_lines(src_xml, width, height)
        except (OSError, ET.ParseError):
            stats["skipped_invalid"] += 1
            continue

        dst_label.write_text("\n".join(lines), encoding="utf-8")
        stats["exported_images"] += 1
        stats["label_files"] += 1
        stats["objects"] += len(lines)

    return stats


def write_dataset_yaml() -> None:
    """
    YOLO-OBB 학습용 dataset.yaml 파일을 생성한다.
    """
    yaml_text = (
        f"path: {OUTPUT_DIR.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: ship\n"
    )
    (OUTPUT_DIR / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    """
    HRSC2016-MS를 OBB 포맷(YOLO-OBB TXT)으로 변환한다.
    """
    split_files = {
        "train": ARCHIVE_DIR / "ImageSets" / "train.txt",
        "val": ARCHIVE_DIR / "ImageSets" / "val.txt",
        "test": ARCHIVE_DIR / "ImageSets" / "test.txt",
    }

    missing = [str(p) for p in split_files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files: {missing}")

    summary = {
        "input_archive": str(ARCHIVE_DIR.resolve()),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "format": "YOLO_OBB_TXT",
        "class_map": {"ship": 0},
        "splits": {},
    }

    for split_name, split_file in split_files.items():
        ids = read_ids(split_file)
        summary["splits"][split_name] = process_split(split_name, ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_dataset_yaml()
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
