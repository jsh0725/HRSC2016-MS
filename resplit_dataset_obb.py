import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SplitRatios:
    train: float
    val: float
    test: float

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        for name, value in (("train", self.train), ("val", self.val), ("test", self.test)):
            if value <= 0:
                raise ValueError(f"Split ratio must be > 0: {name}={value}")


def _discover_pairs(src_root: Path) -> Dict[str, Tuple[Path, Path]]:
    """
    기존 split(train/val/test)을 모두 훑어서
    동일 stem(파일명) 기준 이미지/라벨 쌍을 수집한다.
    """
    image_dirs = [src_root / "images" / "train", src_root / "images" / "val", src_root / "images" / "test"]
    label_dirs = [src_root / "labels" / "train", src_root / "labels" / "val", src_root / "labels" / "test"]

    image_map: Dict[str, Path] = {}
    label_map: Dict[str, Path] = {}

    for d in image_dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                image_map[p.stem] = p

    for d in label_dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() == ".txt":
                label_map[p.stem] = p

    common = sorted(set(image_map).intersection(label_map))
    return {k: (image_map[k], label_map[k]) for k in common}


def _split_ids(ids: List[str], ratios: SplitRatios) -> Dict[str, List[str]]:
    """
    셔플된 id 목록을 비율에 맞춰 train/val/test로 분할한다.
    반올림 때문에 test가 0이 되는 극단 케이스는 안전하게 보정한다.
    """
    n = len(ids)
    n_train = int(round(n * ratios.train))
    n_val = int(round(n * ratios.val))
    n_test = n - n_train - n_val

    # Guard against edge cases from rounding.
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _link_or_copy(src: Path, dst: Path, prefer_hardlink: bool) -> str:
    """
    기본은 하드링크를 시도하고, 실패 시 copy로 폴백한다.
    하드링크를 쓰면 데이터 중복 저장 없이 새 split 폴더를 만들 수 있다.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if prefer_hardlink:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def _write_dataset_yaml(dst_root: Path) -> None:
    """YOLO 학습에서 바로 사용할 dataset.yaml을 생성한다."""
    yaml_text = (
        f"path: {dst_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: ship\n"
    )
    (dst_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly resplit YOLO OBB dataset into train/val/test.")
    parser.add_argument("--src", type=Path, default=Path("processed_obb"), help="Source dataset root")
    parser.add_argument("--dst", type=Path, default=Path("processed_obb_random"), help="Destination dataset root")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.2, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Always copy files instead of trying hardlinks first",
    )
    args = parser.parse_args()

    # 1) 분할 비율 검증
    ratios = SplitRatios(args.train, args.val, args.test)
    ratios.validate()

    # 2) 원본 데이터에서 이미지/라벨 쌍 수집
    pairs = _discover_pairs(args.src)
    if not pairs:
        raise FileNotFoundError(f"No image/label pairs found under: {args.src}")

    # 3) 출력 폴더 준비
    args.dst.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (args.dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 4) 시드 고정 셔플 후 비율 분할
    ids = list(pairs.keys())
    random.Random(args.seed).shuffle(ids)
    split_map = _split_ids(ids, ratios)

    # 5) 분할 결과를 파일 시스템에 반영(하드링크 우선)
    mode_counter = {"hardlink": 0, "copy": 0}
    for split, split_ids in split_map.items():
        for image_id in split_ids:
            src_img, src_lbl = pairs[image_id]
            dst_img = args.dst / "images" / split / src_img.name
            dst_lbl = args.dst / "labels" / split / f"{image_id}.txt"
            mode = _link_or_copy(src_img, dst_img, prefer_hardlink=not args.copy_only)
            _link_or_copy(src_lbl, dst_lbl, prefer_hardlink=not args.copy_only)
            mode_counter[mode] += 1

    # 6) 학습용 메타 파일 저장
    _write_dataset_yaml(args.dst)
    summary = {
        "source": str(args.src.resolve()),
        "destination": str(args.dst.resolve()),
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "counts": {k: len(v) for k, v in split_map.items()},
        "total_pairs": len(ids),
        "file_mode_primary": mode_counter,
    }
    (args.dst / "resplit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
