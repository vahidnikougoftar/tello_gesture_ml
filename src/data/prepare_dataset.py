"""Download and reshape the Sign Language Digits dataset for drone gestures."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.request import urlretrieve
import zipfile

import random

DATASET_URL = (
    "https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/refs/heads/master.zip"
)

GESTURE_MAP: Dict[str, str] = {
    "0": "takeoff",
    "1": "land",
    "2": "forward",
    "3": "backward",
    "4": "left",
    "5": "right",
    "6": "spin",
}


def download_dataset(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "sign_language_digits.zip"
    if zip_path.exists():
        print(f"[skip] zip already present at {zip_path}")
    else:
        print(f"[download] fetching dataset into {zip_path}")
        urlretrieve(DATASET_URL, zip_path)

    extract_dir = raw_dir / "Sign-Language-Digits-Dataset-master"
    if extract_dir.exists():
        print(f"[skip] dataset already extracted at {extract_dir}")
    else:
        print(f"[extract] inflating archive to {raw_dir}")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(raw_dir)
    return extract_dir


def collect_images(dataset_dir: Path) -> Dict[str, List[Path]]:
    image_paths: Dict[str, List[Path]] = {}
    valid_suffixes = {".png", ".jpg", ".jpeg"}
    for digit, gesture in GESTURE_MAP.items():
        digit_dir = dataset_dir / digit
        if not digit_dir.exists():
            raise FileNotFoundError(f"Missing digit folder {digit_dir}")
        files = [
            path
            for path in digit_dir.iterdir()
            if path.is_file() and path.suffix.lower() in valid_suffixes
        ]
        if not files:
            raise RuntimeError(f"No images found for digit {digit}")
        image_paths[gesture] = files
    return image_paths


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_and_copy(
    image_paths: Dict[str, List[Path]],
    processed_dir: Path,
    train_ratio: float,
    seed: int,
) -> None:
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    reset_dir(train_dir)
    reset_dir(val_dir)

    rng = random.Random(seed)
    for gesture, files in image_paths.items():
        files_copy = list(files)
        rng.shuffle(files_copy)
        split_index = max(1, int(len(files_copy) * train_ratio))
        train_files = files_copy[:split_index]
        val_files = files_copy[split_index:]
        if not val_files:
            val_files = train_files[-1:]
            train_files = train_files[:-1]

        gesture_train = train_dir / gesture
        gesture_val = val_dir / gesture
        gesture_train.mkdir(parents=True, exist_ok=True)
        gesture_val.mkdir(parents=True, exist_ok=True)

        copy_files(train_files, gesture_train)
        copy_files(val_files, gesture_val)


def copy_files(files: Iterable[Path], destination: Path) -> None:
    for source in files:
        target = destination / source.name
        shutil.copy2(source, target)


def write_metadata(processed_dir: Path) -> None:
    mapping_path = processed_dir / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as fp:
        json.dump(GESTURE_MAP, fp, indent=2)
    print(f"[meta] wrote {mapping_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and reshape the Sign Language Digits dataset"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where raw downloads live",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for train/val splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of samples that go into the training set",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    args = parser.parse_args()

    extracted = download_dataset(args.raw_dir)
    dataset_dir = extracted / "Dataset"
    images = collect_images(dataset_dir)
    split_and_copy(images, args.processed_dir, args.train_ratio, args.seed)
    write_metadata(args.processed_dir)
    print("[done] dataset ready")


if __name__ == "__main__":
    main()
