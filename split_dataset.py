#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".JPEG", ".JPG", ".PNG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly split class-folder dataset into train/val folders."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source dataset root (class subfolders)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data"),
        help="Destination root to create train/ and val/",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio in [0,1). Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split",
    )
    parser.add_argument(
        "--clear-dst",
        action="store_true",
        help="Delete destination train/val before splitting",
    )
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {e.lower() for e in IMAGE_EXTS}
    )


def ensure_dirs(base: Path, class_name: str) -> tuple[Path, Path]:
    train_cls = base / "train" / class_name
    val_cls = base / "val" / class_name
    train_cls.mkdir(parents=True, exist_ok=True)
    val_cls.mkdir(parents=True, exist_ok=True)
    return train_cls, val_cls


def copy_files(files: Iterable[Path], target_dir: Path) -> int:
    count = 0
    for src in files:
        shutil.copy2(src, target_dir / src.name)
        count += 1
    return count


def split_class_files(files: list[Path], val_ratio: float) -> tuple[list[Path], list[Path]]:
    n = len(files)
    if n == 0:
        return [], []
    if n == 1:
        return files, []

    val_count = int(round(n * val_ratio))
    if val_ratio > 0 and val_count == 0:
        val_count = 1
    if val_count >= n:
        val_count = n - 1

    val_files = files[:val_count]
    train_files = files[val_count:]
    return train_files, val_files


def main() -> None:
    args = parse_args()

    if not args.src.exists() or not args.src.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist or is not a directory: {args.src}")

    if not (0 <= args.val_ratio < 1):
        raise ValueError("--val-ratio must be in [0, 1)")

    train_root = args.dst / "train"
    val_root = args.dst / "val"

    if args.clear_dst:
        if train_root.exists():
            shutil.rmtree(train_root)
        if val_root.exists():
            shutil.rmtree(val_root)

    random.seed(args.seed)

    class_dirs = sorted([d for d in args.src.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class subfolders found in: {args.src}")

    total_train = 0
    total_val = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = list_images(class_dir)
        if not files:
            print(f"[skip] {class_name}: no image files")
            continue

        random.shuffle(files)
        train_files, val_files = split_class_files(files, args.val_ratio)
        train_cls_dir, val_cls_dir = ensure_dirs(args.dst, class_name)

        c_train = copy_files(train_files, train_cls_dir)
        c_val = copy_files(val_files, val_cls_dir)
        total_train += c_train
        total_val += c_val

        print(f"[ok] {class_name}: train={c_train}, val={c_val}, total={len(files)}")

    print("\nDone")
    print(f"Source: {args.src}")
    print(f"Output: {args.dst}")
    print(f"Total train: {total_train}")
    print(f"Total val: {total_val}")


if __name__ == "__main__":
    main()
