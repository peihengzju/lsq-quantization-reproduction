#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets import Image, load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Hugging Face ImageNet-1k cache to ImageFolder layout (train/val)."
    )
    p.add_argument(
        "--dataset-id",
        type=str,
        default="ILSVRC/imagenet-1k",
        help="Hugging Face dataset id",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("Dataset"),
        help="HF datasets cache directory",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("data_imagenet1k"),
        help="Output root for ImageFolder format",
    )
    p.add_argument(
        "--include-test",
        action="store_true",
        help="Also export test split (if labels exist).",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class for quick debugging.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output root before export.",
    )
    return p.parse_args()


def split_to_dir(split_name: str) -> str:
    if split_name == "validation":
        return "val"
    return split_name


def write_image(example: dict, dst_file: Path) -> None:
    image_obj = example["image"]
    if isinstance(image_obj, dict):
        # decode=False case: {'bytes': ..., 'path': ...}
        raw = image_obj.get("bytes")
        src_path = image_obj.get("path")
        if raw is not None:
            dst_file.write_bytes(raw)
            return
        if src_path:
            shutil.copy2(src_path, dst_file)
            return

    # fallback: decoded PIL image
    image_obj.save(dst_file, format="JPEG", quality=95)


def export_split(args: argparse.Namespace, split_name: str) -> tuple[int, int]:
    ds = load_dataset(
        args.dataset_id,
        split=split_name,
        cache_dir=str(args.cache_dir),
    )

    # Keep encoded bytes if available for speed and fidelity.
    ds = ds.cast_column("image", Image(decode=False))

    names = ds.features["label"].names
    split_dir = args.out_root / split_to_dir(split_name)
    split_dir.mkdir(parents=True, exist_ok=True)

    class_written = [0 for _ in range(len(names))]
    total = 0

    target_per_class = args.max_per_class
    pbar = tqdm(ds, total=len(ds), desc=f"export-{split_name}", unit="img")
    for idx, ex in enumerate(pbar):
        if target_per_class is not None and all(x >= target_per_class for x in class_written):
            break

        label = int(ex["label"])
        if target_per_class is not None and class_written[label] >= target_per_class:
            continue

        class_dir = split_dir / f"c{label:04d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        dst = class_dir / f"{idx:08d}.jpg"
        write_image(ex, dst)

        class_written[label] += 1
        total += 1

        if total % 5000 == 0:
            pbar.set_postfix(exported=total, classes=sum(1 for x in class_written if x > 0))

    pbar.close()

    nonzero_classes = sum(1 for x in class_written if x > 0)
    return total, nonzero_classes


def main() -> None:
    args = parse_args()

    if args.overwrite and args.out_root.exists():
        shutil.rmtree(args.out_root)

    args.out_root.mkdir(parents=True, exist_ok=True)

    # Get class name mapping from train split.
    train_ds = load_dataset(
        args.dataset_id,
        split="train",
        cache_dir=str(args.cache_dir),
    )
    label_names = train_ds.features["label"].names
    mapping = {f"c{i:04d}": name for i, name in enumerate(label_names)}
    (args.out_root / "class_names.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    splits = ["train", "validation"]
    if args.include_test:
        splits.append("test")

    for split in splits:
        print(f"Exporting split: {split}")
        total, used_classes = export_split(args, split)
        print(f"Done {split}: images={total}, classes={used_classes}")

    print("\nAll done.")
    print(f"Output: {args.out_root}")
    print("Use this as --data-root for LSQ training.")


if __name__ == "__main__":
    main()
