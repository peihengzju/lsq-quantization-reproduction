from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def resolve_split_dirs(data_root: str) -> tuple[Path, Path]:
    root = Path(data_root)
    train_dir = root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train directory not found: {train_dir}")

    val_dir = root / "val"
    if not val_dir.is_dir():
        alt = root / "validation"
        if alt.is_dir():
            val_dir = alt
        else:
            raise FileNotFoundError(f"val directory not found: {val_dir} (or {alt})")

    return train_dir, val_dir


def infer_num_classes(data_root: str) -> int:
    train_dir, _ = resolve_split_dirs(data_root)
    classes = [p for p in train_dir.iterdir() if p.is_dir()]
    if not classes:
        raise RuntimeError(f"No class directories found under: {train_dir}")
    return len(classes)


def build_imagenet_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
):
    train_dir, val_dir = resolve_split_dirs(data_root)

    train_tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    val_tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    train_set = datasets.ImageFolder(train_dir, transform=train_tfm)
    val_set = datasets.ImageFolder(val_dir, transform=val_tfm)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
