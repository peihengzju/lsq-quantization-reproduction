from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from lsq.data import build_imagenet_loaders, infer_num_classes
from lsq.engine import run_training
from lsq.models import preact_resnet18


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _enable_file_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def parse_args():
    p = argparse.ArgumentParser("Full precision pre-activation ResNet-18 training (paper-aligned)")
    p.add_argument("--data-root", type=str, default="data_imagenet1k")
    p.add_argument("--output-dir", type=str, default="runs/fp_preact18")
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path. Default: <output-dir>/train.log",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)

    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else out_dir / "train.log"
    _enable_file_logging(log_path)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = args.num_classes if args.num_classes is not None else infer_num_classes(args.data_root)
    print(f"Using num_classes: {num_classes}")

    model = preact_resnet18(num_classes=num_classes).to(device)

    train_loader, val_loader = build_imagenet_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    with open(out_dir / "run_args.txt", "w", encoding="utf-8") as f:
        run_args = vars(args).copy()
        run_args["num_classes"] = num_classes
        run_args["log_file"] = str(log_path)
        f.write(str(run_args))

    checkpoint_meta = {
        "repo": "LSQ reproduction",
        "model_name": "preact_resnet18",
        "checkpoint_type": "fp",
        "data": {
            "data_root": args.data_root,
            "num_classes": num_classes,
            "train_transform": ["Resize(256)", "RandomCrop(224)", "RandomHorizontalFlip(0.5)", "ToTensor()"],
            "val_transform": ["Resize(256)", "CenterCrop(224)", "ToTensor()"],
            "normalize": None,
        },
        "train": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
        },
    }

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        out_dir=args.output_dir,
        checkpoint_meta=checkpoint_meta,
    )


if __name__ == "__main__":
    main()
