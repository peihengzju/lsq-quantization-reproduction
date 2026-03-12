from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class TrainStats:
    loss: float
    top1: float
    top5: float


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    batch_size = target.size(0)
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k * (100.0 / batch_size)).item())
    return res


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        top1, top5 = accuracy(outputs.detach(), targets, topk=(1, 5))
        running_loss += loss.item()
        running_top1 += top1
        running_top5 += top5

    n = len(loader)
    return TrainStats(
        loss=running_loss / n,
        top1=running_top1 / n,
        top5=running_top5 / n,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        top1, top5 = accuracy(outputs, targets, topk=(1, 5))

        running_loss += loss.item()
        running_top1 += top1
        running_top5 += top5

    n = len(loader)
    return TrainStats(
        loss=running_loss / n,
        top1=running_top1 / n,
        top5=running_top5 / n,
    )


def save_checkpoint(state: dict, out_dir: str, filename: str = "last.pth"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state, out / filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs: int,
    device,
    out_dir: str,
    checkpoint_meta: dict | None = None,
):
    criterion = nn.CrossEntropyLoss()
    best_top1 = 0.0
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_path / "metrics.csv"

    with open(metrics_csv, "w", encoding="utf-8", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_top1",
                "train_top5",
                "val_loss",
                "val_top1",
                "val_top5",
                "best_top1",
                "epoch_time_sec",
            ]
        )

        for epoch in range(epochs):
            t0 = time.time()
            train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_stats = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_stats.top1 > best_top1:
                best_top1 = val_stats.top1
                best_state = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_top1": best_top1,
                }
                if checkpoint_meta is not None:
                    best_state["meta"] = checkpoint_meta
                save_checkpoint(
                    best_state,
                    out_dir,
                    filename="best.pth",
                )

            last_state = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_top1": best_top1,
            }
            if checkpoint_meta is not None:
                last_state["meta"] = checkpoint_meta
            save_checkpoint(
                last_state,
                out_dir,
                filename="last.pth",
            )

            dt = time.time() - t0
            writer.writerow(
                [
                    epoch + 1,
                    f"{train_stats.loss:.6f}",
                    f"{train_stats.top1:.4f}",
                    f"{train_stats.top5:.4f}",
                    f"{val_stats.loss:.6f}",
                    f"{val_stats.top1:.4f}",
                    f"{val_stats.top5:.4f}",
                    f"{best_top1:.4f}",
                    f"{dt:.4f}",
                ]
            )
            mf.flush()
            print(
                f"[Epoch {epoch + 1:03d}/{epochs:03d}] "
                f"train_loss={train_stats.loss:.4f} train_top1={train_stats.top1:.2f} "
                f"val_loss={val_stats.loss:.4f} val_top1={val_stats.top1:.2f} "
                f"val_top5={val_stats.top5:.2f} time={dt:.1f}s"
            )

    print(f"Finished training. Best val@1 = {best_top1:.2f}")
