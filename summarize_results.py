#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize LSQ training runs under a runs directory.")
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory containing run subdirectories (default: runs)",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save summary CSV. Default: <runs-root>/summary.csv",
    )
    p.add_argument(
        "--sort-by",
        type=str,
        default="best_top1",
        choices=["best_top1", "last_val_top1", "name"],
        help="Sort key for output rows (default: best_top1)",
    )
    p.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default is descending for numeric keys)",
    )
    return p.parse_args()


def _read_run_args(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return ast.literal_eval(raw) if raw else {}
    except Exception:
        return {}


def _read_metrics(path: Path) -> tuple[str, float | None, float | None]:
    if not path.exists():
        return "", None, None
    last = None
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    if last is None:
        return "", None, None
    epoch = str(last.get("epoch", ""))
    try:
        val_top1 = float(last.get("val_top1", ""))
    except Exception:
        val_top1 = None
    try:
        val_top5 = float(last.get("val_top5", ""))
    except Exception:
        val_top5 = None
    return epoch, val_top1, val_top5


def _read_best_top1_from_metrics(path: Path) -> float | None:
    if not path.exists():
        return None
    best = None
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("best_top1", "")
            try:
                v = float(raw)
            except Exception:
                continue
            if best is None or v > best:
                best = v
    return best


def _read_best_top1_from_ckpt(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        import torch  # local import so script still works without torch installed
    except Exception:
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if isinstance(ckpt, dict):
        v = ckpt.get("best_top1")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _read_from_train_log(path: Path) -> tuple[float | None, float | None, float | None]:
    if not path.exists():
        return None, None, None

    best_top1 = None
    last_val_top1 = None
    last_val_top5 = None
    best_pat = re.compile(r"Finished training\. Best val@1 = ([0-9]+(?:\.[0-9]+)?)")
    last_pat = re.compile(r"val_top1=([0-9]+(?:\.[0-9]+)?)\s+val_top5=([0-9]+(?:\.[0-9]+)?)")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = best_pat.search(line)
            if m:
                try:
                    best_top1 = float(m.group(1))
                except Exception:
                    pass
            m2 = last_pat.search(line)
            if m2:
                try:
                    last_val_top1 = float(m2.group(1))
                    last_val_top5 = float(m2.group(2))
                except Exception:
                    pass

    return best_top1, last_val_top1, last_val_top5


def _fmt_float(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _collect_row(run_dir: Path) -> dict[str, Any] | None:
    if not run_dir.is_dir():
        return None

    run_args = _read_run_args(run_dir / "run_args.txt")
    last_epoch, last_val_top1, last_val_top5 = _read_metrics(run_dir / "metrics.csv")
    best_top1 = _read_best_top1_from_metrics(run_dir / "metrics.csv")
    if best_top1 is None:
        best_top1 = _read_best_top1_from_ckpt(run_dir / "best.pth")
    if best_top1 is None or last_val_top1 is None or last_val_top5 is None:
        log_best, log_last_top1, log_last_top5 = _read_from_train_log(run_dir / "train.log")
        if best_top1 is None:
            best_top1 = log_best
        if last_val_top1 is None:
            last_val_top1 = log_last_top1
        if last_val_top5 is None:
            last_val_top5 = log_last_top5

    is_quant = "fp_ckpt" in run_args
    if is_quant:
        mode = "lsq"
        bits = f"W{run_args.get('w_bits', '?')}A{run_args.get('a_bits', '?')}"
    elif run_args:
        mode = "fp"
        bits = "FP32"
    else:
        # Not a training run directory
        return None

    row = {
        "name": run_dir.name,
        "mode": mode,
        "bits": bits,
        "epochs": run_args.get("epochs", ""),
        "lr": run_args.get("lr", ""),
        "weight_decay": run_args.get("weight_decay", ""),
        "batch_size": run_args.get("batch_size", ""),
        "best_top1": best_top1,
        "last_epoch": last_epoch,
        "last_val_top1": last_val_top1,
        "last_val_top5": last_val_top5,
        "path": str(run_dir),
    }
    return row


def _sort_rows(rows: list[dict[str, Any]], key: str, ascending: bool) -> list[dict[str, Any]]:
    if key == "name":
        return sorted(rows, key=lambda r: r["name"], reverse=not ascending)

    def val(r: dict[str, Any]) -> float:
        x = r.get(key)
        return float("-inf") if x is None else float(x)

    return sorted(rows, key=val, reverse=not ascending)


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "name",
        "mode",
        "bits",
        "epochs",
        "best_top1",
        "last_val_top1",
        "last_val_top5",
    ]
    lines = [headers]
    for r in rows:
        lines.append(
            [
                str(r["name"]),
                str(r["mode"]),
                str(r["bits"]),
                str(r["epochs"]),
                _fmt_float(r["best_top1"]),
                _fmt_float(r["last_val_top1"]),
                _fmt_float(r["last_val_top5"]),
            ]
        )

    widths = [max(len(row[i]) for row in lines) for i in range(len(headers))]
    for idx, row in enumerate(lines):
        text = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(text)
        if idx == 0:
            print("  ".join("-" * w for w in widths))


def _write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "mode",
        "bits",
        "epochs",
        "lr",
        "weight_decay",
        "batch_size",
        "best_top1",
        "last_epoch",
        "last_val_top1",
        "last_val_top5",
        "path",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    args = parse_args()
    if not args.runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {args.runs_root}")

    rows: list[dict[str, Any]] = []
    for p in sorted(args.runs_root.iterdir()):
        row = _collect_row(p)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"No valid run directories found under: {args.runs_root}")
        return

    rows = _sort_rows(rows, args.sort_by, args.ascending)
    _print_table(rows)

    out_csv = args.output_csv if args.output_csv is not None else (args.runs_root / "summary.csv")
    _write_csv(rows, out_csv)
    print(f"\nSaved summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
