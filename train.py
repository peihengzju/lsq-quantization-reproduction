from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim

from lsq.data import build_imagenet_loaders, infer_num_classes
from lsq.engine import run_training
from lsq.models import LSQConfig, apply_lsq_quantization, preact_resnet18


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
    p = argparse.ArgumentParser("LSQ quantized fine-tuning (paper-aligned)")
    p.add_argument(
        "--data-root",
        type=str,
        default="data_imagenet1k",
        help="ImageNet root containing train/ and val/ (default: data_imagenet1k)",
    )
    p.add_argument("--fp-ckpt", type=str, required=True, help="Full precision checkpoint path")
    p.add_argument("--output-dir", type=str, default="runs/lsq_preact18")
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path. Default: <output-dir>/train.log",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--w-bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--a-bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--first-last-bits", type=int, default=8)
    p.add_argument("--disable-first-last-8bit", action="store_true")
    p.add_argument(
        "--signed-input-first-layer",
        action="store_true",
        help="Quantize first-layer input with signed activation range",
    )
    p.add_argument(
        "--w-grad-scale-mode",
        type=str,
        default="lsq",
        choices=["lsq", "n", "none"],
        help="Weight step-size gradient scale mode: lsq=1/sqrt(N*Qp), n=1/sqrt(N), none=1",
    )
    p.add_argument(
        "--a-grad-scale-mode",
        type=str,
        default="lsq",
        choices=["lsq", "n", "none"],
        help="Activation step-size gradient scale mode: lsq=1/sqrt(N*Qp), n=1/sqrt(N), none=1",
    )
    p.add_argument(
        "--w-grad-scale-factor",
        type=float,
        default=1.0,
        help="Additional multiplier for weight step-size gradient scale",
    )
    p.add_argument(
        "--a-grad-scale-factor",
        type=float,
        default=1.0,
        help="Additional multiplier for activation step-size gradient scale",
    )

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--momentum", type=float, default=0.9)

    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def default_epochs_for_bits(bits: int) -> int:
    return 1 if bits == 8 else 90


def default_lr_for_bits(bits: int) -> float:
    return 0.001 if bits == 8 else 0.01


def default_weight_decay_for_bits(bits: int) -> float:
    # Table 2 in LSQ paper (ResNet-18):
    # 2-bit: 0.25e-4, 3-bit: 0.5e-4, 4/8-bit: 1e-4
    if bits == 2:
        return 0.25e-4
    if bits == 3:
        return 0.5e-4
    return 1e-4


def load_fp_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


def build_checkpoint_meta(
    args: argparse.Namespace,
    *,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    layer_policy: list[dict],
) -> dict:
    return {
        "repo": "LSQ reproduction",
        "model_name": "preact_resnet18",
        "checkpoint_type": "lsq",
        "data": {
            "data_root": args.data_root,
            "num_classes": num_classes,
            "train_transform": ["Resize(256)", "RandomCrop(224)", "RandomHorizontalFlip(0.5)", "ToTensor()"],
            "val_transform": ["Resize(256)", "CenterCrop(224)", "ToTensor()"],
            "normalize": None,
        },
        "quantization": {
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "first_last_bits": args.first_last_bits,
            "quantize_first_last_8bit": not args.disable_first_last_8bit,
            "signed_input_first_layer": args.signed_input_first_layer,
            "w_grad_scale_mode": args.w_grad_scale_mode,
            "a_grad_scale_mode": args.a_grad_scale_mode,
            "w_grad_scale_factor": args.w_grad_scale_factor,
            "a_grad_scale_factor": args.a_grad_scale_factor,
            "layer_policy": layer_policy,
        },
        "train": {
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "fp_ckpt": args.fp_ckpt,
        },
    }


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

    epochs = args.epochs if args.epochs is not None else default_epochs_for_bits(args.w_bits)
    lr = args.lr if args.lr is not None else default_lr_for_bits(args.w_bits)
    wd = args.weight_decay if args.weight_decay is not None else default_weight_decay_for_bits(args.w_bits)
    num_classes = args.num_classes if args.num_classes is not None else infer_num_classes(args.data_root)

    print(f"Using num_classes: {num_classes}")
    model = preact_resnet18(num_classes=num_classes)
    load_fp_checkpoint(model, args.fp_ckpt)

    qcfg = LSQConfig(
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        first_last_bits=args.first_last_bits,
        quantize_first_last_8bit=not args.disable_first_last_8bit,
        signed_input_first_layer=args.signed_input_first_layer,
        w_grad_scale_mode=args.w_grad_scale_mode,
        a_grad_scale_mode=args.a_grad_scale_mode,
        w_grad_scale_factor=args.w_grad_scale_factor,
        a_grad_scale_factor=args.a_grad_scale_factor,
    )
    layer_policy = apply_lsq_quantization(model, qcfg)
    model = model.to(device)

    print(f"Training config: epochs={epochs}, lr={lr}, weight_decay={wd}, momentum={args.momentum}")
    print(
        "Grad-scale config: "
        f"w_mode={args.w_grad_scale_mode}, a_mode={args.a_grad_scale_mode}, "
        f"w_factor={args.w_grad_scale_factor}, a_factor={args.a_grad_scale_factor}"
    )
    print("Layer quantization policy:")
    for item in layer_policy:
        print(
            f"  {item['name']:<24} {item['type']:<7} "
            f"W{item['w_bits']} A{item['a_bits']} "
            f"a_signed={item['a_signed']}"
        )

    train_loader, val_loader = build_imagenet_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=wd,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    with open(out_dir / "run_args.txt", "w", encoding="utf-8") as f:
        run_args = vars(args).copy()
        run_args["num_classes"] = num_classes
        run_args["log_file"] = str(log_path)
        f.write(str(run_args))

    checkpoint_meta = build_checkpoint_meta(
        args,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        weight_decay=wd,
        layer_policy=layer_policy,
    )

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        out_dir=args.output_dir,
        checkpoint_meta=checkpoint_meta,
    )


if __name__ == "__main__":
    main()
