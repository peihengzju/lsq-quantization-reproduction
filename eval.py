from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from lsq.data import build_imagenet_loaders, infer_num_classes
from lsq.engine.trainer import evaluate
from lsq.models import LSQConfig, apply_lsq_quantization, preact_resnet18


def parse_args():
    p = argparse.ArgumentParser("Evaluate LSQ pre-activation ResNet-18")
    p.add_argument("--data-root", type=str, default="data_imagenet1k")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--w-bits", type=int, default=4)
    p.add_argument("--a-bits", type=int, default=4)
    p.add_argument("--first-last-bits", type=int, default=8)
    p.add_argument("--disable-first-last-8bit", action="store_true")
    p.add_argument("--signed-input-first-layer", action="store_true")
    p.add_argument("--w-grad-scale-mode", type=str, default="lsq", choices=["lsq", "n", "none"])
    p.add_argument("--a-grad-scale-mode", type=str, default="lsq", choices=["lsq", "n", "none"])
    p.add_argument("--w-grad-scale-factor", type=float, default=1.0)
    p.add_argument("--a-grad-scale-factor", type=float, default=1.0)
    p.add_argument("--num-classes", type=int, default=None)
    return p.parse_args()


def is_lsq_checkpoint(state: dict[str, torch.Tensor]) -> bool:
    return any(
        ".w_quant." in k or ".a_quant." in k or k.endswith(".conv.weight")
        for k in state.keys()
    )


def resolve_lsq_config(args, ckpt: dict) -> LSQConfig:
    meta = ckpt.get("meta", {})
    quant_meta = meta.get("quantization", {})
    if quant_meta:
        print("Using quantization config from checkpoint metadata")
    else:
        print("Checkpoint metadata missing quantization config, falling back to CLI args")

    return LSQConfig(
        w_bits=int(quant_meta.get("w_bits", args.w_bits)),
        a_bits=int(quant_meta.get("a_bits", args.a_bits)),
        first_last_bits=int(quant_meta.get("first_last_bits", args.first_last_bits)),
        quantize_first_last_8bit=bool(
            quant_meta.get("quantize_first_last_8bit", not args.disable_first_last_8bit)
        ),
        signed_input_first_layer=bool(
            quant_meta.get("signed_input_first_layer", args.signed_input_first_layer)
        ),
        w_grad_scale_mode=quant_meta.get("w_grad_scale_mode", args.w_grad_scale_mode),
        a_grad_scale_mode=quant_meta.get("a_grad_scale_mode", args.a_grad_scale_mode),
        w_grad_scale_factor=float(quant_meta.get("w_grad_scale_factor", args.w_grad_scale_factor)),
        a_grad_scale_factor=float(quant_meta.get("a_grad_scale_factor", args.a_grad_scale_factor)),
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    is_lsq = is_lsq_checkpoint(state)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    if args.num_classes is not None:
        num_classes = args.num_classes
    elif meta.get("data", {}).get("num_classes") is not None:
        num_classes = int(meta["data"]["num_classes"])
    elif "fc.weight" in state:
        num_classes = int(state["fc.weight"].shape[0])
    elif "fc.linear.weight" in state:
        num_classes = int(state["fc.linear.weight"].shape[0])
    else:
        num_classes = infer_num_classes(args.data_root)

    print(f"Using num_classes: {num_classes}")
    print(f"Checkpoint type: {'LSQ quantized' if is_lsq else 'FP'}")
    data_meta = meta.get("data", {})
    if data_meta:
        print(
            "Data preprocessing from checkpoint metadata: "
            f"train={data_meta.get('train_transform')} val={data_meta.get('val_transform')} "
            f"normalize={data_meta.get('normalize')}"
        )
    model = preact_resnet18(num_classes=num_classes)

    if is_lsq:
        cfg = resolve_lsq_config(args, ckpt if isinstance(ckpt, dict) else {})
        apply_lsq_quantization(model, cfg)

    model.load_state_dict(state, strict=True)
    model = model.to(device)

    _, val_loader = build_imagenet_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss()
    stats = evaluate(model, val_loader, criterion, device)
    print(f"val_loss={stats.loss:.4f} val_top1={stats.top1:.2f} val_top5={stats.top5:.2f}")


if __name__ == "__main__":
    main()
