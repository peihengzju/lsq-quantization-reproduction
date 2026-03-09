# LSQ Reproduction on ImageNet (Pre-Activation ResNet-18)

Implementation-focused reproduction of **Learned Step Size Quantization (LSQ)** from arXiv `1902.08153v3`.

- Paper copy in this repo: [`1902.08153v3.pdf`](./1902.08153v3.pdf)
- Core objective: reproduce LSQ training behavior with a clean, reusable PyTorch pipeline
- Positioning: graduate-level research engineering project for portfolio / resume use

## Overview

Quantization-aware training is critical for efficient deployment on constrained hardware. This project reproduces LSQ with paper-aligned defaults and an end-to-end training workflow.

Implemented scope:

- Model: pre-activation ResNet-18
- Two-stage training:
  1. Full-precision baseline training
  2. LSQ quantized fine-tuning from FP checkpoint
- Quantized modules: Conv/Linear weights and input activations
- First/last layer policy: optional higher precision (default 8-bit)
- Optimizer: SGD + momentum (0.9)
- LR schedule: cosine annealing

Paper-aligned defaults (ResNet-18):

- Epoch/LR by bit-width:
  - 2/3/4-bit: 90 epochs, LR=0.01
  - 8-bit: 1 epoch, LR=0.001
- Weight decay (Table 2):
  - 2-bit: `0.25e-4`
  - 3-bit: `0.5e-4`
  - 4/8-bit: `1e-4`

## Repository Structure

- `train_fp.py`: full-precision training
- `train.py`: LSQ quantized fine-tuning
- `eval.py`: evaluation for FP/LSQ checkpoints
- `lsq/quant/lsq.py`: LSQ quantizer (`gradscale`, `roundpass`, `quantize`)
- `lsq/models/preact_resnet.py`: pre-activation ResNet-18 + quantization wrapping
- `lsq/data/imagenet.py`: ImageNet data loading and transforms
- `export_hf_imagenet.py`: export HF ImageNet cache to `ImageFolder`
- `split_dataset.py`: split class-folder dataset into train/val

## Quick Start

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Prepare dataset

Expected layout for `--data-root`:

```text
<data-root>/
  train/
    class_x/xxx.jpg
  val/
    class_x/yyy.jpg
```

Option A (Hugging Face cache export):

```bash
python export_hf_imagenet.py \
  --dataset-id ILSVRC/imagenet-1k \
  --cache-dir ./Dataset \
  --out-root ./data_imagenet1k
```

Option B (split existing class-folder dataset):

```bash
python split_dataset.py \
  --src /path/to/class_folder_dataset \
  --dst ./data \
  --val-ratio 0.2 \
  --seed 42
```

### 3) Train full-precision baseline

```bash
python train_fp.py \
  --data-root data_imagenet1k \
  --output-dir runs/fp_preact18
```

### 4) LSQ fine-tuning (example: W4A4)

```bash
python train.py \
  --data-root data_imagenet1k \
  --fp-ckpt runs/fp_preact18/best.pth \
  --output-dir runs/lsq4_preact18 \
  --w-bits 4 \
  --a-bits 4 \
  --first-last-bits 8
```

### 5) Evaluate checkpoint

```bash
python eval.py \
  --data-root data_imagenet1k \
  --ckpt runs/lsq4_preact18/best.pth \
  --w-bits 4 \
  --a-bits 4
```

## Engineering Highlights

- Reproduced LSQ quantization mechanics from paper pseudocode in modular PyTorch components
- Encoded paper-specific training defaults directly in CLI-driven training scripts
- Built a reproducible train/eval pipeline suitable for rapid experiment iteration
- Added practical dataset tooling for real-world workflow setup

## Notes

- This repository emphasizes implementation fidelity and workflow reproducibility.
- Final metrics can vary by hardware, data preprocessing details, and training budget.
