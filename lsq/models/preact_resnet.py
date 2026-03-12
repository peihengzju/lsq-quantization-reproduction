from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn

from lsq.quant.lsq import QuantConv2d, QuantLinear

LayerPolicy = Dict[str, Union[int, bool, str]]


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        identity = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out += identity
        return out


class PreActResNet(nn.Module):
    def __init__(
        self,
        block: Type[PreActBasicBlock],
        layers: List[int],
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: Type[PreActBasicBlock],
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        modules = []
        for s in strides:
            modules.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*modules)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@dataclass
class LSQConfig:
    w_bits: int = 4
    a_bits: int = 4
    first_last_bits: int = 8
    quantize_first_last_8bit: bool = True
    signed_input_first_layer: bool = False
    w_grad_scale_mode: str = "lsq"
    a_grad_scale_mode: str = "lsq"
    w_grad_scale_factor: float = 1.0
    a_grad_scale_factor: float = 1.0


def preact_resnet18(num_classes: int = 1000) -> PreActResNet:
    return PreActResNet(PreActBasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def _replace_module(parent: nn.Module, name: str, module: nn.Module) -> None:
    setattr(parent, name, module)


def _resolve_bits(module_name: str, cfg: LSQConfig) -> Tuple[int, int]:
    if cfg.quantize_first_last_8bit and module_name in ("conv1", "fc"):
        return cfg.first_last_bits, cfg.first_last_bits
    return cfg.w_bits, cfg.a_bits


def apply_lsq_quantization(model: nn.Module, cfg: LSQConfig) -> List[LayerPolicy]:
    layer_policy: List[LayerPolicy] = []

    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue

        parts = module_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]

        if isinstance(module, nn.Conv2d):
            w_bits, a_bits = _resolve_bits(module_name, cfg)
            a_signed = module_name == "conv1" and cfg.signed_input_first_layer
            wrapped = QuantConv2d(
                module,
                w_bits=w_bits,
                a_bits=a_bits,
                a_signed=a_signed,
                w_grad_scale_mode=cfg.w_grad_scale_mode,
                a_grad_scale_mode=cfg.a_grad_scale_mode,
                w_grad_scale_factor=cfg.w_grad_scale_factor,
                a_grad_scale_factor=cfg.a_grad_scale_factor,
                quantize_input=True,
            )
            _replace_module(parent, child_name, wrapped)
            layer_policy.append(
                {
                    "name": module_name,
                    "type": "Conv2d",
                    "w_bits": w_bits,
                    "a_bits": a_bits,
                    "a_signed": a_signed,
                    "w_g_mode": cfg.w_grad_scale_mode,
                    "a_g_mode": cfg.a_grad_scale_mode,
                    "quantize_input": True,
                }
            )

        elif isinstance(module, nn.Linear):
            w_bits, a_bits = _resolve_bits(module_name, cfg)
            wrapped = QuantLinear(
                module,
                w_bits=w_bits,
                a_bits=a_bits,
                a_signed=False,
                w_grad_scale_mode=cfg.w_grad_scale_mode,
                a_grad_scale_mode=cfg.a_grad_scale_mode,
                w_grad_scale_factor=cfg.w_grad_scale_factor,
                a_grad_scale_factor=cfg.a_grad_scale_factor,
                quantize_input=True,
            )
            _replace_module(parent, child_name, wrapped)
            layer_policy.append(
                {
                    "name": module_name,
                    "type": "Linear",
                    "w_bits": w_bits,
                    "a_bits": a_bits,
                    "a_signed": False,
                    "w_g_mode": cfg.w_grad_scale_mode,
                    "a_g_mode": cfg.a_grad_scale_mode,
                    "quantize_input": True,
                }
            )

    return layer_policy
