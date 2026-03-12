import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x: torch.Tensor) -> torch.Tensor:
    y = torch.round(x)
    return y.detach() - x.detach() + x


def nweights(v: torch.Tensor) -> int:
    return max(v.numel(), 1)


def nfeatures(v: torch.Tensor) -> int:
    if v.dim() <= 1:
        return max(v.numel(), 1)
    # Follow the paper's activation-layer heuristic by excluding batch count.
    return max(v[0].numel(), 1)


class LSQQuantizer(nn.Module):
    """Learned Step Size Quantization (per-tensor), aligned with LSQ appendix pseudocode."""

    def __init__(
        self,
        n_bits: int,
        is_activation: bool,
        signed: Optional[bool] = None,
        grad_scale_mode: Literal["lsq", "n", "none"] = "lsq",
        grad_scale_factor: float = 1.0,
        init_on_first_batch: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        assert 2 <= n_bits <= 8, "n_bits should be in [2, 8]"
        self.n_bits = n_bits
        self.is_activation = is_activation
        self.grad_scale_mode = grad_scale_mode
        self.grad_scale_factor = float(grad_scale_factor)
        self.init_on_first_batch = init_on_first_batch
        self.eps = eps

        if signed is None:
            signed = not is_activation
        self.signed = signed

        if signed:
            self.qn = -(2 ** (n_bits - 1))
            self.qp = 2 ** (n_bits - 1) - 1
        else:
            self.qn = 0
            self.qp = 2**n_bits - 1

        self.s = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def _init_step_size(self, v: torch.Tensor) -> None:
        mean_abs = v.detach().abs().mean()
        init = 2.0 * mean_abs / math.sqrt(self.qp)
        init = torch.clamp(init, min=self.eps)
        self.s.copy_(init)
        self.initialized.fill_(True)

    def _compute_grad_scale(self, v: torch.Tensor) -> float:
        # LSQ uses 1/sqrt(NW * Qp) for weights and 1/sqrt(NF * Qp) for activations.
        n = nfeatures(v) if self.is_activation else nweights(v)
        if self.grad_scale_mode == "none":
            base = 1.0
        elif self.grad_scale_mode == "n":
            base = 1.0 / math.sqrt(n)
        elif self.grad_scale_mode == "lsq":
            base = 1.0 / math.sqrt(n * self.qp)
        else:
            raise ValueError(f"Unsupported grad_scale_mode: {self.grad_scale_mode}")
        return base * self.grad_scale_factor

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if self.training and self.init_on_first_batch and not bool(self.initialized):
            self._init_step_size(v)

        s = torch.clamp(self.s, min=self.eps)
        g = self._compute_grad_scale(v)
        s_scaled = grad_scale(s, g)

        v_bar = v / s_scaled
        v_hat = torch.clamp(v_bar, self.qn, self.qp)
        v_tilde = round_pass(v_hat)
        return v_tilde * s_scaled

    def extra_repr(self) -> str:
        mode = "act" if self.is_activation else "weight"
        sign = "signed" if self.signed else "unsigned"
        return (
            f"mode={mode}, bits={self.n_bits}, {sign}, qn={self.qn}, qp={self.qp}, "
            f"g_mode={self.grad_scale_mode}, g_factor={self.grad_scale_factor:g}"
        )


class QuantConv2d(nn.Module):
    """Conv2d with LSQ quantized weights and input activations."""

    def __init__(
        self,
        conv: nn.Conv2d,
        w_bits: int,
        a_bits: int,
        a_signed: bool = False,
        w_grad_scale_mode: Literal["lsq", "n", "none"] = "lsq",
        a_grad_scale_mode: Literal["lsq", "n", "none"] = "lsq",
        w_grad_scale_factor: float = 1.0,
        a_grad_scale_factor: float = 1.0,
        quantize_input: bool = True,
    ) -> None:
        super().__init__()
        self.conv = conv
        self.w_quant = LSQQuantizer(
            n_bits=w_bits,
            is_activation=False,
            signed=True,
            grad_scale_mode=w_grad_scale_mode,
            grad_scale_factor=w_grad_scale_factor,
        )
        self.a_quant = LSQQuantizer(
            n_bits=a_bits,
            is_activation=True,
            signed=a_signed,
            grad_scale_mode=a_grad_scale_mode,
            grad_scale_factor=a_grad_scale_factor,
        )
        self.quantize_input = quantize_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_input:
            x = self.a_quant(x)
        qw = self.w_quant(self.conv.weight)
        return nn.functional.conv2d(
            x,
            qw,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )


class QuantLinear(nn.Module):
    """Linear layer with LSQ quantized weights and input activations."""

    def __init__(
        self,
        linear: nn.Linear,
        w_bits: int,
        a_bits: int,
        a_signed: bool = False,
        w_grad_scale_mode: Literal["lsq", "n", "none"] = "lsq",
        a_grad_scale_mode: Literal["lsq", "n", "none"] = "lsq",
        w_grad_scale_factor: float = 1.0,
        a_grad_scale_factor: float = 1.0,
        quantize_input: bool = True,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.w_quant = LSQQuantizer(
            n_bits=w_bits,
            is_activation=False,
            signed=True,
            grad_scale_mode=w_grad_scale_mode,
            grad_scale_factor=w_grad_scale_factor,
        )
        self.a_quant = LSQQuantizer(
            n_bits=a_bits,
            is_activation=True,
            signed=a_signed,
            grad_scale_mode=a_grad_scale_mode,
            grad_scale_factor=a_grad_scale_factor,
        )
        self.quantize_input = quantize_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_input:
            x = self.a_quant(x)
        qw = self.w_quant(self.linear.weight)
        return nn.functional.linear(x, qw, self.linear.bias)


def quant_range(
    n_bits: int,
    is_activation: bool,
    signed: Optional[bool] = None,
) -> Tuple[int, int]:
    if signed is None:
        signed = not is_activation
    if not signed:
        return 0, 2**n_bits - 1
    return -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
