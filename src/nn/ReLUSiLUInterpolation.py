from typing import Optional

import torch
from analogvnn.nn.activation.Activation import Activation
from torch import nn, Tensor


class ReLUSiLUInterpolation(Activation):
    def __init__(self, interpolate_factor: float, scaling_factor: float):
        super(ReLUSiLUInterpolation, self).__init__()
        self._zero = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.interpolate_factor = nn.Parameter(torch.tensor(interpolate_factor), requires_grad=False)
        self.scaling_factor = nn.Parameter(torch.tensor(scaling_factor), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        relu = torch.maximum(self._zero, x)
        silu = x / (1 + torch.exp(-x))
        return relu + self.interpolate_factor * (silu - relu)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        x = self.inputs
        e_x = torch.exp(x)
        grad_silu = (e_x * (1 + x + e_x)) / torch.pow(1 + e_x, torch.tensor(2))
        grad_relu = (x >= 0).type(torch.float)
        grad = grad_relu + self.interpolate_factor * (grad_silu - grad_relu)
        return grad_output * grad

    def extra_repr(self) -> str:
        return f"interpolate_factor={self.interpolate_factor}, scaling_factor={self.scaling_factor}"
