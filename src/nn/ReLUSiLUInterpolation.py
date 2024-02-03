from typing import Optional

import torch
from analogvnn.nn.activation.Activation import Activation
from torch import nn, Tensor


class ReLUSiLUInterpolation(Activation):
    def __init__(self, interpolate_factor: float, alpha: float = 0):
        super(ReLUSiLUInterpolation, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.interpolate_factor = nn.Parameter(torch.tensor(interpolate_factor), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        relu = torch.maximum(x * self.alpha, x)
        silu = x / (1 + torch.exp(-x))
        return relu + self.interpolate_factor * (silu - relu)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        return grad_output * self.backward_inputs(self.inputs)

    def backward_inputs(self, x):
        e_x = torch.exp(x)
        grad_silu = (e_x * (1 + x + e_x)) / torch.pow(1 + e_x, torch.tensor(2))
        grad_relu = (x < 0).type(torch.float) * self.alpha + (x >= 0).type(torch.float)
        grad = grad_relu + self.interpolate_factor * (grad_silu - grad_relu)
        return grad

    def extra_repr(self) -> str:
        return f"interpolate_factor={self.interpolate_factor}, scaling_factor={self.scaling_factor}"
