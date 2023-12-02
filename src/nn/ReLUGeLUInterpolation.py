from typing import Optional

import numpy as np
import torch
from analogvnn.nn.activation.Activation import Activation
from torch import nn, Tensor


class ReLUGeLUInterpolation(Activation):
    def __init__(self, interpolate_factor: float, scaling_factor: float, alpha: float = 0):
        super(ReLUGeLUInterpolation, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.interpolate_factor = nn.Parameter(torch.tensor(interpolate_factor), requires_grad=False)
        self.scaling_factor = nn.Parameter(torch.tensor(scaling_factor), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        relu = torch.maximum(x * self.alpha, x)
        gelu = 0.5 * x * (1 + torch.erf(x * self.scaling_factor / np.sqrt(2)))
        return relu + self.interpolate_factor * (gelu - relu)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        x = self.inputs
        grad_gelu = (
                0.5 * (1 + torch.erf(x * self.scaling_factor / np.sqrt(2)))
                + np.sqrt(1 / (2 * np.pi)) * x * self.scaling_factor
                * torch.exp(-torch.pow(x * self.scaling_factor, 2) / 2)
        )

        grad_relu = (x < 0).type(torch.float) * self.alpha + (x >= 0).type(torch.float)
        grad = grad_relu + self.interpolate_factor * (grad_gelu - grad_relu)
        return grad_output * grad

    def extra_repr(self) -> str:
        return f"interpolate_factor={self.interpolate_factor}, scaling_factor={self.scaling_factor}"
