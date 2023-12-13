from typing import Optional

import numpy as np
import torch
from analogvnn.nn.activation.Activation import Activation
from torch import nn, Tensor


class ReGLUGeGLUInterpolation(Activation):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def __init__(self, interpolate_factor: float, scaling_factor: float, alpha: float = 0):
        super(ReGLUGeGLUInterpolation, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.interpolate_factor = nn.Parameter(torch.tensor(interpolate_factor), requires_grad=False)
        self.scaling_factor = nn.Parameter(torch.tensor(scaling_factor), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        relu = torch.maximum(b * self.alpha, b)
        gelu = 0.5 * b * (1 + torch.erf(b * self.scaling_factor / np.sqrt(2)))
        result = relu + self.interpolate_factor * (gelu - relu)
        return a * result

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        x = self.inputs
        a, b = x.chunk(2, dim=-1)
        relu = torch.maximum(b * self.alpha, b)
        gelu = 0.5 * b * (1 + torch.erf(b * self.scaling_factor / np.sqrt(2)))

        grad_gelu = (
                0.5 * (1 + torch.erf(b * self.scaling_factor / np.sqrt(2)))
                + np.sqrt(1 / (2 * np.pi)) * b * self.scaling_factor
                * torch.exp(-torch.pow(b * self.scaling_factor, 2) / 2)
        )
        grad_relu = (b < 0).type(torch.float) * self.alpha + (b >= 0).type(torch.float)
        grad_b = a * (grad_relu + self.interpolate_factor * (grad_gelu - grad_relu))
        grad_a = relu + self.interpolate_factor * (gelu - relu)
        grad = torch.cat([grad_a, grad_b], dim=-1)
        return grad_output * grad

    def extra_repr(self) -> str:
        return f"interpolate_factor={self.interpolate_factor}, scaling_factor={self.scaling_factor}"
