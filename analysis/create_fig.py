import numpy as np
import torch
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import Clamp
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from matplotlib import pyplot as plt

from src.nn.ReLUGeLUInterpolation import ReLUGeLUInterpolation
from src.nn.ReLUSiLUInterpolation import ReLUSiLUInterpolation


def figure2():
    fig, ax = plt.subplots(2, 5, figsize=(2 * 6.50, 4 * 2 * 3 / 4 * 2 / 3), sharex=True, sharey=True)

    for i in range(5):
        x = torch.linspace(-3, 3, 1000)
        s = np.linspace(1, 3, 5)[i]
        regu = ReLUGeLUInterpolation(interpolate_factor=1, scaling_factor=s)
        ax[0][i].set_title(f"GELU s={s:.2f}")
        ax[0][i].plot(x, regu(x), label="f(x)")
        ax[0][i].plot(x, regu.backward_inputs(x), label="f'(x)")
        ax[0][i].legend()
        ax[0][i].set_ylim(-0.25, 1.5)
        ax[0][i].set_xlim(-1.5, 1.5)

    for i in range(5):
        x = torch.linspace(-3, 3, 1000)
        x_prime = 1 / 6 * torch.sign(x) * torch.ceil(torch.abs(x * 6) - 0.5)
        s = np.linspace(1, 3, 5)[i]
        regu = ReLUGeLUInterpolation(interpolate_factor=1, scaling_factor=s)
        ax[1][i].set_title(f"GELU s={s:.2f}")
        ax[1][i].scatter(x, regu(x_prime), label="f(x)", s=1)
        ax[1][i].scatter(x, regu.backward_inputs(x_prime), label="f'(x)", s=1)
        ax[1][i].legend()
        ax[1][i].set_ylim(-0.25, 1.5)
        ax[1][i].set_xlim(-1.5, 1.5)

    plt.tight_layout()
    plt.show()
    fig.savefig("../_results/Figure 2.png", dpi=600)
    fig.savefig("../_results/Figure 2.svg", dpi=600)
    fig.savefig("../_results/Figure 2.pdf", dpi=600)


def figure3():
    fig, ax = plt.subplots(3, 5, figsize=(2 * 6.50, 4 * 2 * 3 / 4), sharex=True, sharey=True)

    for i in range(5):
        x = torch.linspace(-3, 3, 1000)
        regu = ReLUGeLUInterpolation(interpolate_factor=i / 4, scaling_factor=1)
        suffix = ""
        if i == 0:
            suffix = " (ReLU)"
        elif i == 4:
            suffix = " (GELU)"
        ax[0][i].set_title(f"i={i / 4:.2f}{suffix}")
        ax[0][i].plot(x, regu(x), label="I(x)")
        ax[0][i].plot(x, regu.backward_inputs(x), label="I'(x)")
        ax[0][i].legend()
        ax[0][i].set_ylim(-0.25, 1.5)
        ax[0][i].set_xlim(-1.5, 1.5)

    for i in range(5):
        x = torch.linspace(-3, 3, 1000)
        x_prime = 1 / 6 * torch.sign(x) * torch.ceil(torch.abs(x * 6) - 0.5)
        regu = ReLUGeLUInterpolation(interpolate_factor=i / 4, scaling_factor=1)
        suffix = ""
        if i == 0:
            suffix = " (ReLU)"
        elif i == 4:
            suffix = " (GELU)"
        ax[1][i].set_title(f"i={i / 4:.2f}{suffix}")
        ax[1][i].scatter(x, regu(x_prime), label="I(x)", s=1)
        ax[1][i].scatter(x, regu.backward_inputs(x_prime), label="I'(x)", s=1)
        ax[1][i].legend()
        ax[1][i].set_ylim(-0.25, 1.5)
        ax[1][i].set_xlim(-1.5, 1.5)

    for i in range(5):
        x = torch.linspace(-3, 3, 1000)
        x_prime = GaussianNoise(leakage=0.5, precision=6)(x)
        x_prime = 1 / 6 * torch.sign(x_prime) * torch.ceil(torch.abs(x_prime * 6) - 0.5)
        regu = ReLUGeLUInterpolation(interpolate_factor=i / 4, scaling_factor=1)
        suffix = ""
        if i == 0:
            suffix = " (ReLU)"
        elif i == 4:
            suffix = " (GELU)"
        ax[2][i].set_title(f"i={i / 4:.2f}{suffix}")
        ax[2][i].scatter(x, regu(x_prime), label="I(x)", s=1)
        ax[2][i].scatter(x, regu.backward_inputs(x_prime), label="I'(x)", s=1)
        ax[2][i].legend()
        ax[2][i].set_ylim(-0.25, 1.5)
        ax[2][i].set_xlim(-1.5, 1.5)
        ax[2][i].axhline(y=regu.backward_inputs(torch.tensor(0.0001)), color='red', linestyle='--', linewidth=1)
        ax[2][i].axhline(y=regu.backward_inputs(torch.tensor(0)), color='red', linestyle='--', linewidth=1)
        ax[2][i].axvline(x=-1 / 6, color='black', linestyle='--', linewidth=1)
        ax[2][i].axvline(x=1 / 6, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()
    fig.savefig("../_results/Figure 3.png", dpi=600)
    fig.savefig("../_results/Figure 3.svg", dpi=600)
    fig.savefig("../_results/Figure 3.pdf", dpi=600)


def figure4():
    bound = 1
    precision = 2 ** 1
    clamp = Clamp()
    rp = ReducePrecision(precision=precision)
    noise = GaussianNoise(leakage=0.5, precision=precision)

    inputs = np.linspace(-bound, bound, precision * 2 + 1)
    weights = np.linspace(-bound, bound, precision * 2 + 1)

    orig_mul = torch.tensor(np.matmul(inputs.reshape(-1, 1), weights.reshape(1, -1)))

    x1 = noise(torch.tensor(inputs, requires_grad=False))
    x1 = clamp(x1)
    x1 = rp(x1)
    x2 = noise(torch.tensor(weights, requires_grad=False))
    x2 = clamp(x2)
    x2 = rp(x2)
    n_mul = np.matmul(x1.reshape(-1, 1), x2.reshape(1, -1))
    n_mul = noise(n_mul)
    n_mul = clamp(n_mul)
    n_mul = rp(n_mul)

    activation = ReLUGeLUInterpolation(interpolate_factor=1, scaling_factor=1)
    grad_noise = activation.backward_inputs(n_mul)
    grad_orig = activation.backward_inputs(orig_mul)
    gelu_diff = np.abs(grad_noise.detach().numpy() - grad_orig.detach().numpy())

    activation = ReLUGeLUInterpolation(interpolate_factor=0, scaling_factor=1)
    grad_noise = activation.backward_inputs(n_mul)
    grad_orig = activation.backward_inputs(orig_mul)
    relu_diff = np.abs(grad_noise.detach().numpy() - grad_orig.detach().numpy())

    fig, ax = plt.subplots(1, 4, figsize=(2 * 6.50, 3.5))
    ax[0].contourf(inputs, weights, relu_diff, levels=10, vmax=1, vmin=0)
    ax[1].contourf(inputs, weights, relu_diff, levels=10, vmax=1, vmin=0)
    ax[1].set_xlim(-0.25, 0.25)
    ax[1].set_ylim(-0.25, 0.25)
    ax[3].contourf(inputs, weights, gelu_diff, levels=10, vmax=0.01, vmin=0)
    ax[2].contourf(inputs, weights, gelu_diff, levels=10, vmax=0.01, vmin=0)
    ax[3].set_xlim(-0.25, 0.25)
    ax[3].set_ylim(-0.25, 0.25)

    for i in ax:
        i.set_xlabel("Input")
        i.set_ylabel("Weight")
        i.set_aspect('equal', 'box')
    ax[0].set_title("ReLU")
    ax[1].set_title("ReLU (Zoomed)")
    ax[2].set_title("GELU")
    ax[3].set_title("GELU (Zoomed)")
    ax[3].set_title("GELU (Zoomed)")

    # colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(ax[0].contourf(inputs, weights, relu_diff, levels=100, vmax=1, vmin=0),
                 ticks=np.linspace(0, 1, 11).tolist(), cax=cbar_ax)
    cbar_ax.set_ylabel("Gradient Error")

    plt.tight_layout()
    plt.show()
    # fig.savefig("../_results/Figure 4.png", dpi=1200)
    fig.savefig("../_results/Figure 4.pdf")
    # fig.savefig("../_results/Figure 4.svg")


def figure35():
    x = np.linspace(0, 1, 100)
    discontinuity = np.zeros_like(x)
    for i, v in enumerate(x.tolist()):
        activation = ReLUGeLUInterpolation(interpolate_factor=v, scaling_factor=1)
        discontinuity[i] = activation.backward_inputs(torch.tensor(0.0001)) - activation.backward_inputs(
            torch.tensor(0))

    fig, ax = plt.subplots(1, 1, figsize=(2 * 6.5 * 1 / 3, 2 * 1.61803398874))
    plt.plot(x, discontinuity)
    ax.set_xlabel("Interpolation Factor")
    ax.set_ylabel("ReLU-GELU Gradient Discontinuity at 0")
    plt.tight_layout()
    plt.show()
    fig.savefig("../_results/Figure 3-5.png")
    fig.savefig("../_results/Figure 3-5.svg")
    fig.savefig("../_results/Figure 3-5.pdf")


def figure5():
    x = torch.linspace(-1.55, 1.55, 500)
    relu = ReLUGeLUInterpolation(interpolate_factor=0, scaling_factor=1)
    gelu = ReLUGeLUInterpolation(interpolate_factor=1, scaling_factor=1)
    silu = ReLUSiLUInterpolation(interpolate_factor=1)

    fig, ax = plt.subplots(1, 3, figsize=(2 * 6.50, 2.5), sharex=True, sharey=True)
    ax[0].plot(x, relu(x), label="ReLU")
    ax[0].plot(x, relu.backward_inputs(x), label="ReLU'")
    ax[1].plot(x, gelu(x), label="GELU")
    ax[1].plot(x, gelu.backward_inputs(x), label="GELU'")
    ax[2].plot(x, silu(x), label="SiLU")
    ax[2].plot(x, silu.backward_inputs(x), label="SiLU'")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[0].set_xlim(-1.5, 1.5)
    plt.tight_layout()
    plt.show()
    fig.savefig("../_results/Figure 5.png", dpi=600)
    fig.savefig("../_results/Figure 5.svg", dpi=600)
    fig.savefig("../_results/Figure 5.pdf", dpi=600)


def figure6():
    bound = 1
    precision = 2 ** 1
    clamp = Clamp()
    rp = ReducePrecision(precision=precision)
    noise = GaussianNoise(leakage=0.5, precision=precision)

    inputs = np.linspace(-bound, bound, precision * 2 + 1)
    weights = np.linspace(-bound, bound, precision * 2 + 1)

    orig_mul = torch.tensor(np.matmul(inputs.reshape(-1, 1), weights.reshape(1, -1)))

    x1 = noise(torch.tensor(inputs, requires_grad=False))
    x1 = clamp(x1)
    x1 = rp(x1)
    x2 = noise(torch.tensor(weights, requires_grad=False))
    x2 = clamp(x2)
    x2 = rp(x2)
    n_mul = np.matmul(x1.reshape(-1, 1), x2.reshape(1, -1))
    n_mul = noise(n_mul)
    n_mul = clamp(n_mul)
    n_mul = rp(n_mul)

    fig, ax = plt.subplots(1, 5, figsize=(2 * 6.50, 3.5))
    for i in reversed(list(range(5))):
        activation = ReLUGeLUInterpolation(interpolate_factor=i / 4, scaling_factor=1)
        grad_noise = activation.backward_inputs(n_mul)
        grad_orig = activation.backward_inputs(orig_mul)
        gelu_diff = np.abs(grad_noise.detach().numpy() - grad_orig.detach().numpy())

        ax[i].contourf(inputs, weights, gelu_diff, levels=10, vmax=1, vmin=0)
        ax[i].set_xlim(-0.25, 0.25)
        ax[i].set_ylim(-0.25, 0.25)
        ax[i].set_title(f"i={i / 4:.2f}")

    for i in ax:
        i.set_xlabel("Input")
        i.set_ylabel("Weight")
        i.set_aspect('equal', 'box')

    # colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(ax[0].contourf(inputs, weights, gelu_diff, levels=100, vmax=1, vmin=0),
                 ticks=np.linspace(0, 1, 11).tolist(), cax=cbar_ax)
    cbar_ax.set_ylabel("Gradient Error")

    plt.tight_layout()
    plt.show()
    # fig.savefig("../_results/Figure 6.png", dpi=1200)
    fig.savefig("../_results/Figure 6.pdf")
    # fig.savefig("../_results/Figure 6.svg")


def figure25():
    precision = 2 ** 6
    scaling_factors = np.linspace(0.5, 5, 100)
    effective_precision = np.zeros_like(scaling_factors)
    for i, v in enumerate(scaling_factors.tolist()):
        activation = ReLUGeLUInterpolation(interpolate_factor=1, scaling_factor=v)
        values = activation.backward_inputs(torch.tensor(1 / precision)) - activation.backward_inputs(torch.tensor(0))
        effective_precision[i] = values

    effective_precision = np.log2(1 / effective_precision)
    fig, ax = plt.subplots(1, 1, figsize=(2 * 6.5 * 1 / 3, 2 * 1.61803398874))
    plt.plot(scaling_factors, effective_precision)
    ax.set_xlim(0.75, 3.25)
    ax.set_ylim(4.5, 6.5)
    ax.set_xlabel("Scaling Factor")
    ax.set_ylabel("Effective Bit-Precision near 0")
    plt.tight_layout()
    plt.show()
    fig.savefig("../_results/Figure 2-5.png")
    fig.savefig("../_results/Figure 2-5.svg")
    fig.savefig("../_results/Figure 2-5.pdf")


if __name__ == '__main__':
    figure2()
    figure25()
    figure3()
    figure35()
    figure4()
    figure5()
    figure6()
