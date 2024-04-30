import dataclasses
import json
from typing import List, Tuple, Type, Optional

import torch
import torchvision
from analogvnn.nn.module.Layer import Layer
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import Clamp
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch import optim
from torchvision.datasets import VisionDataset

from src.nn.ReGLUGeGLUInterpolation import ReGLUGeGLUInterpolation
from src.nn.ReLUGeLUInterpolation import ReLUGeLUInterpolation


@dataclasses.dataclass
class ViTRunParameters:
    name: Optional[str] = None
    data_folder: Optional[str] = None

    patch_size: int = 4
    dim: int = 512
    depth: int = 6
    heads: int = 8
    mlp_dim: int = 512
    pool: str = 'cls'
    channels: int = 3
    dim_head: int = 64
    dropout: float = 0.5
    emb_dropout: float = 0.5

    activation_fn = ReLUGeLUInterpolation
    activation_i: float = 0.0
    activation_s: float = 1.0
    activation_alpha: float = 0.0
    norm_class = Clamp
    precision_class = ReducePrecision
    noise_class = GaussianNoise
    precision: float = 64.0
    leakage: float = None

    dataset: Type[VisionDataset] = torchvision.datasets.CIFAR10
    input_shape: Tuple[int, int] = (32, 32)
    num_classes: int = 10
    color: bool = True

    loss_function = nn.CrossEntropyLoss
    accuracy_function: str = None
    optimizer: Type[optim.Optimizer] = optim.Adam
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    batch_size: int = 512
    epochs: int = 150
    last_epoch: Optional[int] = 0

    device: Optional[torch.device] = None
    is_cuda: bool = False
    test_run: bool = False
    tensorboard: bool = False
    save_data: bool = True
    timestamp: str = None

    def create_norm_layer(self) -> Clamp:
        return self.norm_class()

    def create_precision_layer(self) -> ReducePrecision:
        return self.precision_class(precision=self.precision)

    def create_noise_layer(self) -> GaussianNoise:
        return self.noise_class(leakage=self.leakage, precision=self.precision)

    def get_activation_fn(self) -> Layer:
        return self.activation_fn(
            interpolate_factor=self.activation_i,
            scaling_factor=self.activation_s,
            alpha=self.activation_alpha
        )

    def create_doa_layer(self) -> List[Layer]:
        layer_list = [
            self.create_norm_layer(),
            self.create_precision_layer(),
            self.create_noise_layer(),
        ]
        layer_list = [x for x in layer_list if x is not None]
        return nn.Sequential(*layer_list)

    @property
    def json(self):
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    def __repr__(self):
        return f"{self.__class__.__name__}({json.dumps(self.json)})"


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            mlp_dim,
            dropout,
            activation_fn: Type[Layer],
            activation_i: float = 0.,
            activation_s: float = 1.,
            activation_alpha: float = 1.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            activation_fn(activation_i, activation_s, activation_alpha),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_dim / 2) if activation_fn == ReGLUGeGLUInterpolation else mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.5):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            activation_fn: Type[Layer],
            activation_i: float = 0.,
            activation_s: float = 1.,
            activation_alpha: float = 1.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            attention = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            feed_forward = FeedForward(
                dim=dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                activation_i=activation_i,
                activation_s=activation_s,
                activation_alpha=activation_alpha
            )
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attention),
                PreNorm(dim, feed_forward),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, hyperparameters: ViTRunParameters):
        super().__init__()
        self.hyperparameters = hyperparameters

        image_height, image_width = self.hyperparameters.input_shape, self.hyperparameters.input_shape
        patch_height, patch_width = pair(self.hyperparameters.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = self.hyperparameters.channels * patch_height * patch_width
        assert self.hyperparameters.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, self.hyperparameters.dim),
        )

        self.doa_layers = self.hyperparameters.create_doa_layer()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.hyperparameters.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hyperparameters.dim))
        self.emb_dropout_layer = nn.Dropout(self.hyperparameters.emb_dropout)

        self.transformer = Transformer(
            dim=self.hyperparameters.dim,
            depth=self.hyperparameters.depth,
            heads=self.hyperparameters.heads,
            dim_head=self.hyperparameters.dim_head,
            mlp_dim=self.hyperparameters.mlp_dim,
            dropout=self.hyperparameters.dropout,
            activation_fn=self.hyperparameters.activation_fn,
            activation_i=self.hyperparameters.activation_i,
            activation_s=self.hyperparameters.activation_s,
            activation_alpha=self.hyperparameters.activation_alpha
        )

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.hyperparameters.dim),
            nn.Linear(self.hyperparameters.dim, self.hyperparameters.num_classes)
        )

    def forward(self, img):
        x = img

        for layer in self.doa_layers:
            x = layer(x)

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.emb_dropout_layer(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.hyperparameters.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def parameters(self, recurse: bool = True):
        return filter(lambda p: p.__class__ == nn.Parameter, super().parameters(recurse))
