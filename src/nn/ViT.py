from typing import Type, Optional

import torch
from analogvnn.nn.module.Layer import Layer
from analogvnn.nn.normalize.Normalize import Normalize
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


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
            nn.Linear(mlp_dim, dim),
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
    def __init__(
            self, *,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            heads,
            mlp_dim,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.,
            activation_fn: Type[Layer],
            activation_i: float,
            activation_s: float,
            activation_alpha: float,
            norm_class: Optional[Type[Normalize]],
            precision_class: Type[Layer],
            precision: Optional[int],
            noise_class: Type[Layer],
            leakage: Optional[float],
            device: torch.device = "cpu",
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.pool = pool
        self.channels = channels
        self.dim_head = dim_head
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.activation_fn = activation_fn
        self.activation_i = activation_i
        self.activation_s = activation_s
        self.activation_alpha = activation_alpha
        self.norm_class = norm_class
        self.precision_class = precision_class
        self.precision = precision
        self.noise_class = noise_class
        self.leakage = leakage
        self.device = device

        image_height, image_width = image_size
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.norm_class_layer = norm_class() if norm_class is not None else None
        self.precision_class_layer = precision_class(precision=precision) if precision_class is not None else None
        self.noise_class_layer = noise_class(leakage=leakage, precision=precision) if noise_class is not None else None

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.emb_dropout_layer = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            activation_fn=activation_fn,
            activation_i=activation_i,
            activation_s=activation_s,
            activation_alpha=activation_alpha
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = img
        if self.norm_class_layer is not None:
            x = self.norm_class_layer(x)
        if self.precision_class_layer is not None:
            x = self.precision_class_layer(x)
        if self.noise_class_layer is not None:
            x = self.noise_class_layer(x)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.emb_dropout_layer(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def parameters(self, recurse: bool = True):
        return filter(lambda p: p.__class__ == nn.Parameter, super().parameters(recurse))

    def hyperparameters(self):
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_classes": self.num_classes,
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
            "mlp_dim": self.mlp_dim,
            "pool": self.pool,
            "channels": self.channels,
            "dim_head": self.dim_head,
            "dropout": self.dropout,
            "emb_dropout": self.emb_dropout,
            'activation_fn': self.activation_fn.__name__,
            'activation_i': self.activation_i,
            'activation_s': self.activation_s,
            'activation_alpha': self.activation_alpha,
            'norm_class_y': self.norm_class.__name__ if self.norm_class is not None else str(None),
            'precision_class_y': self.precision_class.__name__ if self.precision_class is not None else str(None),
            'precision_y': self.precision,
            'noise_class_y': self.noise_class.__name__ if self.noise_class is not None else str(None),
            'leakage_y': self.leakage,
            'device': str(self.device),
        }
