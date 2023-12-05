import math
from typing import Type, Optional

import torch
from analogvnn.nn.module.Layer import Layer
from analogvnn.nn.normalize.Normalize import Normalize
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
            self,
            num_transformer_layers: int,
            num_tokens: int,
            embedding_dim: int,
            dim_feedforward: int,
            num_heads: int,
            dropout: float,
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
        self.model_type = 'Transformer'
        self.num_transformer_layers = num_transformer_layers
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.dim_feedforward = dim_feedforward
        self.num_heads = num_heads
        self.dropout = dropout
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

        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(
            embedding_dim,
            num_heads,
            dim_feedforward,
            dropout,
            activation=activation_fn(activation_i, activation_s, activation_alpha),
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_transformer_layers)
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_tokens)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Resets gradients of all model parameters. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def parameters(self, recurse: bool = True):
        return filter(lambda p: p.__class__ == nn.Parameter, super().parameters(recurse))

    def hyperparameters(self):
        return {
            'model_type': self.model_type,
            'num_transformer_layers': self.num_transformer_layers,
            'num_tokens': self.num_tokens,
            'embedding_dim': self.embedding_dim,
            'dim_feedforward': self.dim_feedforward,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'activation_fn': self.activation_fn.__name__,
            'activation_i': self.activation_i,
            'activation_s': self.activation_s,
            'activation_alpha': self.activation_alpha,
            'norm_class': self.norm_class,
            'precision_class': self.precision_class,
            'precision': self.precision,
            'noise_class': self.noise_class,
            'leakage': self.leakage,
            'device': str(self.device),
        }
