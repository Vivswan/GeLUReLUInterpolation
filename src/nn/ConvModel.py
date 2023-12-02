from typing import Type, Union, List

import torch
from analogvnn.nn.Linear import Linear
from analogvnn.nn.module.FullSequential import FullSequential
from analogvnn.nn.module.Layer import Layer
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Normalize import Normalize
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from torch import nn
from torch.nn import Flatten

from src.fn.pick_instanceof_module import pick_instanceof_module
from src.nn.ReLUGeLUInterpolation import ReLUGeLUInterpolation


class ConvModel(FullSequential):
    def __init__(
            self,
            input_shape, num_classes,
            num_conv_layer: int = 6,
            num_linear_layer: int = 3,
            activation_fn: Type[Layer] = ReLUGeLUInterpolation,
            activation_i: float = 1.0,
            activation_s: float = 1.0,
            activation_alpha: float = 0.0,
            norm_class: Type[Normalize] = None,
            precision_class: Type[Union[ReducePrecision]] = None,
            precision: Union[int, None] = None,
            noise_class: Type[Union[GaussianNoise]] = None,
            leakage: Union[float, None] = None,
    ):
        super(ConvModel, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_conv_layer = num_conv_layer
        self.num_linear_layer = num_linear_layer
        self.activation_fn = activation_fn
        self.activation_i = activation_i
        self.activation_s = activation_s
        self.activation_alpha = activation_alpha
        self.norm_class = norm_class
        self.precision_class = precision_class
        self.precision = precision
        self.noise_class = noise_class
        self.leakage = leakage

        self.all_layers: List[nn.Module] = []

        temp_x = torch.zeros(input_shape, requires_grad=False)
        if self.num_conv_layer >= 1:
            self.add_doa_layers()
            self.all_layers.append(nn.Conv2d(
                in_channels=self.input_shape[1],
                out_channels=48,
                kernel_size=(3, 3),
                padding=(1, 1)
            ))
            temp_x = self.all_layers[-1](temp_x)
            self.add_aod_layers()
        if self.num_conv_layer >= 2:
            self.add_doa_layers()
            self.all_layers.append(nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3)))
            temp_x = self.all_layers[-1](temp_x)
            self.add_aod_layers()
            self.all_layers.append(nn.MaxPool2d(2, 2))
            temp_x = self.all_layers[-1](temp_x)
        if self.num_conv_layer >= 3:
            self.all_layers.append(nn.Dropout(0.25))
            self.add_doa_layers()
            self.all_layers.append(nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3), padding=(1, 1)))
            temp_x = self.all_layers[-1](temp_x)
            self.add_aod_layers()
        if self.num_conv_layer >= 4:
            self.add_doa_layers()
            self.all_layers.append(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3)))
            temp_x = self.all_layers[-1](temp_x)
            self.add_aod_layers()
            self.all_layers.append(nn.MaxPool2d(2, 2))
            temp_x = self.all_layers[-1](temp_x)
            self.all_layers.append(nn.Dropout(0.25))
        if self.num_conv_layer >= 5:
            self.add_doa_layers()
            self.all_layers.append(nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), padding=(1, 1)))
            temp_x = self.all_layers[-1](temp_x)
            self.add_aod_layers()
        if self.num_conv_layer >= 6:
            self.add_doa_layers()
            self.all_layers.append(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3)))
            temp_x = self.all_layers[-1](temp_x)
            self.add_aod_layers()
            self.all_layers.append(nn.MaxPool2d(2, 2))
            temp_x = self.all_layers[-1](temp_x)
            self.all_layers.append(nn.Dropout(0.25))

        self.all_layers.append(Flatten(start_dim=1))
        temp_x = self.all_layers[-1](temp_x)

        if self.num_linear_layer >= 3:
            self.add_doa_layers()
            self.all_layers.append(Linear(in_features=temp_x.shape[1], out_features=512))
            temp_x = self.all_layers[-1](temp_x)
            self.add_aod_layers()
            self.all_layers.append(nn.Dropout(0.5))
        if self.num_linear_layer >= 2:
            self.add_doa_layers()
            self.all_layers.append(Linear(in_features=temp_x.shape[1], out_features=256))
            temp_x = self.all_layers[-1](temp_x)
            self.add_aod_layers()
            self.all_layers.append(nn.Dropout(0.5))

        self.add_doa_layers()
        self.all_layers.append(Linear(in_features=temp_x.shape[1], out_features=num_classes))
        self.add_aod_layers()

        self.conv2d_layers = pick_instanceof_module(self.all_layers, nn.Conv2d)
        self.max_pool2d_layers = pick_instanceof_module(self.all_layers, nn.MaxPool2d)
        self.linear_layers = pick_instanceof_module(self.all_layers, Linear)
        self.activation_layers = pick_instanceof_module(self.all_layers, self.activation_fn)
        self.norm_layers = pick_instanceof_module(self.all_layers, norm_class)
        self.precision_layers = pick_instanceof_module(self.all_layers, precision_class)
        self.noise_layers = pick_instanceof_module(self.all_layers, noise_class)

        for i in self.linear_layers:
            nn.init.kaiming_uniform_(i.weight)

        self.add_sequence(*self.all_layers)

    def add_doa_layers(self):
        if self.norm_class is not None:
            self.all_layers.append(self.norm_class())
        if self.precision_class is not None:
            self.all_layers.append(self.precision_class(precision=self.precision))
        if self.noise_class is not None:
            self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))

    def add_aod_layers(self):
        if self.noise_class is not None:
            self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))
        if self.norm_class is not None:
            self.all_layers.append(self.norm_class())
        if self.precision_class is not None:
            self.all_layers.append(self.precision_class(precision=self.precision))

        self.all_layers.append(self.activation_fn(
            interpolate_factor=self.activation_i,
            scaling_factor=self.activation_s,
            alpha=self.activation_alpha
        ))

    def hyperparameters(self):
        return {
            'nn_model_class': self.__class__.__name__,

            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'num_conv_layer': self.num_conv_layer,
            'num_linear_layer': self.num_linear_layer,
            'activation_fn': self.activation_fn.__name__,
            'activation_i': self.activation_i,
            'activation_s': self.activation_s,
            'activation_alpha': self.activation_alpha,
            'norm_class_y': self.norm_class.__name__ if self.norm_class is not None else str(None),
            'precision_class_y': self.precision_class.__name__ if self.precision_class is not None else str(None),
            'precision_y': self.precision,
            'noise_class_y': self.noise_class.__name__ if self.noise_class is not None else str(None),
            'leakage_y': self.leakage,

            'loss_class': self.loss_function.__class__.__name__,
            'accuracy_fn': self.accuracy_function.__name__,
            'optimiser_superclass': self.optimizer.__class__.__name__,
        }
