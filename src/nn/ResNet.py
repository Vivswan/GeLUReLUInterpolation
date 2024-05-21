import dataclasses
import json
from dataclasses import dataclass
from functools import partial
from typing import Callable, Union
from typing import Optional, Tuple
from typing import Type

import torch.backends.cudnn
import torchvision
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import Clamp
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from torch import Tensor
from torch import nn
from torch import optim
from torchvision.datasets import VisionDataset

from src.nn.ReLUGeLUInterpolation import ReLUGeLUInterpolation

cfgs = {
    "18": [2, 2, 2, 2],
    "34": [3, 4, 6, 3],
    "50": [3, 4, 6, 3],
    "101": [3, 4, 23, 3],
    "152": [3, 8, 36, 3],
}


@dataclass
class ResNetRunParameters:
    name: Optional[str] = None
    data_folder: Optional[str] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = "18"
    groups: int = 1
    width_per_group: int = 64
    norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    dilation: int = 1

    activation_fn = ReLUGeLUInterpolation
    activation_i: float = 0.0
    activation_s: float = 1.0
    activation_alpha: float = 0.0
    norm_class = Clamp
    precision_class = ReducePrecision
    noise_class = GaussianNoise
    precision: float = 64.0
    leakage: float = None

    dataset: Type[VisionDataset] = torchvision.datasets.CIFAR100
    input_shape: Tuple[int, int] = (32, 32)
    num_classes: int = 100
    color: bool = True

    loss_function = nn.CrossEntropyLoss
    accuracy_function: str = None
    optimizer: Type[optim.Optimizer] = partial(optim.Adam, lr=0.001)
    batch_size: int = 256 + 128
    epochs: int = 100
    last_epoch: Optional[int] = 0

    device: Optional[torch.device] = None
    is_cuda: bool = False
    test_run: bool = False
    tensorboard: bool = False
    save_data: bool = True
    timestamp: str = None

    def create_norm_layer(self) -> Clamp:
        layer = self.norm_class()
        layer.use_autograd_graph = True
        return layer

    def create_precision_layer(self) -> ReducePrecision:
        layer = self.precision_class(precision=self.precision)
        layer.use_autograd_graph = True
        return layer

    def create_noise_layer(self) -> GaussianNoise:
        layer = self.noise_class(leakage=self.leakage, precision=self.precision)
        layer.use_autograd_graph = True
        return layer

    def get_activation_fn(self) -> ReLUGeLUInterpolation:
        layer = self.activation_fn(
            interpolate_factor=self.activation_i,
            scaling_factor=self.activation_s,
            alpha=self.activation_alpha
        )
        layer.use_autograd_graph = True
        return layer

    def create_doa_layer(self) -> nn.Module:
        layer_list = [
            self.create_norm_layer(),
            self.create_precision_layer(),
            self.create_noise_layer(),
        ]
        layer_list = [x for x in layer_list if x is not None]
        return nn.Sequential(*layer_list) if len(layer_list) != 0 else nn.Identity()

    def create_aod_layer(self) -> nn.Module:
        layer_list = [
            self.create_noise_layer(),
            self.create_norm_layer(),
            self.create_precision_layer(),
        ]
        layer_list = [x for x in layer_list if x is not None]
        return nn.Sequential(*layer_list) if len(layer_list) != 0 else nn.Identity()

    @property
    def json(self):
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    def __repr__(self):
        return f"{self.__class__.__name__}({json.dumps(self.json)})"


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            hyperparameters: ResNetRunParameters,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.hyperparameters = hyperparameters
        self.stride = stride

        if self.hyperparameters.groups != 1 or self.hyperparameters.width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and width_per_group=64")
        if self.hyperparameters.dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.block = nn.Sequential(
            self.hyperparameters.create_doa_layer(),
            conv3x3(inplanes, planes, stride),
            self.hyperparameters.norm_layer(planes),
            self.hyperparameters.get_activation_fn(),
            self.hyperparameters.create_doa_layer(),
            conv3x3(planes, planes),
            self.hyperparameters.norm_layer(planes),
        )
        self.downsample = downsample if downsample is not None else self.hyperparameters.create_doa_layer()
        self.activation = self.hyperparameters.get_activation_fn()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.downsample(x) + self.block(x))


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            hyperparameters: ResNetRunParameters,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.hyperparameters = hyperparameters
        self.stride = stride
        width = int(planes * (self.hyperparameters.width_per_group / 64.0)) * self.hyperparameters.groups

        self.block = nn.Sequential(
            self.hyperparameters.create_doa_layer(),
            conv1x1(inplanes, width),
            self.hyperparameters.norm_layer(width),
            self.hyperparameters.get_activation_fn(),
            self.hyperparameters.create_doa_layer(),
            conv3x3(width, width, stride, self.hyperparameters.groups, self.hyperparameters.dilation),
            self.hyperparameters.norm_layer(width),
            self.hyperparameters.get_activation_fn(),
            self.hyperparameters.create_doa_layer(),
            conv1x1(width, planes * self.expansion),
            self.hyperparameters.norm_layer(planes * self.expansion),
        )
        self.downsample = downsample if downsample is not None else self.hyperparameters.create_doa_layer()
        self.activation = self.hyperparameters.get_activation_fn()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.downsample(x) + self.block(x))


class ResNet(nn.Module):
    def __init__(self, hyperparameters: ResNetRunParameters):
        super().__init__()
        self.hyperparameters = hyperparameters

        if self.hyperparameters.model_version in ["18", "34"]:
            block = BasicBlock
        elif self.hyperparameters.model_version in ["50", "101", "152"]:
            block = Bottleneck
        else:
            raise NotImplementedError(f"Model {self.hyperparameters.model_version} not implemented")

        self.inplanes = 64
        layers = cfgs[self.hyperparameters.model_version]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.block = nn.Sequential(
            self.hyperparameters.create_doa_layer(),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            self.hyperparameters.norm_layer(num_features=64),
            self.hyperparameters.get_activation_fn(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            self.hyperparameters.create_doa_layer(),
            nn.Linear(512 * block.expansion, self.hyperparameters.num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.hyperparameters.create_doa_layer(),
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self.hyperparameters.norm_layer(planes * block.expansion),
            )

        layers = [block(
            hyperparameters=self.hyperparameters,
            inplanes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
        )]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                hyperparameters=self.hyperparameters,
                inplanes=self.inplanes,
                planes=planes,
            ))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
