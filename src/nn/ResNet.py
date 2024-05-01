import dataclasses
import json
from dataclasses import dataclass
from functools import partial
from typing import Callable, Union
from typing import Optional, Tuple
from typing import Type, List

import torch
import torch.backends.cudnn
import torchvision
from analogvnn.nn.module.Layer import Layer
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
    optimizer: Type[optim.Optimizer] = partial(optim.Adam, lr=0.001, weight_decay=0.0001)
    batch_size: int = 32
    epochs: int = 200
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

    def create_doa_layer(self) -> List[Layer]:
        layer_list = [
            self.create_norm_layer(),
            self.create_precision_layer(),
            self.create_noise_layer(),
        ]
        layer_list = [x for x in layer_list if x is not None]
        return layer_list

    def create_aod_layer(self) -> List[Layer]:
        layer_list = [
            self.create_noise_layer(),
            self.create_norm_layer(),
            self.create_precision_layer(),
        ]
        layer_list = [x for x in layer_list if x is not None]
        return layer_list

    @property
    def json(self):
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    def __repr__(self):
        return f"{self.__class__.__name__}({json.dumps(self.json)})"


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
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
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            hyperparameters: ResNetRunParameters
    ) -> None:
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

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.hyperparameters.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.hyperparameters.num_classes)

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
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self.hyperparameters.norm_layer(planes * block.expansion),
            )

        layers = [block(
            self.inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            groups=self.hyperparameters.groups,
            base_width=self.hyperparameters.width_per_group,
            dilation=self.hyperparameters.dilation,
            norm_layer=self.hyperparameters.norm_layer,
        )]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.hyperparameters.groups,
                    base_width=self.hyperparameters.width_per_group,
                    dilation=self.hyperparameters.dilation,
                    norm_layer=self.hyperparameters.norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
