import argparse
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Dict, Tuple, cast
from typing import Type, Union, List

import torch
import torch.backends.cudnn
import torchinfo
import torchvision
from analogvnn.nn.module.FullSequential import FullSequential
from analogvnn.nn.module.Layer import Layer
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import Clamp
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from analogvnn.parameter.PseudoParameter import PseudoParameter
from analogvnn.utils.is_cpu_cuda import is_cpu_cuda
from torch import nn
from torch import optim
from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms

from src.dataloaders.load_vision_dataset import load_vision_dataset
from src.fn.cross_entropy_loss_accuracy import cross_entropy_loss_accuracy
from src.fn.data_dirs import data_dirs
from src.nn.ReLUGeLUInterpolation import ReLUGeLUInterpolation
from src.nn.WeightModel import WeightModel

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


@dataclass
class VGGRunParameters:
    name: Optional[str] = None
    data_folder: Optional[str] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = "A"

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
    batch_size: int = 500
    epochs: int = 200
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

    def get_activation_fn(self) -> ReLUGeLUInterpolation:
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
        return layer_list

    @property
    def json(self):
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    def __repr__(self):
        return f"{self.__class__.__name__}({json.dumps(self.json)})"


class VGGModel(FullSequential):
    def __init__(self, hyperparameters: VGGRunParameters):
        super(VGGModel, self).__init__()
        self.hyperparameters = hyperparameters
        self.layers_cfg = cfgs[self.hyperparameters.model_version]
        self.all_layers: List[nn.Module] = []

        in_channels = self.hyperparameters.input_shape[1]
        for v in self.layers_cfg:
            if v == "M":
                self.all_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                self.all_layers += [
                    *self.hyperparameters.create_doa_layer(),
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    self.hyperparameters.get_activation_fn(),
                ]
                in_channels = v

        self.all_layers += [
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(start_dim=1),

            *self.hyperparameters.create_doa_layer(),
            nn.Linear(512 * 7 * 7, 4096),
            self.hyperparameters.get_activation_fn(),
            nn.Dropout(p=0.5),

            *self.hyperparameters.create_doa_layer(),
            nn.Linear(4096, 4096),
            self.hyperparameters.get_activation_fn(),
            nn.Dropout(p=0.5),

            *self.hyperparameters.create_doa_layer(),
            nn.Linear(4096, self.hyperparameters.num_classes),
        ]
        self.all_layers = [x for x in self.all_layers if x is not None]

        for m in self.all_layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.add_sequence(*self.all_layers)


def run_model(parameters: VGGRunParameters):
    torch.backends.cudnn.benchmark = True
    is_cpu_cuda.use_cuda_if_available()

    if parameters.device is not None:
        is_cpu_cuda.set_device(str(parameters.device))
    device, is_cuda = is_cpu_cuda.is_using_cuda
    parameters.device = device
    parameters.is_cuda = is_cuda

    if parameters.data_folder is None:
        raise Exception("data_folder is None")

    if parameters.name is None:
        parameters.name = hashlib.sha256(str(parameters).encode("utf-8")).hexdigest()[:8]

    print(f"Parameters: {parameters}")
    print(f"Name: {parameters.name}")
    print(f"Device: {parameters.device}")

    paths = data_dirs(
        parameters.data_folder,
        name=parameters.name,
        timestamp=parameters.timestamp
    )
    parameters.timestamp = paths.timestamp
    log_file = paths.logs.joinpath(f"{paths.name}_logs.txt")

    if paths.tensorboard.exists():
        for file in paths.tensorboard.iterdir():
            file.unlink()
        paths.tensorboard.rmdir()

    print(f"Timestamp: {paths.timestamp}")
    print(f"Storage name: {paths.name}")
    print()

    print(f"Loading Data...")
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=parameters.dataset,
        path=paths.dataset,
        batch_size=parameters.batch_size,
        is_cuda=is_cuda,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    print(f"Creating Models...")
    parameters.input_shape = input_shape
    parameters.num_classes = len(classes)
    parameters.model_name = VGGModel.__name__
    nn_model = VGGModel(hyperparameters=parameters)
    weight_model = WeightModel(
        norm_class=parameters.norm_class,
        precision_class=parameters.precision_class,
        precision=parameters.precision,
        noise_class=parameters.noise_class,
        leakage=parameters.leakage,
    )

    PseudoParameter.parametrize_module(nn_model, transformation=weight_model)

    nn_model.loss_function = parameters.loss_function()
    nn_model.accuracy_function = cross_entropy_loss_accuracy
    parameters.accuracy_function = nn_model.accuracy_function.__name__
    nn_model.optimizer = parameters.optimizer(params=nn_model.parameters())

    nn_model.compile(device=device)
    weight_model.compile(device=device)

    print(f"Creating Log File...")
    with open(log_file, "a+", encoding="utf-8") as file:
        file.write(json.dumps(parameters.json, sort_keys=True, indent=2) + "\n\n")
        file.write(str(nn_model.optimizer) + "\n\n")

        file.write(str(nn_model) + "\n\n")
        file.write(str(weight_model) + "\n\n")
        file.write(torchinfo.summary(nn_model, input_size=input_shape, device=device).__repr__() + "\n\n")
        file.write(torchinfo.summary(weight_model, input_size=(1, 1), device=device).__repr__() + "\n\n")

    loss_accuracy = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    print(f"Starting Training...")
    for epoch in range(parameters.epochs):
        train_loss, train_accuracy = nn_model.train_on(
            train_loader,
            epoch=epoch,
            test_run=parameters.test_run,
        )
        test_loss, test_accuracy = nn_model.test_on(
            test_loader,
            epoch=epoch,
            test_run=parameters.test_run
        )

        loss_accuracy["train_loss"].append(train_loss)
        loss_accuracy["train_accuracy"].append(train_accuracy)
        loss_accuracy["test_loss"].append(test_loss)
        loss_accuracy["test_accuracy"].append(test_accuracy)

        str_epoch = str(epoch + 1).zfill(math.ceil(math.log10(parameters.epochs)))
        print_str = f'({str_epoch})' \
                    f' Training Loss: {train_loss:.4f},' \
                    f' Training Accuracy: {100. * train_accuracy:.0f}%,' \
                    f' Testing Loss: {test_loss:.4f},' \
                    f' Testing Accuracy: {100. * test_accuracy:.0f}%\n'
        print(print_str)

        parameters.last_epoch = epoch
        with open(log_file, "a+", encoding="utf-8") as file:
            file.write(print_str)

        if epoch >= 9 and train_accuracy < 0.125 and parameters.dataset == torchvision.datasets.CIFAR10:
            break

        if epoch >= 9 and train_accuracy < 0.0125 and parameters.dataset == torchvision.datasets.CIFAR100:
            break

        if test_accuracy < (max(loss_accuracy["test_accuracy"]) - 5 / 100):
            break

        if parameters.test_run:
            break

    if parameters.save_data:
        torch.save(str(nn_model), f"{paths.model_data}/{parameters.last_epoch}_str_nn_model")
        torch.save(str(weight_model), f"{paths.model_data}/{parameters.last_epoch}_str_weight_model")
        torch.save(parameters.json, f"{paths.model_data}/{parameters.last_epoch}_parameters_json")
        torch.save(loss_accuracy, f"{paths.model_data}/{parameters.last_epoch}_loss_accuracy")

    with open(log_file, "a+", encoding="utf-8") as file:
        file.write("Run Completed Successfully...")
    print()


def this_path():
    return Path(__file__)


def get_parameters(kwargs) -> VGGRunParameters:
    parameters = VGGRunParameters()

    for key, value in kwargs.items():
        if hasattr(parameters, key):
            setattr(parameters, key, value)

    return parameters


def run_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--data_folder", type=str, required=True)

    parser.add_argument("--activation_i", type=float, default=1.0)
    parser.add_argument("--precision", type=int, default=64.0)
    parser.add_argument("--leakage", type=float, required=True)

    parser.add_argument("--test_run", action='store_true')
    parser.set_defaults(test_run=False)
    parser.add_argument("--save_data", action='store_true')
    parser.set_defaults(save_data=False)
    kwargs = vars(parser.parse_known_args()[0])
    print(json.dumps(kwargs))
    print()

    kwargs["data_folder"] = Path(kwargs["data_folder"]).absolute()
    return kwargs


def run():
    kwargs = run_parser()
    parameters = get_parameters(kwargs)
    run_model(parameters)


if __name__ == '__main__':
    run()
