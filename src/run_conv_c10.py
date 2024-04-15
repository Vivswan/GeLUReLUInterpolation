import argparse
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Type, Union, List

import numpy as np
import torch
import torch.backends.cudnn
import torchinfo
import torchvision
from analogvnn.nn.Linear import Linear
from analogvnn.nn.module.FullSequential import FullSequential
from analogvnn.nn.module.Layer import Layer
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import Clamp
from analogvnn.nn.normalize.Normalize import Normalize
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from analogvnn.parameter.PseudoParameter import PseudoParameter
from analogvnn.utils.is_cpu_cuda import is_cpu_cuda
from torch import nn
from torch import optim
from torch.nn import Flatten
from torchvision.datasets import VisionDataset

from src.dataloaders.load_vision_dataset import load_vision_dataset
from src.fn.cross_entropy_loss_accuracy import cross_entropy_loss_accuracy
from src.fn.data_dirs import data_dirs
from src.fn.misc import select_class, check
from src.fn.pick_instanceof_module import pick_instanceof_module
from src.nn.ReLUGeLUInterpolation import ReLUGeLUInterpolation
from src.nn.ReLUSiLUInterpolation import ReLUSiLUInterpolation
from src.nn.WeightModel import WeightModel


class ConvModel(FullSequential):
    def __init__(
            self,
            input_shape, num_classes,
            num_conv_layer: int = 6,
            num_linear_layer: int = 3,
            activation_fn: Type[Union[ReLUGeLUInterpolation, ReLUSiLUInterpolation]] = ReLUGeLUInterpolation,
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

@dataclass
class ConvRunParameters:
    name: Optional[str] = None
    data_folder: Optional[str] = None

    num_conv_layer: int = 6
    num_linear_layer: int = 3
    activation_fn: Type[Layer] = ReLUGeLUInterpolation
    activation_i: float = 0
    activation_s: float = 1
    activation_alpha: float = 0
    norm_class: Optional[Type[Normalize]] = None
    precision_class: Type[Layer] = None
    precision: Optional[int] = None
    noise_class: Type[Layer] = None
    leakage: Optional[float] = None

    dataset: Type[VisionDataset] = torchvision.datasets.CIFAR10
    color: bool = True
    batch_size: int = 512
    epochs: int = 200
    lr: float = 1e-3

    device: Optional[torch.device] = None
    test_logs: bool = False
    test_run: bool = False
    tensorboard: bool = False
    save_data: bool = True
    timestamp: str = None

    @property
    def nn_model_params(self):
        return {
            "num_linear_layer": self.num_linear_layer,
            "activation_fn": self.activation_fn,
            "activation_i": self.activation_i,
            "activation_s": self.activation_s,
            "activation_alpha": self.activation_alpha,
            "norm_class": self.norm_class,
            "precision_class": self.precision_class,
            "precision": self.precision,
            "noise_class": self.noise_class,
            "leakage": self.leakage,
        }

    @property
    def weight_model_params(self):
        return {
            "norm_class": self.norm_class,
            "precision_class": self.precision_class,
            "precision": self.precision,
            "noise_class": self.noise_class,
            "leakage": self.leakage,
        }

    @property
    def json(self):
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    def __repr__(self):
        return f"{self.__class__.__name__}({json.dumps(self.json)})"


def run_model(parameters: ConvRunParameters):
    torch.backends.cudnn.benchmark = True
    is_cpu_cuda.use_cuda_if_available()

    if parameters.device is not None:
        is_cpu_cuda.set_device(str(parameters.device))
    device, is_cuda = is_cpu_cuda.is_using_cuda
    parameters.device = device

    if parameters.data_folder is None:
        raise Exception("data_folder is None")

    if parameters.name is None:
        parameters.name = hashlib.sha256(str(parameters).encode("utf-8")).hexdigest()

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

    print(f"Timestamp: {paths.timestamp}")
    print(f"Storage name: {paths.name}")
    print()

    print(f"Loading Data...")
    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=parameters.dataset,
        path=paths.dataset,
        batch_size=parameters.batch_size,
        is_cuda=is_cuda,
        grayscale=not parameters.color
    )

    nn_model_params = parameters.nn_model_params
    weight_model_params = parameters.weight_model_params
    nn_model_params["input_shape"] = input_shape
    nn_model_params["num_classes"] = len(classes)

    print(f"Creating Models...")
    nn_model = ConvModel(**nn_model_params)
    weight_model = WeightModel(**weight_model_params)
    if parameters.tensorboard:
        nn_model.create_tensorboard(paths.tensorboard)

    PseudoParameter.parametrize_module(nn_model, transformation=weight_model)

    nn_model.loss_function = nn.CrossEntropyLoss()
    nn_model.accuracy_function = cross_entropy_loss_accuracy
    nn_model.optimizer = optim.Adam(params=nn_model.parameters(), lr=parameters.lr)

    nn_model.compile(device=device, layer_data=True)
    weight_model.compile(device=device)

    parameter_log = {
        'dataset': parameters.dataset.__name__,
        'batch_size': parameters.batch_size,
        'is_cuda': is_cuda,
        'color': parameters.color,
        'epochs': parameters.epochs,

        **nn_model.hyperparameters(),
        **weight_model.hyperparameters(),
    }

    print(f"Creating Log File...")
    with open(log_file, "a+", encoding="utf-8") as file:
        file.write(json.dumps(parameters.json, sort_keys=True, indent=2) + "\n\n")
        file.write(json.dumps(parameter_log, sort_keys=True, indent=2) + "\n\n")
        file.write(str(nn_model.optimizer) + "\n\n")

        file.write(str(nn_model) + "\n\n")
        file.write(str(weight_model) + "\n\n")
        file.write(torchinfo.summary(nn_model, input_size=input_shape, device=device).__repr__() + "\n\n")
        file.write(torchinfo.summary(weight_model, input_size=(1, 1), device=device).__repr__() + "\n\n")

    if parameters.tensorboard:
        nn_model.tensorboard.tensorboard.add_text("parameter", json.dumps(parameters.json, sort_keys=True, indent=2))

    loss_accuracy = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    print(f"Starting Training...")
    for epoch in range(parameters.epochs):
        if parameters.test_logs:
            break

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

        parameter_log["last_epoch"] = epoch
        with open(log_file, "a+", encoding="utf-8") as file:
            file.write(print_str)

        if train_accuracy < 0.125 and epoch >= 9 and parameters.dataset == torchvision.datasets.CIFAR10:
            break

        if parameters.test_run:
            break

    if parameters.save_data:
        torch.save(str(nn_model), f"{paths.model_data}/{parameter_log['last_epoch']}_str_nn_model")
        torch.save(str(weight_model), f"{paths.model_data}/{parameter_log['last_epoch']}_str_weight_model")

        torch.save(parameters.json, f"{paths.model_data}/{parameter_log['last_epoch']}_parameters_json")
        torch.save(parameter_log, f"{paths.model_data}/{parameter_log['last_epoch']}_parameter_log")
        torch.save(loss_accuracy, f"{paths.model_data}/{parameter_log['last_epoch']}_loss_accuracy")

    if parameters.tensorboard:
        parameter_log["input_shape"] = "_".join([str(x) for x in parameter_log["input_shape"]])
        metric_dict = {
            "train_loss": loss_accuracy["train_loss"][-1],
            "train_accuracy": loss_accuracy["train_accuracy"][-1],
            "test_loss": loss_accuracy["test_loss"][-1],
            "test_accuracy": loss_accuracy["test_accuracy"][-1],
            "min_train_loss": np.min(loss_accuracy["train_loss"]),
            "max_train_accuracy": np.max(loss_accuracy["train_accuracy"]),
            "min_test_loss": np.min(loss_accuracy["test_loss"]),
            "max_test_accuracy": np.max(loss_accuracy["test_accuracy"]),
        }
        nn_model.tensorboard.tensorboard.add_hparams(
            hparam_dict=parameter_log,
            metric_dict=metric_dict
        )

    with open(log_file, "a+", encoding="utf-8") as file:
        file.write("Run Completed Successfully...")
    print()


def this_path():
    return Path(__file__)


def get_parameters(kwargs) -> ConvRunParameters:
    parameters = ConvRunParameters()

    if kwargs["color"].lower() == "true":
        kwargs["color"] = True
    elif kwargs["color"].lower() == "false":
        kwargs["color"] = False
    else:
        raise ValueError("Invalid value for color")

    if kwargs["activation_fn"].lower() == "gelu":
        kwargs["activation_fn"] = ReLUGeLUInterpolation
    elif kwargs["activation_fn"].lower() == "silu":
        kwargs["activation_fn"] = ReLUSiLUInterpolation
    else:
        raise ValueError("Invalid value for activation_fn")

    if kwargs["dataset"].lower() == "cifar10":
        kwargs["dataset"] = torchvision.datasets.CIFAR10
    elif kwargs["dataset"].lower() == "cifar100":
        kwargs["dataset"] = torchvision.datasets.CIFAR100
    else:
        raise ValueError("Invalid value for dataset")
    
    for key, value in kwargs.items():
        if hasattr(parameters, key):
            setattr(parameters, key, value)

    select_class(parameters, 'norm_class', [None, Clamp])
    select_class(parameters, 'precision_class', [None, ReducePrecision])
    select_class(parameters, 'noise_class', [None, GaussianNoise])

    check(parameters, "precision_class", "precision")
    check(parameters, "noise_class", "leakage")

    return parameters


def run_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="CIFAR10")

    parser.add_argument("--num_conv_layer", type=int, default=6)
    parser.add_argument("--num_linear_layer", type=int, default=3)
    parser.add_argument("--activation_fn", type=str, default="gelu")
    parser.add_argument("--activation_i", type=float, default=1.0)
    parser.add_argument("--activation_s", type=float, default=1.0)
    parser.add_argument("--activation_alpha", type=float, default=0.0)
    parser.add_argument("--norm_class", type=str, default=None)
    parser.add_argument("--precision_class", type=str, default=None)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--noise_class", type=str, default=None)
    parser.add_argument("--leakage", type=float, default=None)

    parser.add_argument("--lr", type=float, default=ConvRunParameters.lr)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--color", type=str, default=str(ConvRunParameters.color))

    parser.add_argument("--test_logs", action='store_true')
    parser.set_defaults(test_logs=False)
    parser.add_argument("--test_run", action='store_true')
    parser.set_defaults(test_run=False)
    parser.add_argument("--tensorboard", action='store_true')
    parser.set_defaults(tensorboard=False)
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
