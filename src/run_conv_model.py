import argparse
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Optional

import numpy as np
import torch.backends.cudnn
import torchinfo
import torchvision
from analogvnn.nn.module.Layer import Layer
from analogvnn.nn.normalize.Clamp import Clamp
from analogvnn.nn.normalize.Normalize import Normalize
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from analogvnn.parameter.PseudoParameter import PseudoParameter
from analogvnn.utils.is_cpu_cuda import is_cpu_cuda
from keras.src.layers import GaussianNoise
from torch import optim, nn
from torchvision.datasets import VisionDataset

from src.dataloaders.load_vision_dataset import load_vision_dataset
from src.fn.cross_entropy_loss_accuracy import cross_entropy_loss_accuracy
from src.fn.data_dirs import data_dirs
from src.fn.misc import select_class, check
from src.nn.ConvModel import ConvModel
from src.nn.ReLUGeLUInterpolation import ReLUGeLUInterpolation
from src.nn.ReLUSiLUInterpolation import ReLUSiLUInterpolation
from src.nn.WeightModel import WeightModel


@dataclass
class RunParameters:
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
        return f"RunParameters({json.dumps(self.json)})"


def run_model(parameters: RunParameters):
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
    nn_model.optimizer = optim.Adam(params=nn_model.parameters())

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

        if train_accuracy < 0.125 and epoch >= 9:
            break

        if parameters.test_run:
            break

    if parameters.save_data:
        torch.save(str(nn_model), f"{paths.model_data}/{parameter_log['last_epoch']}_str_nn_model")
        torch.save(str(weight_model), f"{paths.model_data}/{parameter_log['last_epoch']}_str_weight_model")

        torch.save(parameters.json, f"{paths.model_data}/{parameter_log['last_epoch']}_parameters_json")
        torch.save(parameter_log, f"{paths.model_data}/{parameter_log['last_epoch']}_parameter_log")
        torch.save(loss_accuracy, f"{paths.model_data}/{parameter_log['last_epoch']}_loss_accuracy")

        torch.save(nn_model.hyperparameters(),
                   f"{paths.model_data}/{parameter_log['last_epoch']}_hyperparameters_nn_model")
        torch.save(weight_model.hyperparameters(),
                   f"{paths.model_data}/{parameter_log['last_epoch']}_hyperparameters_weight_model")

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


def get_parameters(kwargs) -> RunParameters:
    parameters = RunParameters()

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

    parser.add_argument("--num_conv_layer", type=int, default=6)
    parser.add_argument("--num_linear_layer", type=int, default=3)
    parser.add_argument("--activation_fn", type=str, default="gelu")
    parser.add_argument("--activation_i", type=float, default=1.0)
    parser.add_argument("--activation_s", type=float, default=1.0)
    parser.add_argument("--norm_class", type=str, default=None)
    parser.add_argument("--precision_class", type=str, default=None)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--noise_class", type=str, default=None)
    parser.add_argument("--leakage", type=float, default=None)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--color", type=str, default=str(RunParameters.color))

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
