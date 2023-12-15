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
import torchvision
from analogvnn.nn.module.Layer import Layer
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import Clamp
from analogvnn.nn.normalize.Normalize import Normalize
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from analogvnn.utils.is_cpu_cuda import is_cpu_cuda
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from src.dataloaders.load_vision_dataset import load_vision_dataset
from src.fn.cross_entropy_loss_accuracy import cross_entropy_loss_accuracy
from src.fn.data_dirs import data_dirs
from src.fn.misc import select_class, check
from src.nn.ReGLUGeGLUInterpolation import ReGLUGeGLUInterpolation
from src.nn.ReLUGeLUInterpolation import ReLUGeLUInterpolation
from src.nn.ReLUSiLUInterpolation import ReLUSiLUInterpolation
from src.nn.ViT import ViT
from src.nn.WeightModel import WeightModel


@dataclass
class ViTRunParameters:
    name: Optional[str] = None
    data_folder: Optional[str] = None

    patch_size: int = 4
    dim: int = 512
    depth: int = 6
    heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.5
    emb_dropout: float = 0.5

    activation_fn: Type[Layer] = ReLUGeLUInterpolation
    activation_i: float = 0.0
    activation_s: float = 1.0
    activation_alpha: float = 0.0
    norm_class: Optional[Type[Normalize]] = None
    precision_class: Type[Layer] = None
    precision: Optional[int] = None
    noise_class: Type[Layer] = None
    leakage: Optional[float] = None

    dataset: Type[VisionDataset] = torchvision.datasets.CIFAR10
    color: bool = True
    batch_size: int = 512
    epochs: int = 150

    device: Optional[torch.device] = None
    test_logs: bool = False
    test_run: bool = False
    tensorboard: bool = False
    save_data: bool = True
    timestamp: str = None

    @property
    def nn_model_params(self):
        return {
            "patch_size": self.patch_size,
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
            "emb_dropout": self.emb_dropout,

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


def train_on(
        model: ViT,
        train_loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        accuracy_function: callable,
        optimizer: Optimizer,
        epoch: int,
        device: torch.device,
        test_run: bool,
):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_size = 0

    if isinstance(train_loader, DataLoader):
        # noinspection PyTypeChecker
        dataset_size = len(train_loader.dataset)
    else:
        dataset_size = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accuracy = accuracy_function(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(inputs)
        total_accuracy += accuracy * len(inputs)
        total_size += len(inputs)

        print_mod = int(dataset_size / (len(inputs) * 5))
        if print_mod > 0 and batch_idx % print_mod == 0 and batch_idx > 0:
            print(
                f'Train Epoch:'
                f' {((epoch + 1) if epoch is not None else "")}'
                f' [{batch_idx * len(inputs)}/{dataset_size} ({100. * batch_idx / len(train_loader):.0f}%)]'
                f'\tLoss: {total_loss / total_size:.6f}'
                f'\tAccuracy: {total_accuracy / total_size * 100:.2f}%'
            )
        if test_run:
            break

    total_loss /= total_size
    total_accuracy /= total_size
    return total_loss, total_accuracy


def test_on(
        model: ViT,
        testloader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        accuracy_function: callable,
        device: torch.device,
        test_run: bool,
):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_size = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accuracy = accuracy_function(outputs, targets)

            total_loss += loss.item() * len(inputs)
            total_accuracy += accuracy * len(inputs)
            total_size += len(inputs)
            if test_run:
                break

    total_loss /= total_size
    total_accuracy /= total_size
    return total_loss, total_accuracy


def run_model(parameters: ViTRunParameters):
    torch.backends.cudnn.benchmark = True
    is_cpu_cuda.use_cuda_if_available()

    if parameters.device is not None:
        is_cpu_cuda.set_device(str(parameters.device))
    device, is_cuda = is_cpu_cuda.is_using_cuda
    parameters.device = device

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
    nn_model_params["image_size"] = input_shape[-1]
    nn_model_params["num_classes"] = len(classes)

    print(f"Creating Models...")
    nn_model = ViT(**nn_model_params)
    weight_model = WeightModel(**weight_model_params)
    if parameters.tensorboard:
        weight_model.create_tensorboard(paths.tensorboard)

    weight_model.compile(device=device)
    nn_model = nn_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    accuracy_function = cross_entropy_loss_accuracy
    optimizer = optim.Adam(params=nn_model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, parameters.epochs)

    parameter_log = {
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
        file.write(str(nn_model) + "\n\n")
        file.write(str(weight_model) + "\n\n")

    if parameters.tensorboard:
        weight_model.tensorboard.tensorboard.add_text(
            "parameter",
            json.dumps(parameters.json, sort_keys=True, indent=2)
        )

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

        train_loss, train_accuracy = train_on(
            model=nn_model,
            train_loader=train_loader,
            criterion=loss_function,
            accuracy_function=accuracy_function,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            test_run=parameters.test_run,
        )
        test_loss, test_accuracy = test_on(
            model=nn_model,
            testloader=test_loader,
            accuracy_function=accuracy_function,
            criterion=loss_function,
            device=device,
            test_run=parameters.test_run,
        )
        scheduler.step()

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

    if parameters.tensorboard:
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
        weight_model.tensorboard.tensorboard.add_hparams(
            hparam_dict=parameter_log,
            metric_dict=metric_dict
        )

    with open(log_file, "a+", encoding="utf-8") as file:
        file.write("Run Completed Successfully...")
    print()


def this_path():
    return Path(__file__)


def get_parameters(kwargs) -> ViTRunParameters:
    parameters = ViTRunParameters()

    if kwargs["activation_fn"].lower() == "gelu":
        kwargs["activation_fn"] = ReLUGeLUInterpolation
    elif kwargs["activation_fn"].lower() == "silu":
        kwargs["activation_fn"] = ReLUSiLUInterpolation
    elif kwargs["activation_fn"].lower() == "gege":
        kwargs["activation_fn"] = ReGLUGeGLUInterpolation
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

    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--activation_fn", type=str, default="gelu")
    parser.add_argument("--activation_i", type=float, default=1.0)
    parser.add_argument("--norm_class", type=str, default=None)
    parser.add_argument("--precision_class", type=str, default=None)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--noise_class", type=str, default=None)
    parser.add_argument("--leakage", type=float, default=None)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--color", type=str, default=str(ViTRunParameters.color))

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
