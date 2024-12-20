import argparse
import hashlib
import json
import math
from pathlib import Path

import torch.backends.cudnn
import torchvision
from analogvnn.utils.is_cpu_cuda import is_cpu_cuda
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.dataloaders.load_vision_dataset import load_vision_dataset
from src.fn.cross_entropy_loss_accuracy import cross_entropy_loss_accuracy
from src.fn.data_dirs import data_dirs
from src.nn.ReGLUGeGLUInterpolation import ReGLUGeGLUInterpolation
from src.nn.ReLUGeLUInterpolation import ReLUGeLUInterpolation
from src.nn.ReLUSiLUInterpolation import ReLUSiLUInterpolation
from src.nn.ViT import ViT, ViTRunParameters


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

    if paths.tensorboard.exists():
        for file in paths.tensorboard.iterdir():
            file.unlink()
        paths.tensorboard.rmdir()

    parameters.timestamp = paths.timestamp
    log_file = paths.logs.joinpath(f"{paths.name}_logs.txt")

    print(f"Timestamp: {paths.timestamp}")
    print(f"Storage name: {paths.name}")
    print()

    print(f"Loading Data...")
    transform = []
    if not parameters.color:
        transform.append(transforms.Grayscale())
    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)

    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=parameters.dataset,
        path=paths.dataset,
        batch_size=parameters.batch_size,
        is_cuda=is_cuda,
        train_transform=transform,
        test_transform=transform,
    )

    print(f"Creating Models...")
    parameters.input_shape = input_shape[-1]
    parameters.num_classes = len(classes)
    nn_model = ViT(parameters).to(device)

    loss_function = parameters.loss_function()
    accuracy_function = cross_entropy_loss_accuracy
    parameters.accuracy_function = accuracy_function.__name__
    optimizer = parameters.optimizer(params=nn_model.parameters())
    scheduler = parameters.scheduler(optimizer, parameters.epochs)

    print(f"Creating Log File...")
    with open(log_file, "a+", encoding="utf-8") as file:
        file.write(json.dumps(parameters.json, sort_keys=True, indent=2) + "\n\n")
        file.write(str(nn_model) + "\n\n")

    loss_accuracy = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    print(f"Starting Training...")
    for epoch in range(parameters.epochs):
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
        torch.save(parameters.json, f"{paths.model_data}/{parameters.last_epoch}_parameters_json")
        torch.save(loss_accuracy, f"{paths.model_data}/{parameters.last_epoch}_loss_accuracy")

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

    if kwargs["dataset"].lower() == "cifar10":
        kwargs["dataset"] = torchvision.datasets.CIFAR10
    elif kwargs["dataset"].lower() == "cifar100":
        kwargs["dataset"] = torchvision.datasets.CIFAR100
    else:
        raise ValueError("Invalid value for dataset")

    for key, value in kwargs.items():
        if hasattr(parameters, key):
            setattr(parameters, key, value)

    return parameters


def run_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="CIFAR10")

    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--activation_fn", type=str, default="gelu")
    parser.add_argument("--activation_i", type=float, default=1.0)
    parser.add_argument("--precision", type=int, required=True)
    parser.add_argument("--leakage", type=float, required=True)

    parser.add_argument("--color", type=str, default=str(ViTRunParameters.color))

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
