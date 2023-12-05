import argparse
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Optional, Tuple

import numpy as np
import torch.backends.cudnn
from analogvnn.nn.module.Layer import Layer
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import Clamp
from analogvnn.nn.normalize.Normalize import Normalize
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from analogvnn.parameter.PseudoParameter import PseudoParameter
from analogvnn.utils.is_cpu_cuda import is_cpu_cuda
from torch import optim, nn, Tensor
from torch.optim import Optimizer

from src.dataloaders.load_text_dataset import load_wikitext2_dataset
from src.fn.data_dirs import data_dirs
from src.fn.misc import select_class, check
from src.nn.ReLUGeLUInterpolation import ReLUGeLUInterpolation
from src.nn.ReLUSiLUInterpolation import ReLUSiLUInterpolation
from src.nn.TransformerModel import TransformerModel
from src.nn.WeightModel import WeightModel


@dataclass
class TransformerRunParameters:
    name: Optional[str] = None
    data_folder: Optional[str] = None

    num_transformer_layers: int = 2
    embedding_dim: int = 256
    dim_feedforward: int = 256
    num_heads: int = 4
    dropout: float = 0.2
    activation_fn: Type[Layer] = ReLUGeLUInterpolation
    activation_i: float = 0
    activation_s: float = 1
    activation_alpha: float = 0
    norm_class: Optional[Type[Normalize]] = None
    precision_class: Type[Layer] = None
    precision: Optional[int] = None
    noise_class: Type[Layer] = None
    leakage: Optional[float] = None

    color: bool = True
    batch_size: int = 20
    epochs: int = 10
    bptt: int = 35

    device: Optional[torch.device] = None
    test_logs: bool = False
    test_run: bool = False
    tensorboard: bool = False
    save_data: bool = True
    timestamp: str = None

    @property
    def nn_model_params(self):
        return {
            "num_transformer_layers": self.num_transformer_layers,
            "embedding_dim": self.embedding_dim,
            "dim_feedforward": self.dim_feedforward,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
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


def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


def train_on(
        model: nn.Module,
        train_data,
        bptt: int,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler,
        ntokens: int,
        epoch: int,
        test_run: bool = False,
):
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200

    num_batches = len(train_data) // bptt
    cur_loss = 0
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)

        model.zero_grad()
        optimizer.zero_grad()

        output = model(data)
        output_flat = output.view(-1, ntokens)
        loss = criterion(output_flat, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        cur_loss += loss.item()
        total_loss += loss.item() * len(data)
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            cur_loss = cur_loss / log_interval
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | lr {lr:02.2f} | loss {cur_loss:5.4f}')
            cur_loss = 0

        if test_run:
            break

    lr = scheduler.get_last_lr()[0]
    loss = total_loss / train_data.size(0)
    return lr, loss


def test_on(
        model: nn.Module,
        eval_data,
        bptt: int,
        criterion: nn.Module,
        ntokens: int,
):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


def run_model(parameters: TransformerRunParameters):
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
    train_data, val_data, test_data, num_tokens = load_wikitext2_dataset(paths.dataset, parameters.batch_size)

    nn_model_params = parameters.nn_model_params
    weight_model_params = parameters.weight_model_params
    nn_model_params["num_tokens"] = num_tokens
    nn_model_params["device"] = device

    print(f"Creating Models...")
    nn_model = TransformerModel(**nn_model_params)
    weight_model = WeightModel(**weight_model_params)
    if parameters.tensorboard:
        weight_model.create_tensorboard(paths.tensorboard)

    PseudoParameter.parametrize_module(nn_model, transformation=weight_model)
    # for i in nn_model.modules():
    #     if i.__class__ == nn.Linear:
    #         PseudoParameter.parametrize_module(i, transformation=weight_model)
    weight_model.compile(device=device)
    nn_model = nn_model.to(device)
    train_data = train_data.to(device)
    test_data = test_data.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=nn_model.parameters(), lr=5.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

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
        weight_model.tensorboard.tensorboard.add_text("parameter",
                                                      json.dumps(parameters.json, sort_keys=True, indent=2))

    loss_accuracy = {
        "lr": [],
        "train_loss": [],
        "test_loss": [],
    }

    print(f"Starting Training...")
    for epoch in range(parameters.epochs):
        if parameters.test_logs:
            break

        lr, train_loss = train_on(
            model=nn_model,
            train_data=train_data,
            bptt=parameters.bptt,
            criterion=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            ntokens=num_tokens,
            epoch=epoch,
            test_run=parameters.test_run,
        )
        test_loss = test_on(
            model=nn_model,
            eval_data=test_data,
            bptt=parameters.bptt,
            criterion=loss_function,
            ntokens=num_tokens,
        )

        loss_accuracy["lr"].append(lr)
        loss_accuracy["train_loss"].append(train_loss)
        loss_accuracy["test_loss"].append(test_loss)

        str_epoch = str(epoch + 1).zfill(math.ceil(math.log10(parameters.epochs)))
        print_str = f'({str_epoch}) Training Loss: {train_loss:.4f}, Testing Loss: {test_loss:.4f}\n'
        print(print_str)

        parameter_log["last_epoch"] = epoch
        with open(log_file, "a+", encoding="utf-8") as file:
            file.write(print_str)

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
            "test_loss": loss_accuracy["test_loss"][-1],
            "max_train_loss": np.min(loss_accuracy["train_loss"]),
            "min_test_loss": np.min(loss_accuracy["test_loss"]),
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


def get_parameters(kwargs) -> TransformerRunParameters:
    parameters = TransformerRunParameters()

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

    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--activation_fn", type=str, default="gelu")
    parser.add_argument("--activation_i", type=float, default=1.0)
    parser.add_argument("--norm_class", type=str, default=None)
    parser.add_argument("--precision_class", type=str, default=None)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--noise_class", type=str, default=None)
    parser.add_argument("--leakage", type=float, default=None)

    parser.add_argument("--device", type=str, default=None)
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
