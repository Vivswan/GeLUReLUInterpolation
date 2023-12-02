import argparse
import json
from pathlib import Path

from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import *
from analogvnn.nn.precision.ReducePrecision import ReducePrecision

from src.nn.ReLUGeLUInterpolation import ReLUGeLUInterpolation
from src.nn.ReLUSiLUInterpolation import ReLUSiLUInterpolation
from src.run_model import run_model, RunParameters


def this_path():
    return Path(__file__)


def __select_class(main_object, name, class_list):
    value = getattr(main_object, name)
    if value is None and None in class_list:
        setattr(main_object, name, None)
        return None
    if value is None and None not in class_list:
        raise Exception(f"{name} must be in {class_list}")

    for cl in class_list:
        if cl is None:
            continue
        if value == cl:
            setattr(main_object, name, cl)
            return cl
        if isinstance(value, str):
            if value == cl.__name__:
                setattr(main_object, name, cl)
                return cl

    raise Exception(f"{name} must be in {class_list}")


def __check(main_obj, first, second):
    if getattr(main_obj, first) is None:
        if getattr(main_obj, second) is not None:
            raise Exception(f'{first}=None then {second} must be None')
    else:
        if getattr(main_obj, second) is None:
            raise Exception(f'{second} must not be None')


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

    __select_class(parameters, 'norm_class', [None, Clamp])
    __select_class(parameters, 'precision_class', [None, ReducePrecision])
    __select_class(parameters, 'noise_class', [None, GaussianNoise])

    __check(parameters, "precision_class", "precision")
    __check(parameters, "noise_class", "leakage")

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
    parser.add_argument("--activation_alpha", type=float, default=0)
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
