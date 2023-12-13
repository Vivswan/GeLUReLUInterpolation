import argparse
import copy
import hashlib
import inspect
import itertools
import math
import os
import random
import shutil
import subprocess
import time
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
from pathlib import Path

import torch
import torchvision
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import *
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from natsort import natsorted, ns

from src.run_vit_model import this_path

combination_dict = OrderedDict({
    "color": [False, True],
    "norm_class": [None, Clamp],
    "precision_class": [None, ReducePrecision],
    "noise_class": [None, GaussianNoise],

    "depth": [1, 2, 3, 4, 5, 6],
    "activation_fn": ["gelu", "silu", 'gege'],
    "activation_i": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "precision": [None, 4, 8, 16, 32, 64],
    "leakage": [None, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
})

RUN_LIST = {
    # "gelu_n": "depth:3,norm_class:Clamp,precision_class:ReducePrecision,noise_class:GaussianNoise,activation_fn:gelu",
    "gege_n": "depth:3,norm_class:Clamp,precision_class:ReducePrecision,noise_class:GaussianNoise,activation_fn:gege",
    "silu_n": "depth:3,norm_class:Clamp,precision_class:ReducePrecision,noise_class:GaussianNoise,activation_fn:silu",
    "gelu_d": "leakage:0.8,norm_class:Clamp,precision_class:ReducePrecision,noise_class:GaussianNoise,activation_fn:gelu",
    "silu_d": "leakage:0.8,norm_class:Clamp,precision_class:ReducePrecision,noise_class:GaussianNoise,activation_fn:silu",
}


def prepare_data_folder(folder_path):
    folder_path = Path(folder_path)
    runtime_path = folder_path.joinpath("runtime")
    datasets_path = folder_path.joinpath("datasets")
    models_path = folder_path.joinpath("models")
    tensorboard_path = folder_path.joinpath("tensorboard")
    logs_path = folder_path.joinpath("logs")

    for p in [folder_path, runtime_path, datasets_path, models_path, tensorboard_path, logs_path]:
        if p.exists():
            continue
        os.mkdir(p)

    torchvision.datasets.CIFAR10(root=str(datasets_path.absolute()), download=True)


def run_command(command):
    command, data_folder, run_combination, index = command
    data_folder = Path(data_folder).absolute()
    base_path = data_folder.parent

    print(f"Trying to run {run_combination}::{index}")
    run_check_file = Path(__file__).parent.joinpath(f"_crc_slurm/run_{run_combination}/run_{index}")
    try:
        t = run_check_file.parent.joinpath(f"current_run_{index}.log")
        run_check_file.rename(t)
        run_check_file = t
    except FileNotFoundError:
        return 

    print(f"Running {run_combination}::{index}")
    random_number = random.randint(0, 1000000)
    new_data_folder = base_path.joinpath(f"_result_{run_combination}_{index}_{random_number}")
    shutil.copytree(data_folder, new_data_folder)
    data_folder = new_data_folder

    runtime = Path(data_folder).joinpath("runtime")
    hash_id = hashlib.sha256(str(command).encode("utf-8")).hexdigest()
    timestamp = f"{int(time.time() * 1000)}"

    data_folder = Path(data_folder).absolute()
    command += f" --data_folder {data_folder}"
    if "--timestamp" not in command:
        command += f" --timestamp {timestamp}"
    else:
        timestamp = command.split("--timestamp")[-1]
        timestamp = timestamp.strip().split(" ")[0]

    if "--name" not in command:
        command += f" --name {hash_id}"
    else:
        hash_id = command.split("--name")[-1]
        hash_id = hash_id.strip().split(" ")[0][:8]

    filename = f"{timestamp}_{hash_id}"
    print(f"Running {filename} :: {command}")
    rc = -1
    while rc != 0:
        with open(run_check_file, "w+", encoding="utf-8") as out:
            out.write(command + "\n")
            out.write(f"Running {filename} :: {command}\n\n")

            p = subprocess.Popen(command, shell=True, stdout=out, stderr=out)
            p.wait()
            rc = p.returncode

            out.write(f"\n\n")
            if rc == 0:
                out.write(f"Success {p.pid} :: {filename} :: {command}")
                print(f"Success {p.pid} :: {filename} :: {command}")
            else:
                out.write(f"Failed  {p.pid} :: {filename} :: {rc} :: {command}")
                print(f"Failed  {p.pid} :: {filename} :: {rc} :: {command}")
                
            out.write(f"\n\n{rc}")


    t = run_check_file.parent.joinpath(f"completed_run_{index}.log")
    run_check_file.rename(t)
    run_check_file = t
    print(f"Finished {run_combination}::{index}")
    runtime.joinpath(f"{filename}.log").write_text(run_check_file.read_text(encoding="utf-8"), encoding="utf-8")
    targz = base_path.joinpath(f"{data_folder.name}.tar.gz")

    shutil.rmtree(data_folder.joinpath("datasets"))
    shutil.make_archive(data_folder, 'gztar', data_folder)
    shutil.move(targz, Path(f"~/storage/{targz.name}").expanduser())
    shutil.rmtree(data_folder)
    targz.unlink()
    
    run_check_file.unlink()

def create_command_list(extra_arg="", select=""):
    cd_copy = copy.deepcopy(combination_dict)
    if len(select) > 0:
        for parameter in select.split(","):
            parameter = parameter.strip()
            key, value = parameter.split(":")
            not_in = value[0] == "~"
            if not_in:
                value = value[1:]

            values = cd_copy[key]
            cd_copy[key] = []
            for v in values:
                if inspect.isclass(v):
                    v = v.__name__
                if value == str(v) and not not_in:
                    cd_copy[key].append(v)
                if value != str(v) and not_in:
                    cd_copy[key].append(v)

    if cd_copy["precision_class"] == [None]:
        cd_copy["precision"] = [None]
    if cd_copy["noise_class"] == [None]:
        cd_copy["leakage"] = [None]

    combinations = list(itertools.product(*list(cd_copy.values())))
    command_list = []
    for c in combinations:
        command_dict = dict(zip(list(cd_copy.keys()), c))

        if (command_dict["noise_class"] is None) != (command_dict["leakage"] is None):
            continue
        if (command_dict["precision_class"] is None) != (command_dict["precision"] is None):
            continue

        path = this_path().relative_to(Path(__file__).parent).__str__()
        command_str = f'python {path}'
        for key, value in command_dict.items():
            if value is None:
                continue
            if inspect.isclass(value):
                command_str += f' --{key} "{value.__name__}"'
            elif isinstance(value, str):
                command_str += f' --{key} "{value}"'
            else:
                command_str += f' --{key} {value}'

        command_str += f" {extra_arg}"
        command_list.append(command_str.strip())
    command_list = natsorted(command_list, alg=ns.IGNORECASE)
    return command_list


def run_combination_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--run_combination", type=str)
    parser.add_argument("--memory_required", type=float)
    parser.set_defaults(single_run=False)
    me = parser.add_mutually_exclusive_group(required=True)
    me.add_argument("--create", action='store_true')
    me.add_argument("--run", action='store_true')
    me.add_argument("--single_run", action='store_true')
    all_arguments = parser.parse_known_args()

    if all_arguments[0].create:
        return create_slurm_scripts()

    kwargs = vars(all_arguments[0])
    extra_arg = ""
    for i in all_arguments[1]:
        extra_arg += f' "{i}"' if " " in i else f' {i}'

    data_folder = Path(kwargs['data_folder']).absolute()
    cuda_mem = torch.cuda.mem_get_info()[1] / 1024 / 1024 / 1024 if torch.cuda.is_available() else 0
    num_process = max(math.floor(cuda_mem / kwargs['memory_required']), 1)
    if kwargs['single_run']:
        num_process = 1

    print(f"data_folder: {data_folder}")
    print(f"run_combination: {kwargs['run_combination']}")
    print(f"cuda_mem: {cuda_mem:.2f}GB")
    prepare_data_folder(data_folder)

    command_list = create_command_list(extra_arg, RUN_LIST[kwargs['run_combination']])
    command_list = [(x, str(data_folder), kwargs['run_combination'], i) for i, x in enumerate(command_list)]
    random.shuffle(command_list)
    
    with ThreadPool(num_process) as pool:
        pool.map(run_command, command_list)


def create_slurm_scripts():
    output_path = Path(__file__).parent.joinpath("_crc_slurm")
    if output_path.exists():
        for f in output_path.iterdir():
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)
    output_path.mkdir(parents=True, exist_ok=True)

    template_file = Path(__file__).parent.joinpath("template/run_template.slurm").read_text(encoding="utf-8")
    for i in RUN_LIST:
        output_path.joinpath(f"run_{i}.slurm").write_text(
            template_file
            .replace("@@@RunScript@@@", Path(__file__).name.split(".")[0])
            .replace("@@@memory_required@@@", "9")
            .replace("@@@run_combination@@@", i),
            encoding="utf-8"
        )
        runs_folder = output_path.joinpath(f"run_{i}")
        runs_folder.mkdir(parents=True, exist_ok=True)
        size = len(create_command_list('', RUN_LIST[i]))
        print(f"{i}: {size}, {size * 5 / 60 / 100}")
        for j in range(0, size):
            runs_folder.joinpath(f"run_{j}").write_text("")


if __name__ == '__main__':
    run_combination_main()
