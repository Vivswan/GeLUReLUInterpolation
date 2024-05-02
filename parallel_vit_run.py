import argparse
import copy
import hashlib
import inspect
import itertools
import os
import shutil
import subprocess
import time
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
from pathlib import Path

import torchvision
from natsort import natsorted, ns

from src.run_vit_model import this_path

combination_dict = OrderedDict({
    "color": [True, False],

    "depth": [1, 2, 3, 4, 5, 6],
    "activation_fn": ["gelu", "silu", "gege"],
    "activation_i": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "precision": [4, 8, 16, 32, 64],
    "leakage": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],

    "dataset": ["cifar10", "cifar100"],
})

RUN_LIST = {
    # "gelu_d": "leakage:0.8,activation_fn:gelu,dataset:cifar10",
    # "silu_d": "leakage:0.8,activation_fn:silu,dataset:cifar10",

    # "gelu_4n": "depth:4,activation_fn:gelu,dataset:cifar10",
    # "silu_4n": "depth:4,activation_fn:silu,dataset:cifar10",
    # "gege_4n": "depth:4,activation_fn:gege,dataset:cifar10",

    # "c100_gelu_d": "leakage:0.8,activation_fn:gelu,dataset:cifar100,precision:64,color:True",
    # "c100_gelu_4n": "depth:4,activation_fn:gelu,dataset:cifar100,precision:64,color:True",
    "c100_gelu_2n": "depth:2,activation_fn:gelu,dataset:cifar100,precision:64,color:True",
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
    torchvision.datasets.CIFAR100(root=str(datasets_path.absolute()), download=True)


def run_command(command):
    data_folder, command = command
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
        hash_id = hash_id.strip().split(" ")[0]

    filename = f"{timestamp}_{hash_id}"
    out_file = runtime.joinpath(f"{filename}.log")

    print(f"Running {filename} :: {command}")
    with open(out_file, "w+", encoding="utf-8") as out:
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

    combinations = list(itertools.product(*list(cd_copy.values())))
    command_list = []

    path = this_path().relative_to(Path(__file__).parent).__str__()
    for c in combinations:
        command_dict = dict(zip(list(cd_copy.keys()), c))

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
    parser.add_argument("--run_index", type=int, default=-1)
    parser.add_argument("--single_run", action='store_true')
    parser.set_defaults(single_run=False)
    parser.add_argument("--create", action='store_true')
    parser.set_defaults(create=False)
    all_arguments = parser.parse_known_args()

    if all_arguments[0].create:
        create_slurm_scripts()
        for name, value in RUN_LIST.items():
            size = len(create_command_list('', value))
            print(f"{name}: {size}, {size * 5 / 60 / 100}")
        print()
        return

    print(all_arguments)
    kwargs = vars(all_arguments[0])
    extra_arg = ""
    for i in all_arguments[1]:
        if " " in i:
            extra_arg += f' "{i}"'
        else:
            extra_arg += f' {i}'

    print(f"data_folder: {kwargs['data_folder']}")
    print(f"run_combination: {kwargs['run_combination']}")
    print(f"run_index: {kwargs['run_index']}")
    prepare_data_folder(kwargs['data_folder'])

    if kwargs['run_index'] < 1:
        raise Exception("run_index must be >= 1")

    command_list = create_command_list(extra_arg, RUN_LIST[kwargs['run_combination']])
    command_list = [command_list[kwargs['run_index'] - 1]]
    command_list = [(kwargs['data_folder'], x) for x in command_list]

    with ThreadPool(1) as pool:
        pool.map(run_command, command_list)


def create_slurm_scripts():
    output_path = Path(__file__).parent.joinpath("_crc_slurm")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()

    template_file = Path(__file__).parent.joinpath("template/run_array_template.slurm").read_text(encoding="utf-8")
    for i in RUN_LIST:
        end = len(create_command_list('', RUN_LIST[i]))
        for j in range(1, end + 2, 900):
            r_start = j
            r_end = min(j + 900 - 1, end)
            if r_end <= r_start:
                continue
            with open(f"_crc_slurm/run_{i}_{j}.slurm", "w", encoding="utf-8") as slurm_file:
                slurm_file.write(
                    template_file
                    .replace("@@@RunScript@@@", Path(__file__).name.split(".")[0])
                    .replace("@@@run_combination@@@", i)
                    .replace("@@@array@@@", f"{0}-{r_end - r_start}")
                    .replace("@@@ArrayTaskIdOffset@@@", f"{r_start}")
                )


if __name__ == '__main__':
    run_combination_main()
