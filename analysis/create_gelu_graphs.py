import itertools
import json
import math
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.colors
import numpy as np
import seaborn
import torch
from analogvnn.nn.activation.Gaussian import GeLU
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import Clamp
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from matplotlib import pyplot as plt
from seaborn.palettes import _color_to_rgb, _ColorPalette
from tqdm import tqdm

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def main_color_palette(n_colors=6, as_cmap=False):  # noqa
    if as_cmap:
        n_colors = 256

    hues = np.linspace(130, -115, int(n_colors)) % 360
    saturation = np.linspace(1, 1, int(n_colors)) * 99
    lightness = np.linspace(0.85, 0.3, int(n_colors)) * 99

    palette = [
        _color_to_rgb((h_i, s_i, l_i), input="husl")
        for h_i, s_i, l_i in zip(hues, saturation, lightness)
    ]
    palette = list(reversed(palette))
    if as_cmap:
        return matplotlib.colors.ListedColormap(palette, "hsl")
    else:
        return _ColorPalette(palette)


def to_title_case(string: str):
    string = string.split(".")[-1]
    string = [(x[0].upper() + x[1:].lower()) for x in string.split("_")]
    string = " ".join(string)
    if string.split(" ")[0] == "Std":
        string = " ".join(["σ", *string.split(" ")[1:]])
    string = string.replace(" W", "").replace(" Y", "")
    return string.replace('Leakage', 'Error Probability')


def apply_if_not_none(value, func):
    if value is None:
        return value
    return func(value)


def sanitise_data(data):
    data["train_loss"] = data["loss_accuracy"]["train_loss"][-1] * 100
    data["train_accuracy"] = data["loss_accuracy"]["train_accuracy"][-1] * 100
    data["test_loss"] = data["loss_accuracy"]["test_loss"][-1] * 100
    data["test_accuracy"] = data["loss_accuracy"]["test_accuracy"][-1] * 100
    data["min_train_loss"] = np.min(data["loss_accuracy"]["train_loss"][-1]) * 100
    data["max_train_accuracy"] = np.max(data["loss_accuracy"]["train_accuracy"][-1]) * 100
    data["min_test_loss"] = np.min(data["loss_accuracy"]["test_loss"][-1]) * 100
    data["max_test_accuracy"] = np.max(data["loss_accuracy"]["test_accuracy"][-1]) * 100

    py = data["hyperparameters_nn_model"]["precision_y"]
    pw = data["hyperparameters_weight_model"]["precision_w"]
    data["bit_precision_y"] = 32.0 if py is None else math.log2(py)
    data["bit_precision_w"] = 32.0 if pw is None else math.log2(pw)

    if data["parameter_log"]["precision_class_w"] == 'None':
        data["parameter_log"]["precision_class_w"] = "Digital"
    if data["parameter_log"]["precision_class_y"] == 'None':
        data["parameter_log"]["precision_class_y"] = "Digital"

    if data["parameter_log"]["precision_y"] is not None \
            and data["parameter_log"]["leakage_y"] is not None:
        data["std_y"] = GaussianNoise.calc_std(
            data["parameter_log"]["leakage_y"],
            data["parameter_log"]["precision_y"]
        )

    if data["parameter_log"]["precision_w"] is not None \
            and data["parameter_log"]["leakage_w"] is not None:
        data["std_w"] = GaussianNoise.calc_std(
            data["parameter_log"]["leakage_w"],
            data["parameter_log"]["precision_w"]
        )

    return data


def get_combined_data(data_path):
    data_path = Path(data_path)
    if data_path.is_file():
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.loads(file.read())
        return data

    data = {}
    dp = list(data_path.iterdir())
    for i in tqdm(dp, desc="Loading Data", ascii=True):
        data[i.name] = {}
        for j in i.iterdir():
            data[i.name][j.name[j.name.index("_") + 1:]] = torch.load(j)
        if data[i.name] == {}:
            del data[i.name]
            print(f"Empty: {i.name}")
            continue
        data[i.name] = sanitise_data(data[i.name])
    return data


def compile_data(data_path, name=None):
    data_path = Path(data_path)
    run_data = get_combined_data(data_path / "models")
    name = data_path.name if name is None else name
    torch.save(run_data, data_path.parent.joinpath(name + ".pt"))


def get_key(obj, key):
    key = key.split(".")
    for i in key:
        obj = obj[i]
    return obj


def get_filtered_data(data, filters):
    if filters is None or len(filters.keys()) == 0:
        return data

    filtered_data = {}

    for key, value in data.items():
        check = True
        for filter_key, filter_value in filters.items():
            if isinstance(filter_value, str):
                if not any([get_key(data[key], filter_key) == i for i in filter_value.split("|")]):
                    check = False
                    break
            else:
                if get_key(data[key], filter_key) != filter_value:
                    check = False
                    break

        if check:
            filtered_data[key] = value

    return filtered_data


def get_plot_data(data_path, x_axis, y_axis, subsection=None, colorbar=None, filters=None, add_data=None):
    if add_data is None:
        add_data = {}

    plot_labels = {}
    plot_data = {}
    add_data["x"] = x_axis
    add_data["y"] = y_axis
    add_data["hue"] = subsection
    add_data["style"] = colorbar

    run_data = torch.load(data_path)
    run_data = get_filtered_data(run_data, filters)

    for key, value in add_data.items():
        if value is None:
            continue

        plot_labels[key] = value
        plot_data[key] = []

    for key, value in run_data.items():
        for i in plot_labels:
            plot_data[i].append(get_key(run_data[key], plot_labels[i]))

    if colorbar is None:
        if subsection is not None:
            plot_data["hue_order"] = sorted(list(set(plot_data["hue"])))
            if "Digital" in plot_data["hue_order"]:
                plot_data["hue_order"].remove("Digital")
                plot_data["hue_order"].insert(0, "Digital")
            if "None" in plot_data["hue_order"]:
                plot_data["hue_order"].remove("None")
                plot_data["hue_order"].insert(0, "None")
    else:
        if "hue" not in plot_data:
            plot_data["hue"] = plot_data["style"]
            del plot_data["style"]
        else:
            plot_data["hue"], plot_data["style"] = plot_data["style"], plot_data["hue"]

    zip_list = ["x", "y"]
    if "hue" in plot_data:
        zip_list.append("hue")
    if "style" in plot_data:
        zip_list.append("style")

    if isinstance(plot_data["x"][0], str):
        ziped_list = list(zip(*[plot_data[x] for x in zip_list]))
        ziped_list = sorted(ziped_list, key=lambda tup: -np.sum(np.array(tup[0]) == "None"))
        unziped_list = list(zip(*ziped_list))

        for i, v in enumerate(zip_list):
            plot_data[v] = list(unziped_list[i])
    return plot_data


def pick_max_from_plot_data(plot_data, max_keys, max_value):
    max_keys_value = []
    for i in max_keys:
        max_keys_value.append(list(set(plot_data[i])))

    max_accuracies = {i: 0 for i in list(itertools.product(*max_keys_value))}
    for index, value in enumerate(plot_data[max_value]):
        index_max_key_values = tuple([plot_data[i][index] for i in max_keys])

        if max_accuracies[index_max_key_values] < value:
            max_accuracies[index_max_key_values] = value

    plot_data[max_value] = []
    for i in max_keys:
        plot_data[i] = []

    max_accuracies = sorted(max_accuracies.items(), key=lambda tup: tup[0])
    max_accuracies = sorted(max_accuracies, key=lambda tup: -np.sum(np.array(tup[0]) == "None"))

    for key, value in max_accuracies:
        for index, val in enumerate(max_keys):
            plot_data[max_keys[index]].append(key[index])

        plot_data[max_value].append(value)
    return plot_data


def pre_plot(size_factor):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.75)

    # fig_size = [3.25, 1.85]
    # fig_size = [2.00, 1.75]
    fig_size = 1.75

    fig_size = tuple((np.array(fig_size) * np.array(size_factor)).tolist())
    fig.set_size_inches(*fig_size)
    fig.set_dpi(200)

    return fig


def post_plot(plot_data, y_lim=(10, 90)):
    x_axis_title = to_title_case(plot_data["x_axis"])
    y_axis_title = to_title_case(plot_data["y_axis"])
    filter_text = ""

    if plot_data["filters"] is not None:
        filter_text = " {" + "-".join(
            [f"{to_title_case(key)}={value}" for key, value in plot_data["filters"].items()]) + "}"
        filter_text = filter_text.replace("|", " or ")
        # plt.title(f"Filters = {filter_text}")

    # if y_lim is not None:
    #     plt.yticks(np.arange(*y_lim, 25))
    #     plt.ylim(y_lim)
    plt.xlabel(x_axis_title)
    plt.ylabel((plot_data["y_prefix"] if "y_prefix" in plot_data else "") + y_axis_title)

    if plot_data["subsection"] is not None:
        if "g" in plot_data:
            h, l = plot_data["g"].get_legend_handles_labels()

            if plot_data["colorbar"] is None:
                subsection_len = len(set(plot_data["hue"]))
            else:
                subsection_len = len(set(plot_data["style"]))

            plt.legend(h[-subsection_len:], l[-subsection_len:], title=to_title_case(plot_data["subsection"]))
        else:
            plt.legend(title=to_title_case(plot_data["subsection"]))
    elif plot_data["colorbar"] is not None:
        plt.legend(title=to_title_case(plot_data["colorbar"]))

    plt.legend().remove()
    plot_data["fig"].tight_layout()
    # plt.show()

    if isinstance(plot_data["data_path"], list):
        run_name = "".join([Path(i).name for i in plot_data["data_path"]])
    else:
        run_name = Path(plot_data["data_path"]).name[:Path(plot_data["data_path"]).name.index(".")]

    subsection_text = "" if plot_data["subsection"] is None else f" #{to_title_case(plot_data['subsection'])}"
    colorbar_text = "" if plot_data["colorbar"] is None else f" #{to_title_case(plot_data['colorbar'])}"

    name = f"{plot_data['prefix']} - {run_name} - {x_axis_title} vs {y_axis_title}{filter_text}{colorbar_text}{subsection_text}"

    # plot_data["fig"].savefig(f'{location}/{name}.pdf', dpi=plot_data["fig"].dpi, transparent=True)
    # plot_data["fig"].savefig(f'{location}/{name}.svg', dpi=plot_data["fig"].dpi, transparent=True)
    plot_data["fig"].savefig(f'{location}/{name}.png', dpi=plot_data["fig"].dpi, transparent=True)

    plt.close('all')


def create_violin_figure(data_path, x_axis, y_axis, subsection=None, colorbar=None, filters=None,
                         size_factor: Tuple[float, float] = 2, color_by=None, name=None, min_vmin=None, max_vmax=None):
    if filters is None:
        filters = {}
    if colorbar is not None:
        subsection = colorbar

    plot_data = get_plot_data(data_path, x_axis, y_axis, subsection=subsection, filters=filters)

    fig = pre_plot(size_factor)

    n_colors = None
    n_colors = len(plot_data["hue_order"]) if ("hue" in plot_data and n_colors is None) else n_colors
    n_colors = len(set(plot_data["x"])) if n_colors is None else n_colors
    color_map = main_color_palette(n_colors=n_colors)
    g = seaborn.violinplot(**plot_data, cut=0, palette=color_map, inner=None, linewidth=0.1)
    color_map = main_color_palette(n_colors=n_colors)

    gs = seaborn.stripplot(**plot_data, palette=color_map, linewidth=0.1, size=3, jitter=1 / 10, dodge=True)
    if colorbar is not None:
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("hsl", color_map)
        if max_vmax is None:
            max_vmax = max(plot_data["hue"])
        if min_vmin is None:
            min_vmin = min(plot_data["hue"])
        norm = matplotlib.colors.Normalize(vmin=min_vmin, vmax=max_vmax)
        scalar_map = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        cbar = plt.colorbar(scalar_map)
        cbar.ax.set_ylabel(to_title_case(colorbar))

    plot_data["data_path"] = data_path
    plot_data["prefix"] = "v"
    plot_data["fig"] = fig
    plot_data["g"] = g
    plot_data["gs"] = gs
    plot_data["x_axis"] = x_axis
    plot_data["y_axis"] = y_axis
    plot_data["subsection"] = subsection
    plot_data["colorbar"] = None
    plot_data["filters"] = filters
    if name is not None:
        plot_data["prefix"] = name + "_" + plot_data["prefix"]
    post_plot(plot_data)


def create_line_figure(data_path, x_axis, y_axis, subsection=None, colorbar=None, filters=None,
                       size_factor: Tuple[float, float] = 2, ci=1, name=None, min_vmin=None, max_vmax=None):
    plot_data = get_plot_data(data_path, x_axis, y_axis, subsection=subsection, colorbar=colorbar, filters=filters)

    fig = pre_plot(size_factor)

    color_map = main_color_palette(n_colors=10)
    if colorbar is not None:
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("hsl", color_map)
        if max_vmax is None:
            max_vmax = max(plot_data["hue"])
        if min_vmin is None:
            min_vmin = min(plot_data["hue"])
        norm = matplotlib.colors.Normalize(vmin=min_vmin, vmax=max_vmax)
        scalar_map = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        # cbar = plt.colorbar(scalar_map)
        # cbar.ax.set_ylabel(to_title_case(colorbar))

    g = seaborn.lineplot(**plot_data, palette=color_map, linewidth=1, errorbar=('ci', ci))

    plot_data["data_path"] = data_path
    plot_data["prefix"] = "l"
    plot_data["fig"] = fig
    plot_data["g"] = g
    plot_data["x_axis"] = x_axis
    plot_data["y_axis"] = y_axis
    plot_data["subsection"] = subsection
    plot_data["colorbar"] = colorbar
    plot_data["filters"] = filters
    plot_data["y_prefix"] = "Average "
    if name is not None:
        plot_data["prefix"] = name + "_" + plot_data["prefix"]
    post_plot(plot_data)


def create_line_figure_max(data_path, x_axis, y_axis, subsection=None, colorbar=None, filters=None,
                           size_factor: Tuple[float, float] = 2.0, x_lim=None, name=None, min_vmin=None, max_vmax=None):
    plot_data = get_plot_data(data_path, x_axis, y_axis, subsection=subsection, colorbar=colorbar, filters=filters)
    fig = pre_plot(size_factor)

    max_keys = ["x"]

    if subsection is not None:
        max_keys.append("hue")
    if colorbar is not None and subsection is not None:
        max_keys.append("style")
    if colorbar is not None and subsection is None:
        max_keys.append("hue")

    plot_data = pick_max_from_plot_data(plot_data, max_keys, "y")

    if colorbar is not None:
        color_map = main_color_palette(n_colors=256)
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("hsl", color_map)
        if max_vmax is None:
            max_vmax = max(plot_data["hue"])
        if min_vmin is None:
            min_vmin = min(plot_data["hue"])
        norm = matplotlib.colors.Normalize(vmin=min_vmin, vmax=max_vmax)
        scalar_map = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        g = seaborn.lineplot(**plot_data, palette=color_map, linewidth=1, errorbar=('ci', 1), hue_norm=norm)
        cbar = plt.colorbar(scalar_map)
        cbar.ax.set_ylabel(to_title_case(colorbar))
    else:
        color_map = main_color_palette(n_colors=len(plot_data["hue_order"]))
        g = seaborn.lineplot(**plot_data, palette=color_map, linewidth=1, errorbar=('ci', 1))

    if x_lim is not None:
        g.set_xlim(*x_lim)
    # g.set(yscale="log")

    plot_data["data_path"] = data_path
    plot_data["prefix"] = "lm"
    plot_data["fig"] = fig
    plot_data["g"] = g
    plot_data["x_axis"] = x_axis
    plot_data["y_axis"] = y_axis
    plot_data["subsection"] = subsection
    plot_data["colorbar"] = colorbar
    plot_data["filters"] = filters
    plot_data["y_prefix"] = "Maximum "
    if name is not None:
        plot_data["prefix"] = name + "_" + plot_data["prefix"]
    post_plot(plot_data)


def create_heatmaps_figure_max(data_path, x_axis, y_axis, subsection=None, filters=None,
                               size_factor: Tuple[float, float] = 2, name=None, min_vmin=None, max_vmax=None):
    plot_data = get_plot_data(data_path, x_axis, y_axis, subsection=subsection, colorbar=None, filters=filters)

    fig = pre_plot(size_factor)

    new_x_axis = []
    new_y_axis = []
    for i in plot_data["x"]:
        if i not in new_x_axis:
            new_x_axis.append(i)
    for i in plot_data["hue"]:
        if i not in new_y_axis:
            new_y_axis.append(i)

    if set(new_x_axis) == set(new_y_axis):
        new_y_axis = new_x_axis
    if set(new_x_axis) == set(new_y_axis) == {"None", "Clamp", "L1Norm", "L1NormM", "L1NormW", "L1NormWM", "L2Norm",
                                              "L2NormM", "L2NormW", "L2NormWM"}:
        new_y_axis = new_x_axis = ["None", "Clamp", "L1Norm", "L1NormM", "L1NormW", "L1NormWM", "L2Norm", "L2NormM",
                                   "L2NormW", "L2NormWM"]

    data = np.zeros((len(new_y_axis), len(new_x_axis)))
    for i, y in enumerate(plot_data['y']):
        index = (new_y_axis.index(plot_data['hue'][i]), new_x_axis.index(plot_data['x'][i]))
        data[index] = max(y, data[index])

    color_map = main_color_palette(n_colors=10)[:4]
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list("hsl", color_map)

    g = seaborn.heatmap(
        data=data,
        xticklabels=new_x_axis,
        yticklabels=new_y_axis,
        cmap=color_map,
        vmin=min_vmin,
        vmax=max_vmax,
        cbar_kws={"label": "Maximum " + to_title_case(y_axis)},
        annot=True,
        fmt=".0f",
    )

    plot_data["data_path"] = data_path
    plot_data["prefix"] = "hm"
    plot_data["fig"] = fig
    plot_data["g"] = g
    plot_data["x_axis"] = x_axis
    plot_data["y_axis"] = subsection
    plot_data["subsection"] = None
    plot_data["colorbar"] = None
    plot_data["filters"] = filters
    if name is not None:
        plot_data["prefix"] = name + "_" + plot_data["prefix"]
    post_plot(plot_data, y_lim=None)


def calculate_max_accuracy(data_path, test_in):
    data_path = Path(data_path)
    plot_data = get_plot_data(data_path, test_in, "loss_accuracy.test_accuracy")
    max_accuracies = {}
    for i in set(plot_data["x"]):
        max_accuracies[i] = 0.0

    for index, value in enumerate(plot_data["y"]):
        value = max(value)
        if max_accuracies[plot_data["x"][index]] < value:
            max_accuracies[plot_data["x"][index]] = value
            max_accuracies[plot_data["x"][index] + "_index"] = index

    print(max_accuracies)


def create_convergence_figure(data_path, size_factor):
    data_path = Path(data_path)
    test_accuracies = {}
    norm_classes = {}
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, (run_name, run_data) in enumerate(data.items()):
        loss_accuracy = run_data["loss_accuracy"]
        test_accuracy = loss_accuracy["test_accuracy"]
        test_accuracies[run_name] = test_accuracy

        if run_data["parameter_log"]["norm_class_w"] != run_data["parameter_log"]["norm_class_y"]:
            continue

        if run_data["parameter_log"]["norm_class_w"] not in norm_classes:
            norm_classes[run_data["parameter_log"]["norm_class_w"]] = []

        norm_classes[run_data["parameter_log"]["norm_class_w"]].append(run_name)

        # if i > 10:
        #     break

    for norm_class, run_names in norm_classes.items():
        fig = pre_plot(size_factor=size_factor)

        count = 0
        avg_test_accuracy = [0] * 200
        max_test_accuracy_run = None
        min_test_accuracy_run = None

        for run_name in run_names:
            if len(test_accuracies[run_name]) < 200:
                continue

            if max_test_accuracy_run is None:
                max_test_accuracy_run = test_accuracies[run_name][-1], run_name
                min_test_accuracy_run = test_accuracies[run_name][-1], run_name

            plt.plot(
                test_accuracies[run_name],
                color="C0",
                alpha=0.1,
            )

            for i, x in enumerate(test_accuracies[run_name]):
                avg_test_accuracy[i] += x

            if np.average(test_accuracies[run_name]) > max_test_accuracy_run[0]:
                max_test_accuracy_run = np.average(test_accuracies[run_name]), run_name

            if np.average(test_accuracies[run_name]) < min_test_accuracy_run[0]:
                min_test_accuracy_run = np.average(test_accuracies[run_name]), run_name

            count += 1

        if count == 0:
            continue

        print(norm_class)

        avg_test_accuracy = [x / count for x in avg_test_accuracy]
        plt.plot(
            avg_test_accuracy,
            color="C1",
            label="Average Performance",
        )
        plt.plot(
            test_accuracies[max_test_accuracy_run[1]],
            color="C2",
            label="Maximum Performance",
        )
        plt.plot(
            test_accuracies[min_test_accuracy_run[1]],
            color="C3",
            label="Minimum Performance",
        )
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy")
        plt.title(f"Convergence of '{norm_class}' models")
        fig.tight_layout()
        fig.savefig(data_path.parent / f"{data_path.stem}_convergence_{norm_class}.png", dpi=600)
        plt.close(fig)


if __name__ == '__main__':
    location = r"C:\X"
    prefix = "gelu"
    # compile_data(f"{location}/{prefix}_ns_results")
    # compile_data(f"{location}/{prefix}_ni_results")
    # compile_data(f"{location}/{prefix}_ps_results")
    # compile_data(f"{location}/{prefix}_pi_results")
    # compile_data(f"{location}/{prefix}_pls_results")
    # compile_data(f"{location}/{prefix}_pli_results")
    # compile_data(f"{location}/{prefix}_ci_results")
    # compile_data(f"{location}/{prefix}_cli_results")
    # compile_data(f"{location}/{prefix}_cli1_results")
    # compile_data(f"{location}/{prefix}_fi_results")
    # compile_data(f"{location}/{prefix}_fli_results")
    # compile_data(f"{location}/{prefix}_fli0_results")
    # compile_data(f"{location}/{prefix}_fli1_results")

    # filters = {
    #     "bit_precision_w": 5,
    # }
    # create_line_figure_max(
    #     f"{location}/{prefix}_pli_results.pt",
    #     "parameter_log.activation_i",
    #     "max_test_accuracy",
    #     colorbar="parameter_log.leakage_w",
    #     name="73",
    #     size_factor=(6.5 * 1 / 3, 1.61803398874),
    #     filters=filters,
    # )
    # create_line_figure_max(
    #     f"{location}/{prefix}_pls_results.pt",
    #     "parameter_log.activation_s",
    #     "max_test_accuracy",
    #     colorbar="parameter_log.leakage_w",
    #     name="73",
    #     size_factor=(6.5 * 1 / 3, 1.61803398874),
    #     filters=filters,
    # )
    # create_line_figure_max(
    #     f"{location}/{prefix}_pli_results.pt",
    #     "parameter_log.activation_i",
    #     "max_test_accuracy",
    #     colorbar="std_w",
    #     name="73",
    #     max_vmax=0.1,
    #     size_factor=(6.5 * 1 / 3, 1.61803398874),
    #     filters=filters,
    # )
    # create_line_figure_max(
    #     f"{location}/{prefix}_pls_results.pt",
    #     "parameter_log.activation_s",
    #     "max_test_accuracy",
    #     colorbar="std_w",
    #     name="73",
    #     size_factor=(6.5 * 1 / 3, 1.61803398874),
    #     max_vmax=0.1,
    #     filters=filters,
    # )
    # create_line_figure_max(
    #     f"{location}/{prefix}_cli1_results.pt",
    #     "parameter_log.activation_i",
    #     "max_test_accuracy",
    #     colorbar="parameter_log.num_conv_layer",
    #     name="73",
    #     size_factor=(6.5 * 1 / 3, 1.61803398874),
    #     filters=filters,
    # )
    # create_line_figure_max(
    #     f"{location}/{prefix}_cli_results.pt",
    #     "parameter_log.activation_i",
    #     "max_test_accuracy",
    #     colorbar="parameter_log.num_conv_layer",
    #     name="73",
    #     size_factor=(6.5 * 1 / 3, 1.61803398874),
    #     filters=filters,
    # )
    # create_line_figure_max(
    #     f"{location}/{prefix}_ci_results.pt",
    #     "parameter_log.activation_i",
    #     "max_test_accuracy",
    #     colorbar="parameter_log.num_conv_layer",
    #     name="73",
    #     size_factor=(6.5 * 1 / 3, 1.61803398874),
    #     filters=filters,
    # )
    # create_line_figure_max(
    #     f"{location}/{prefix}_fli0_results.pt",
    #     "parameter_log.activation_i",
    #     "max_test_accuracy",
    #     colorbar="parameter_log.num_linear_layer",
    #     name="73",
    #     size_factor=(6.5 * 1 / 3, 1.61803398874),
    #     filters=filters,
    # )
    # create_line_figure_max(
    #     f"{location}/{prefix}_fli_results.pt",
    #     "parameter_log.activation_i",
    #     "max_test_accuracy",
    #     colorbar="parameter_log.num_linear_layer",
    #     name="73",
    #     size_factor=(6.5 * 1 / 3, 1.61803398874),
    #     filters=filters,
    # )
    # create_line_figure_max(
    #     f"{location}/{prefix}_fi_results.pt",
    #     "parameter_log.activation_i",
    #     "max_test_accuracy",
    #     colorbar="parameter_log.num_linear_layer",
    #     name="73",
    #     size_factor=(6.5 * 1 / 3, 1.61803398874),
    #     filters=filters,
    # )

    precision = 2 ** 8
    clamp = Clamp()
    rp = ReducePrecision(precision=precision)
    noise = GaussianNoise(leakage=0.5, precision=precision)
    inputs = np.linspace(-0.25, 0.25, precision * 2 + 1)
    weights = np.linspace(-0.25, 0.25, precision * 2 + 1)
    x1 = clamp(torch.tensor(inputs, requires_grad=False))
    x1 = rp(x1)
    x1 = noise(x1)
    x2 = clamp(torch.tensor(weights, requires_grad=False))
    x2 = rp(x2)
    x2 = noise(x2)
    output = np.matmul(x1.reshape(-1, 1), x2.reshape(1, -1))
    output = noise(output)
    output = clamp(output)
    output = rp(output)
    output = np.clip(output, -1, 1)
    output = GeLU()(output)
    # diff = output - np.matmul(inputs.reshape(-1, 1), weights.reshape(1, -1))
    plt.contourf(inputs, weights, output, levels=100)
    plt.colorbar()
    plt.show()
