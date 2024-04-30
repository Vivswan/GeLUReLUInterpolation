import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataPaths:
    name: str
    model_data: Path
    tensorboard: Path
    dataset: Path
    logs: Path
    timestamp: str


def data_dirs(path, name=None, timestamp=None) -> DataPaths:
    if timestamp is None:
        timestamp = str(int(time.time()))

    name = timestamp + ("" if name is None else ("_" + name))

    path = Path(path)
    dataset_path = path.joinpath("datasets")
    logs_path = path.joinpath("logs")
    models_path = path.joinpath(f"models/{name}")
    tensorboard_path = path.joinpath(f"tensorboard/{name}")

    for p in [path, dataset_path, logs_path, models_path, tensorboard_path]:
        p.mkdir(parents=True, exist_ok=True)

    return DataPaths(
        name=name,
        model_data=models_path,
        tensorboard=tensorboard_path,
        dataset=dataset_path,
        logs=logs_path,
        timestamp=timestamp,
    )
