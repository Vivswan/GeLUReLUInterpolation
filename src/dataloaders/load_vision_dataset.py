import inspect
from typing import Tuple, Type

from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


def load_vision_dataset(
        dataset: Type[VisionDataset],
        path,
        batch_size,
        is_cuda=False,
        train_transform=None,
        test_transform=None,
) -> Tuple[DataLoader, DataLoader, list[int], tuple[str]]:
    dataset_kwargs = {
        'batch_size': batch_size,
        'shuffle': True
    }

    if is_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
        }
        dataset_kwargs.update(cuda_kwargs)

    if "train" in inspect.getfullargspec(dataset.__init__).args:
        train_set = dataset(root=path, train=True, download=True, transform=train_transform)
        test_set = dataset(root=path, train=False, download=True, transform=test_transform)
    elif "split" in inspect.getfullargspec(dataset.__init__).args:
        train_set = dataset(root=path, split="train", download=True, transform=train_transform)
        test_set = dataset(root=path, split="test", download=True, transform=test_transform)
    else:
        raise Exception(f"{dataset} does have a pre split of training data.")

    train_loader = DataLoader(train_set, **dataset_kwargs)
    test_loader = DataLoader(test_set, **dataset_kwargs)

    zeroth_element = next(iter(test_loader))[0]
    input_shape = list(zeroth_element.shape)

    return train_loader, test_loader, input_shape, tuple(train_set.classes)
