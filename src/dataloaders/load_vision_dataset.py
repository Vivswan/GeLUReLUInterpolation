import inspect
from typing import Type

from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms


def get_vision_dataset_transformation(grayscale=True):
    transformation = [
        # transforms.RandomCrop(32, padding=4),
        # transforms.Resize(32),
        # transforms.RandomHorizontalFlip(),
    ]
    if grayscale:
        transformation.append(transforms.Grayscale())

    transformation.append(transforms.ToTensor())
    # transformation.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    return transforms.Compose(transformation)


def load_vision_dataset(dataset: Type[VisionDataset], path, batch_size, is_cuda=False, grayscale=True) -> (
        DataLoader, DataLoader, tuple):
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

    if "train" not in inspect.getfullargspec(dataset.__init__).args:
        raise Exception(f"{dataset} does have a pre split of training data.")

    train_set = dataset(root=path, train=True, download=True, transform=get_vision_dataset_transformation(grayscale))
    train_loader = DataLoader(train_set, **dataset_kwargs)

    test_set = dataset(root=path, train=False, download=True, transform=get_vision_dataset_transformation(grayscale))
    test_loader = DataLoader(test_set, **dataset_kwargs)

    zeroth_element = next(iter(test_loader))[0]
    input_shape = list(zeroth_element.shape)

    return train_loader, test_loader, input_shape, tuple(train_set.classes)
