import os

from torch.utils.data import Dataset

from torchvision import datasets
from torchvision.transforms import ToTensor

def info(dataset_name: str):
    info_per_dataset = {
        "MNIST": dict(C=1, H=28, W=28, L=10),
        "FashionMNIST": dict(C=1, H=28, W=28, L=10),
        "CIFAR10": dict(C=3, H=32, W=32, L=10),
        "CIFAR100": dict(C=3, H=32, W=32, L=100),
        "SVHN": dict(C=3, H=32, W=32, L=10)
    }

    if dataset_name not in info_per_dataset:
        raise Exception('Unknown dataset: {dataset_name}.')

    return info_per_dataset[dataset_name]

def load(lab_config: dict, dataset_name: str, train: bool = False) -> Dataset:
    root = os.path.join(lab_config["Directory"]["Datasets"], dataset_name)

    if not os.path.exists(root):
        raise OSError(f"Path given as root: {root} does no exists.")

    match dataset_name:
        case "MNIST":
            return datasets.MNIST(root=root, download=True, transform=ToTensor(), train=train)

        case "FashionMNIST":
            return datasets.FashionMNIST(root=root, download=True, transform=ToTensor(), train=train)

        case "CIFAR10":
            return datasets.CIFAR10(root=root, download=True, transform=ToTensor(), train=train)

        case "CIFAR100":
            return datasets.CIFAR100(root=root, download=True, transform=ToTensor(), train=train)

        case "SVHN":
            return datasets.SVHN(root=root, download=True, transform=ToTensor(), split="train" if train else "test")

        case _:
            raise NotImplemented(f"Dataset loading method was not implemented yet: {dataset_name}")
