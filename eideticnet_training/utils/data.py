import copy
from pathlib import Path

from tqdm import tqdm

from datasets import load_dataset
import numpy as np
import torch
import torchvision as tvis
from torchvision import transforms
from torch.utils.data.dataset import ConcatDataset


class RGBTransform:

    def __init__(self):
        pass

    def __call__(self, sample):
        return sample.convert("RGB")


def torchvision_transforms(dataset):
    if dataset == "MNIST":
        train_transform = transforms.Compose(
            [
                RGBTransform(),
                transforms.RandomResizedCrop(32, scale=(0.4, 1)),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [RGBTransform(), transforms.Resize(32), transforms.ToTensor()]
        )
    elif dataset.lower().startswith("image"):
        # ImageNet, ImageNette, ImageWoof.
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop((224, 224)), transforms.ToTensor()]
        )
        test_transform = transforms.Compose(
            [transforms.CenterCrop((224, 224)), transforms.ToTensor()]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.ToTensor()
    return train_transform, test_transform


@torch.no_grad
def create_sequential_classification(dataset, n_tasks=2):
    tasks = []
    labels = []
    if isinstance(dataset, CustomDataset):
        labels = np.array(dataset.hf_dataset["label"])
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=False, num_workers=10
        )
        for _, y in tqdm(
            loader,
            desc="getting labels for task splitting",
            total=len(loader),
        ):
            labels.append(y.numpy())
        labels = np.concatenate(labels)
    n_classes = labels.max() + 1
    step = n_classes // n_tasks
    for i in range(n_tasks):
        subset = np.isin(labels, np.arange(i * step, i * step + step))
        subset = np.flatnonzero(subset)
        tasks.append(torch.utils.data.Subset(dataset, subset))
        print(f"Task {i} length: {len(tasks[-1])}")
    return tasks


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, i):
        x = self.hf_dataset[i.item()]["image"]
        y = self.hf_dataset[i.item()]["label"]
        return self.transform(x.convert("RGB")), y


def torchvision_dataset(dataset, path="~/datasets"):
    train_transform, test_transform = torchvision_transforms(dataset)
    path = Path(path).expanduser()
    if dataset == "MNIST":
        train_dataset = tvis.datasets.MNIST(
            path / "MNIST",
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = tvis.datasets.MNIST(
            path / "MNIST", train=False, transform=test_transform
        )
    elif dataset == "FashionMNIST":
        train_dataset = tvis.datasets.FashionMNIST(
            path / "FashionMNIST",
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = tvis.datasets.FashionMNIST(
            path / "FashionMNIST", train=False, transform=test_transform
        )
    elif dataset == "SVHN":
        train_dataset = tvis.datasets.SVHN(
            path / "SVHN",
            split="train",
            download=True,
            transform=train_transform,
        )
        test_dataset = tvis.datasets.SVHN(
            path / "SVHN",
            split="test",
            transform=test_transform,
            download=True,
        )
    elif dataset == "CIFAR10":
        train_dataset = tvis.datasets.CIFAR10(
            path / "CIFAR10",
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = tvis.datasets.CIFAR10(
            path / "CIFAR10", train=False, transform=test_transform
        )
    elif dataset == "CIFAR100":
        train_dataset = tvis.datasets.CIFAR100(
            path / "CIFAR100",
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = tvis.datasets.CIFAR100(
            "~/datasets/CIFAR100", train=False, transform=test_transform
        )
    elif dataset == "IMAGENETTE":
        ds = load_dataset("frgfm/imagenette", "160px")
        train_dataset = CustomDataset(ds["train"], train_transform)
        test_dataset = CustomDataset(ds["validation"], test_transform)
    elif dataset == "IMAGEWOOF":
        ds = load_dataset("frgfm/imagewoof", "160px")
        train_dataset = CustomDataset(ds["train"], train_transform)
        test_dataset = CustomDataset(ds["validation"], test_transform)
    elif dataset == "imagenet":
        ds = load_dataset("ILSVRC/imagenet-1k")
        train_dataset = ds["train"]
        test_dataset = ds["validation"]
    return train_dataset, test_dataset


def permute_image(image, permutation):
    if permutation is None:
        return image
    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    return image.view(c, h, w)


def get_pmnist(args, path="~/datasets"):
    np.random.seed(args.random_seed)
    path = Path(path).expanduser()
    permutations = [np.random.permutation(28**2) for i in range(args.num_tasks)]

    def get_mnist(name, train=True, permutation=None):
        dataset_transform = tvis.transforms.Compose(
            [
                tvis.transforms.ToTensor(),
                tvis.transforms.Lambda(
                    lambda image: permute_image(image, permutation)
                ),
            ]
        )
        return tvis.datasets.MNIST(
            path / "PMNIST",
            train=train,
            download=True,
            transform=dataset_transform,
        )

    train_datasets = [get_mnist("mnist", permutation=p) for p in permutations]
    test_datasets = [
        get_mnist("mnist", train=False, permutation=p) for p in permutations
    ]

    return train_datasets, test_datasets


def make_task_classifier_dataset(datasets):
    """
    Make a dataset comprising all task datasets and replace class labels with
    task labels. This allows training a final classifier head that predicts the
    task to which task a given input belongs.
    """

    def update_labels_or_targets(d, task_id, indices=None):
        if hasattr(d, "labels"):
            if indices is not None:
                d.labels[indices] = task_id
            else:
                d.labels[:] = task_id
        elif hasattr(d, "targets"):
            if indices is not None:
                if isinstance(d.targets, list):
                    for i in indices:
                        d.targets[i] = task_id
                else:
                    d.targets[indices] = task_id
            else:
                d.targets[:] = task_id
        elif hasattr(d, "hf_dataset"):
            for i in indices:
                d.hf_dataset[i.item()]["label"] = task_id
        else:
            raise ValueError(
                "Dataset has neither 'labels' nor 'targets' nor 'hf_dataset'."
            )
        return d

    # Make defensive copy.
    datasets = copy.deepcopy(datasets)
    for task_id, dataset in enumerate(datasets):
        if hasattr(dataset, "dataset"):
            update_labels_or_targets(dataset.dataset, task_id, dataset.indices)
        else:
            update_labels_or_targets(dataset, task_id)
    task_classifier_dataset = ConcatDataset(datasets)
    return task_classifier_dataset


"""
def load_mnist(train_kwargs, test_kwargs):
    transform = tvis.transforms.Compose(
        [
            tvis.transforms.ToTensor(),
            tvis.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = tvis.datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_dataset = tvis.datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    return train_loader, test_loader


def make_train_test_split(data, target, test_frac: float = 0.25):
    train_mask = torch.rand(len(data)) > test_frac
    test_mask = ~train_mask
    return (
        data[train_mask],
        target[train_mask],
        data[test_mask],
        target[test_mask],
    )


def make_loader(data, target, batch_size, shuffle=False):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
        if set(target.flatten()) == {0.0, 1.0}:
            # Assume binary cross entropy.
            target = torch.from_numpy(target).float()
        else:
            # Assume cross entropy.
            target = torch.from_numpy(target).long()

    dataset = torch.utils.data.TensorDataset(data, target)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return loader


def make_adversarial_loader(loader, adv_data):
    adv_loader = copy.deepcopy(loader)
    adv_data = torch.from_numpy(adv_data).float()
    if hasattr(adv_loader.dataset, "tensors"):
        tmp = list(adv_loader.dataset.tensors)
        tmp[0] = adv_data
        adv_loader.dataset.tensors = tuple(tmp)
    elif hasattr(adv_loader.dataset, "data"):
        if adv_data.ndim == 4:
            # e.g. for MNIST
            # (10000, 1, 28, 28) -> (10000, 28, 28)
            assert adv_data.shape[1] == 1
            adv_data = adv_data.squeeze(1)
        adv_loader.dataset.data = adv_data
    else:
        raise Exception("Not sure how to handle this loader")
    return adv_loader
"""
