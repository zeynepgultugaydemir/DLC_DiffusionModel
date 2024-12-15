import os
from collections import namedtuple
from typing import Optional, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import default_collate, DataLoader, Dataset, random_split
from torchvision.datasets import CelebA, FashionMNIST

DataInfo = namedtuple('DataInfo', 'image_channels image_size num_classes sigma_data')
DataLoaders = namedtuple('DataLoaders', 'train valid')


def load_dataset_and_make_dataloaders(
        dataset_name: str,
        root_dir: str,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last=True,
        target_class=None
) -> Tuple[DataLoaders, DataInfo]:
    train_dataset, valid_dataset, data_info = load_dataset(dataset_name, root_dir, target_class)
    dl = make_dataloaders(train_dataset, valid_dataset, data_info.num_classes, batch_size, num_workers, pin_memory,
                          drop_last)
    return dl, data_info


def load_dataset(dataset_name='FashionMNIST', root_dir='data', target_class=None) -> Tuple[Dataset, Dataset, DataInfo]:
    if dataset_name == 'FashionMNIST':
        t = T.Compose([T.ToTensor(), T.Pad(2), T.Normalize(mean=(0.5,), std=(0.5,))])
        train_dataset = FashionMNIST(root_dir, download=True, transform=t)
        train_dataset, valid_dataset = random_split(train_dataset, [50000, 10000])
        num_classes = 10

    elif dataset_name == 'CelebA':
        t = T.Compose(
            [T.ToTensor(), T.CenterCrop(178), T.Resize(128, antialias=True), T.Normalize(mean=(0.5,), std=(0.5,))])
        train_dataset = CelebA(root_dir, download=True, transform=t)
        train_dataset, valid_dataset = random_split(train_dataset, [150000, 12770])
        num_classes = None

    else:
        raise RuntimeError('Unknown dataset: ' + dataset_name)

    # Apply filtering if target_class is provided
    if target_class is not None:
        train_indices = [i for i, (_, label) in enumerate(train_dataset) if label == target_class]
        valid_indices = [i for i, (_, label) in enumerate(valid_dataset) if label == target_class]

        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

    x, _ = next(iter(DataLoader(train_dataset, batch_size=10000, shuffle=True)))
    _, c, h, w = x.size()
    assert h == w
    sigma_data = x.std()

    return train_dataset, valid_dataset, DataInfo(c, h, num_classes, sigma_data)


def make_dataloaders(
        train_dataset: Dataset,
        valid_dataset: Dataset,
        num_classes: Optional[int],
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last=True

) -> DataLoaders:
    collate_fn = default_collate if num_classes is not None else lambda batch: (default_collate(batch)[0], None)
    kwargs = {'collate_fn': collate_fn, 'num_workers': num_workers, 'persistent_workers': (num_workers > 0),
              'pin_memory': pin_memory, 'drop_last': drop_last}

    return DataLoaders(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs),
        valid=DataLoader(valid_dataset, batch_size=2 * batch_size, **kwargs)
    )


def get_data_folder_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
