import os

from torchvision.transforms import Compose
from torchvision import transforms
from torch.utils.data import DataLoader

from .utils import get_dataloader_base, UntargetPerturbedDataset
from src.config import settings
from src.attack import LinfPGDAttack

MNIST_DATA_DIR = os.path.join(settings.dataset_dir, "MNIST")
MNIST_TRAIN_STD_3D = (0.3081, 0.3081, 0.3081)
MNIST_TRAIN_MEAN_3D = (0.1307, 0.1307, 0.1307)
MNIST_TRAIN_STD_1D = 0.3081
MNIST_TRAIN_MEAN_1D = 0.1307


def get_mnist_3d_train_dataloader(*,
                                  batch_size: int = settings.batch_size,
                                  num_workers: int = settings.num_worker,
                                  shuffle: bool = True,
                                  normalize: bool = True) -> DataLoader:
    """resize mnist from 1 * 28 * 28 to 3 * 28 * 28"""
    compose_list = [
        # resize original mnist size(28 * 28) to 32 * 32
        transforms.Resize(32),
        # 1 * 32 * 32 to 3 * 32 * 32
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    if normalize:
        compose_list.append(transforms.Normalize(MNIST_TRAIN_MEAN_3D, MNIST_TRAIN_STD_3D))

    return get_dataloader_base(
        dataset="MNIST",
        data_dir=MNIST_DATA_DIR,
        batch_size=batch_size,
        num_workers=num_workers,
        train=True,
        shuffle=shuffle,
        transform=Compose(compose_list)
    )


def get_perturbed_mnist_3d_train_dataloader(*,
                                            batch_size: int = settings.batch_size,
                                            num_workers: int = settings.num_worker,
                                            shuffle: bool = True,
                                            normalize: bool = True,
                                            attacker: LinfPGDAttack) -> DataLoader:
    dataloader = get_mnist_3d_train_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        normalize=normalize
    )
    perturbed_dataset = UntargetPerturbedDataset(
        dataloader=dataloader,
        attacker=attacker
    )

    return DataLoader(perturbed_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


def get_mnist_3d_test_dataloader(*,
                                 batch_size: int = settings.batch_size,
                                 num_workers: int = settings.num_worker,
                                 shuffle: bool = False,
                                 normalize: bool = True) -> DataLoader:
    """resize mnist from 1 * 28 * 28 to 3 * 28 * 28"""
    compose_list = [
        # resize original mnist size(28 * 28) to 32 * 32
        transforms.Resize(32),
        # 1 * 32 * 32 to 3 * 32 * 32
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    if normalize:
        compose_list.append(transforms.Normalize(MNIST_TRAIN_MEAN_3D, MNIST_TRAIN_STD_3D))

    return get_dataloader_base(
        dataset="MNIST",
        data_dir=MNIST_DATA_DIR,
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
        shuffle=shuffle,
        transform=Compose(compose_list)
    )


def get_mnist_1d_train_dataloader(*,
                                  batch_size: int = settings.batch_size,
                                  num_workers: int = settings.num_worker,
                                  shuffle: bool = True,
                                  normalize: bool = True) -> DataLoader:
    """resize mnist from 1 * 28 * 28 to 3 * 28 * 28"""
    compose_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        compose_list.append(transforms.Normalize(MNIST_TRAIN_MEAN_1D, MNIST_TRAIN_STD_1D))

    return get_dataloader_base(
        dataset="MNIST",
        data_dir=MNIST_DATA_DIR,
        batch_size=batch_size,
        num_workers=num_workers,
        train=True,
        shuffle=shuffle,
        transform=Compose(compose_list)
    )


def get_mnist_1d_test_dataloader(*,
                                 batch_size: int = settings.batch_size,
                                 num_workers: int = settings.batch_size,
                                 shuffle: bool = False,
                                 normalize: bool = True) -> DataLoader:
    """resize mnist from 1 * 28 * 28 to 3 * 28 * 28"""
    compose_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        compose_list.append(transforms.Normalize(MNIST_TRAIN_MEAN_1D, MNIST_TRAIN_STD_1D))

    return get_dataloader_base(
        dataset="MNIST",
        data_dir=MNIST_DATA_DIR,
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
        shuffle=shuffle,
        transform=Compose(compose_list)
    )
