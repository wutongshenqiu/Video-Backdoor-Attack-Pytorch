from typing import Tuple

import torchvision
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset
import torch
from torch import Tensor

from ..import logger
from src.attack import LinfPGDAttack
from ..utils import inject_trigger


def get_dataloader_base(*,
                        dataset: str,
                        data_dir: str,
                        batch_size: int,
                        num_workers: int,
                        shuffle: bool,
                        train: bool,
                        transform: Compose) -> DataLoader:
    if not hasattr(torchvision.datasets, dataset):
        raise ValueError("dataset is not supported!")
    _dataset = getattr(torchvision.datasets, dataset)(
        root=data_dir, train=train, download=True, transform=transform
    )
    logger.info(f"dataset: {dataset}, batch size: {batch_size}, num workers: {num_workers}, "
                f"shuffle: {shuffle}, train: {train}")

    return DataLoader(_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


class UntargetPerturbedDataset(Dataset):
    """perturbed dataset generated by specified attack and original dataloader"""

    def __init__(self, *,
                 dataloader: DataLoader,
                 # todo
                 # should be baseclass for attack owing to type hint
                 attacker: LinfPGDAttack) -> None:
        assert not attacker._targeted

        logger.info("prepare perturbed dataset")
        attacker.display()

        self._dataset_len = len(dataloader.dataset)

        perturbed_inputs_tensor_list = []
        labels_tensor_list = []
        for inputs, labels in dataloader:
            perturbed_inputs_tensor = attacker.perturb(inputs, labels)
            perturbed_inputs_tensor_list.append(perturbed_inputs_tensor)
            labels_tensor_list.append(labels)

        self._inputs_list = torch.cat(perturbed_inputs_tensor_list, dim=0)
        self._labels_list = torch.cat(labels_tensor_list, dim=0)
        logger.debug(f"perturbed inputs list len: {self._inputs_list}")
        logger.debug(f"labels list len: {self._labels_list}")

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self._inputs_list[idx], self._labels_list[idx]

    def __len__(self) -> int:
        return self._dataset_len


class TriggerDataset(Dataset):
    """dataset with trigger injected in"""
    def __init__(self, *,
                 dataloader: DataLoader,
                 trigger: Tensor,
                 trigger_size: Tuple[int, int] = (8, 8),
                 layer: int = 1) -> None:
        logger.info("load dataset with trigger injected in")
        logger.info(f"trigger size: {trigger_size}\n"
                    f"layer: {layer}\n")

        self._dataset_len = len(dataloader.dataset)

        trigger_inputs_tensor_list = []
        labels_tensor_list = []
        for inputs, labels in dataloader:
            inject_trigger(
                inputs=inputs,
                trigger=trigger,
                trigger_size=trigger_size,
                layer=layer
            )
            trigger_inputs_tensor_list.append(inputs)
            labels_tensor_list.append(labels)

        self._inputs_list = torch.cat(trigger_inputs_tensor_list, dim=0)
        self._labels_list = torch.cat(labels_tensor_list, dim=0)
        logger.debug(f"trigger inputs list len: {self._inputs_list}")
        logger.debug(f"labels list len: {self._labels_list}")

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self._inputs_list[idx], self._labels_list[idx]

    def __len__(self) -> int:
        return self._dataset_len
