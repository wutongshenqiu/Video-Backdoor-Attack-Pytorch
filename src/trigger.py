from typing import Union, Tuple, Optional, Dict, Callable
import functools

import torch
from torch.nn import Module
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

from src.utils import logger, inject_trigger
from .config import settings


class Trigger:

    class _Decorators:
        @staticmethod
        def check_dataloader_initialized(func: Callable):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                if not self._dataloader:
                    raise AttributeError(f"Before using method `{func.__name__}`, "
                                         f"initialize dataloader with `load_dataset`!")
                return func(self, *args, **kwargs)

            return wrapper

    def __init__(self, *,
                 model: Module,
                 lr: float = 0.01,
                 epoch: int = 100,
                 trigger_size: Union[int, Tuple[int, int]] = 8,
                 device: Union[torch.device, str] = "cuda",
                 loss_function: _Loss = nn.CrossEntropyLoss()) -> None:
        self._model = model
        self._model.to(device)

        self._lr = lr
        self._epoch = epoch
        if isinstance(trigger_size, tuple):
            self._trigger_size = trigger_size
        else:
            self._trigger_size = trigger_size, trigger_size
        self._device = device
        self._loss_function = loss_function

        # below attribute must be initialized via method `load_dataset`
        self._dataloader: Optional[DataLoader] = None
        self._mask: Optional[Tensor] = None

    @_Decorators.check_dataloader_initialized
    def generate(self, target_label: int):
        logger.info(self.gather_parameters())

        labels: Tensor = next(iter(self._dataloader))[1]
        target_label_vector = torch.tensor(
            [target_label] * labels.shape[0],
            dtype=labels.dtype,
            device=self._device
        )

        # store former classified label
        former_classified_label_vector = torch.tensor(
            [0] * labels.shape[0],
            dtype=labels.dtype,
            device=self._device
        )

        self._model.eval()
        for ep in range(self._epoch):
            # number of images with triggers that inferred as targeted label by models
            attack_success_number = 0
            for inputs, _ in self._dataloader:
                # skip the last batch, owing to inconsistent shape
                if inputs.shape != self._mask.shape:
                    logger.debug("skip last batch")
                    continue

                inputs = inputs.to(self._device)
                self._add_trigger(inputs)
                inputs.requires_grad = True

                self._model.zero_grad()

                outputs = self._model(inputs)
                loss = self._loss_function(outputs, target_label_vector) - \
                    self._loss_function(outputs, former_classified_label_vector)
                loss.backward()

                self._mask = self._mask - self._lr * inputs.grad.sign()
                self._mask = torch.clamp(self._mask, 0, 1)

                _, former_classified_label_vector = torch.max(outputs, 1)

                attack_success_number += (former_classified_label_vector == target_label_vector).sum().item()

            logger.info(f"epoch: {ep}, attack success number: {attack_success_number}, "
                        f"attack success rate: {attack_success_number / len(self._dataloader.dataset)}")

            if ep % 9 == 0:
                torch.save(self._mask.data, settings.trigger_dir / f"triggers-epoch{ep}")

    def _add_trigger(self, inputs: Tensor) -> None:
        inject_trigger(
            inputs=inputs,
            trigger=self._mask,
            trigger_size=self._trigger_size,
            layer=1
        )

    def load_dataset(self, dataloader: DataLoader) -> None:
        logger.info("loading dataset")
        # get shape of one batch data
        batch_shape = next(iter(dataloader))[0].shape
        self._init_mask(batch_shape)

        self._dataloader = dataloader

    def _init_mask(self, mask_shape: Tuple[int, int, int, int]) -> None:
        # 255 / 2 should be equal to 0.5
        self._mask = torch.zeros(mask_shape, device=self._device) + 0.5
        logger.info("initialize mask with random number")

    def gather_parameters(self) -> Dict:
        return {
            "learning rate: ": self._lr,
            "epoch: ": self._epoch,
            "triggers size: ": self._trigger_size,
            "device: ": self._device,
            "loss function": type(self._loss_function).__name__,
        }


if __name__ == '__main__':
    from src.networks import resnet18
    from src.utils import get_mnist_3d_train_dataloader
    from .config import settings

    logger.change_log_file(settings.log_dir / "generate_trigger.log")
    device = "cuda:0"

    model = resnet18(num_classes=10)
    model.to(device)
    model.load_state_dict(torch.load(settings.model_dir / "resnet18-mnist3d-best", map_location=device))

    trigger = Trigger(model=model, device=device)
    trigger.load_dataset(get_mnist_3d_train_dataloader())

    trigger.generate(1)
