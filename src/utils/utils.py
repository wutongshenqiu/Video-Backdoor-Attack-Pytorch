import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

import numpy as np
from numpy import ndarray

from typing import Union, Tuple


def evaluate_top1_accuracy(*,
                           model: Module,
                           dataloader: DataLoader,
                           device: Union[torch.device, str]) -> float:
    model.eval()

    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    return correct / len(dataloader.dataset)


def get_raw_image(*,
                  img: Tensor,
                  mean: Union[float, Tensor] = 0.,
                  std: Union[float, Tensor] = 1.) -> ndarray:
    """get raw image in Dataloader, for display with pyplot"""
    # we should unnormalize first
    img = img * std + mean
    np_img = img.numpy()

    return np.transpose(np_img, (1, 2, 0))


def replicate_inputs(x: Tensor) -> Tensor:
    return x.detach().clone()


def clamp(t: Tensor, lower_limit, upper_limit):
    return torch.max(torch.min(t, upper_limit), lower_limit)


def inject_trigger(*,
                   inputs: Tensor,
                   trigger: Tensor,
                   trigger_size: Tuple[int, int],
                   layer: int) -> None:
    """inject triggers in the bottom right corner"""
    for channel in range(inputs.shape[1]):
        for h in range(trigger_size[0]):
            for w in range(trigger_size[1]):
                inputs[:layer, channel, -(h + 1), -(w + 1)] = trigger[:layer, channel, -(h + 1), -(w + 1)]


if __name__ == '__main__':
    from .dataset import get_mnist_3d_test_dataloader
    import matplotlib.pyplot as plt
    from .logger import logger

    logger.change_log_file("tmp.log")

    dataloader = get_mnist_3d_test_dataloader(num_workers=0, batch_size=10)
    trigger = torch.load("./triggers/trigger.pth", map_location="cpu")

    mean = (0.1307, 0.1307, 0.1307)
    std = (0.3081, 0.3081, 0.3081)

    for inputs, labels in dataloader:
        for i in range(5):
            plt.subplot(3, 5, i+1)
            plt.title(labels[i].item())
            plt.axis("off")
            plt.imshow(get_raw_image(img=inputs[i].cpu(),
                                     mean=Tensor(mean).view(3, 1, 1),
                                     std=Tensor(std).view(3, 1, 1)))

        inject_trigger(
            inputs=inputs,
            trigger=trigger,
            trigger_size=(8, 8),
            layer=3,
        )
        for i in range(5):
            plt.subplot(3, 5, i+6)
            plt.title(labels[i].item())
            plt.axis("off")
            plt.imshow(get_raw_image(img=inputs[i].cpu(),
                                     mean=Tensor(mean).view(3, 1, 1),
                                     std=Tensor(std).view(3, 1, 1)))
        break

    plt.show()