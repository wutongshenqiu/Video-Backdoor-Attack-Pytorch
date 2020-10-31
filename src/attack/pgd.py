from torch.nn import Module
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from typing import Union, Tuple, Optional

# from src.utils import logger
from .mixins import LabelMixin
from src.utils import clamp


class LinfPGDAttack(LabelMixin):
    """Projected Gradient Descent attack"""
    def __init__(self, *,
                 device: Union[str, torch.device],
                 targeted: bool = False,
                 model: Module,
                 epsilon: float,
                 step_size: float,
                 mean: Union[float, Tuple, Tensor],
                 std: Union[float, Tuple, Tensor],
                 clip_min: float = 0,
                 clip_max: float = 1,
                 random_start: bool = True,
                 iter_steps: int = 20,
                 loss_function: _Loss = nn.CrossEntropyLoss()) -> None:
        self._device = device
        self._targeted = targeted
        self._model = model
        self._model.to(device)

        self._prepare_mean_std(mean, std)

        self._epsilon = torch.true_divide(epsilon, self._std)
        self._step_size = torch.true_divide(step_size, self._std)
        self._clip_min = torch.true_divide((clip_min - self._mean), self._std)
        self._clip_max = torch.true_divide((clip_max - self._mean), self._std)

        self._random_start = random_start
        self._iter_steps = iter_steps
        self._loss_function = loss_function

    def _prepare_mean_std(self, mean, std):
        def to_tensor(t) -> Tensor:
            if isinstance(t, float):
                return Tensor(t)
            elif isinstance(t, Tuple):
                return Tensor(t).view(len(t), 1, 1)

        self._mean = to_tensor(mean).to(self._device)
        self._std = to_tensor(std).to(self._device)

    def perturb(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
            Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        Args:
            x: inputs tensor
            y: label tensor, if is targeted attack, y must be the targeted labels
        """
        x, y = x.to(self._device), y.to(self._device)
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x, device=self._device)
        if self._random_start:
            delta = self._random_delta(delta)

        # adversarial example
        xt: Tensor = x + delta
        xt.requires_grad = True

        self._model.eval()
        for it in range(self._iter_steps):
            outputs = self._model(xt)
            loss = self._loss_function(outputs, y)
            if self._targeted:
                loss = -loss
            loss.backward()

            grad_sign = xt.grad.detach().sign()
            xt.data = xt.detach() + self._step_size * grad_sign
            xt.data = clamp(xt - x, -self._epsilon, self._epsilon) + x
            xt.data = clamp(xt.detach(), self._clip_min, self._clip_max)
            xt.grad.data.zero_()

        return xt

    def _random_delta(self, delta: Tensor) -> Tensor:
        delta.data.uniform_(-1, 1)
        return delta * self._epsilon

    def display(self) -> None:
        logger.info(
            f"attack name: {type(self).__name__}\n"
            f"device: {self._device}\n"
            f"targeted: {self._targeted}\n"
            f"epsilon: {self._epsilon}\n"
            f"step size: {self._step_size}\n"
            f"mean: {self._mean}\n"
            f"std: {self._std}\n"
            f"clip min: {self._clip_min}\n"
            f"clip max: {self._clip_max}\n"
            f"random start: {self._random_start}\n"
            f"iter steps: {self._iter_steps}\n"
            f"loss function: {type(self._loss_function).__name__}\n"
        )


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from src.utils import get_mnist_3d_test_dataloader
    from src.networks import resnet18
    from src.utils import logger, get_raw_image
    from src.config import settings

    logger.change_log_file(settings.log_dir / "tmp.log")

    device = "cuda:0"
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load("./models/resnet18-mnist3d-best", map_location=device))
    model.to(device)
    model.eval()

    mean = (0.1307, 0.1307, 0.1307)
    std = (0.3081, 0.3081, 0.3081)
    attacker = LinfPGDAttack(
        device=device,
        model=model,
        epsilon=0.2,
        step_size=0.3/4,
        std=std,
        mean=mean,
        targeted=True
    )

    for inputs, labels in get_mnist_3d_test_dataloader(num_workers=0, batch_size=64):
        inputs = inputs.to(device)
        labels = torch.ones_like(labels)
        labels = labels.to(device)
        outputs = model(inputs)

        adv_inputs = attacker.perturb(inputs, labels)
        adv_outputs = model(adv_inputs)

        _, ori_predicted = torch.max(outputs, 1)
        ori_correct = (ori_predicted == labels).sum().item()
        print(f"original accuracy: {ori_correct / len(inputs)}")

        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_correct = (adv_predicted == labels).sum().item()

        print(f"original label: {labels}")
        print(f"normal predicted label: {ori_predicted}")
        print(f"adversarial pridicted label: {adv_predicted}")

        print(f"adversarial accuracy: {adv_correct / len(inputs)}")

        for i in range(5):
            plt.subplot(2, 5, i+1)
            plt.title(ori_predicted[i].item())
            plt.axis("off")
            plt.imshow(get_raw_image(img=inputs[i].cpu(),
                                     mean=Tensor(mean).view(3, 1, 1),
                                     std=Tensor(std).view(3, 1, 1)))

        for i in range(5):
            plt.subplot(2, 5, i+6)
            plt.title(adv_predicted[i].item())
            plt.axis("off")
            plt.imshow(get_raw_image(img=adv_inputs[i].detach().cpu(),
                                     mean=Tensor(mean).view(3, 1, 1),
                                     std=Tensor(std).view(3, 1, 1)))

        plt.show()
        break

