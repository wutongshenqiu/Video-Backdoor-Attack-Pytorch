import torch
from torch import Tensor
from torch.nn import Module

from typing import Tuple

from src.utils import replicate_inputs


class LabelMixin:
    _model: Module
    _targeted: bool

    def _get_predicted_label(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            outputs = self._model(x)
        _, y = torch.max(outputs, dim=1)

        return y

    def _verify_and_process_inputs(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if self._targeted:
            assert y is not None
        else:
            if y is None:
                y = self._get_predicted_label(x)

        x = replicate_inputs(x)
        y = replicate_inputs(y)

        return x, y
