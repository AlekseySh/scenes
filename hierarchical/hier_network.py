from typing import Tuple

import numpy as np
from torch import nn, Tensor
from torchvision import models


class Classifier(nn.Module):
    n_levels: int
    _model: nn.Module
    _boundaries: Tuple[int, ...]

    def __init__(self, level_sizes: Tuple[int, ...]):
        super().__init__()

        self._model = models.resnet18(pretrained=True)
        self._model.avgpool = nn.AdaptiveAvgPool2d(1)

        in_dim = self._model.fc.in_features
        self._model.fc = nn.Linear(in_features=in_dim, out_features=sum(level_sizes))

        self.n_levels = len(level_sizes)

        bounds = np.cumsum(level_sizes).tolist()
        bounds.insert(0, 0)
        self._boundaries = tuple(bounds)

    def forward(self, inp: Tensor) -> Tuple[Tensor, ...]:
        logits_flat = self._model(inp)

        logits_list = []
        for i_level in range(self.n_levels):
            lb = self._boundaries[i_level]
            rb = self._boundaries[i_level + 1]
            logits_list.append(logits_flat[:, lb: rb])

        logits = tuple(logits_list)
        return logits
