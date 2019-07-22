from typing import Tuple

import numpy as np
from torch import nn, Tensor
from torchvision import models


class Classifier(nn.Module):
    n_levels: int
    _splitted_heads: bool
    _model: nn.Module
    _boundaries: Tuple[int, ...]

    def __init__(self, level_sizes: Tuple[int, ...], splitted_heads: bool):
        super().__init__()

        self._splitted_heads = splitted_heads
        self.n_levels = len(level_sizes)

        self._model = models.resnet18(pretrained=True)
        self._fix_arch(level_sizes)

        bounds = np.cumsum(level_sizes).tolist()
        bounds.insert(0, 0)
        self._boundaries = tuple(bounds)

    def _fix_arch(self, level_sizes: Tuple[int, ...]):
        self._model.avgpool = nn.AdaptiveAvgPool2d(1)

        in_dim = self._model.fc.in_features
        out_dim = self._model.fc.out_features
        if self._splitted_heads:
            self._heads = nn.ModuleList([nn.Sequential(nn.Linear(in_features=out_dim, out_features=sz),
                                                       nn.Dropout(p=0.5),
                                                       nn.ReLU()
                                                       )
                                         for sz in level_sizes])

        else:
            self._model.fc = nn.Linear(in_features=in_dim, out_features=sum(level_sizes))

    def forward(self, inp: Tensor) -> Tuple[Tensor, ...]:
        if self._splitted_heads:
            return self._forward_many_head(inp)

        else:
            return self._forward_one_head(inp)

    def _forward_many_head(self, inp: Tensor) -> Tuple[Tensor, ...]:
        logits_flat = self._model(inp)
        logits_list = [head(logits_flat) for head in self._heads]
        logits = tuple(logits_list)
        return logits

    def _forward_one_head(self, inp: Tensor) -> Tuple[Tensor, ...]:
        logits_flat = self._model(inp)

        logits_list = []
        for i_level in range(self.n_levels):
            lb = self._boundaries[i_level]
            rb = self._boundaries[i_level + 1]
            logits_list.append(logits_flat[:, lb: rb])

        logits = tuple(logits_list)
        return logits

    def set_train_mode(self) -> None:
        self.train()
        for param in self.parameters():
            param.requires_grad = True

    def set_test_mode(self) -> None:
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
