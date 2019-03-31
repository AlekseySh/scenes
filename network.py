import logging
from pathlib import Path
from typing import List, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torch.nn.functional import softmax

logger = logging.getLogger(__name__)


class Arch:
    RESNET18 = 'resnet18'
    RESNET34 = 'resnet34'
    RESNET50 = 'resnet50'
    VGG11 = 'vgg11'
    INCEPTION = 'inception'


class Classifier(nn.Module):
    arch: Arch

    _n_classes: int
    _model: nn.Module

    def __init__(self, arch: Arch, n_classes: int, pretrained: bool):
        super().__init__()

        logger.info(f'Creating classifier based on: {arch}')
        self._n_classes = n_classes

        if arch == Arch.RESNET18:
            self._model = models.resnet18(pretrained=pretrained)

        elif arch == Arch.RESNET34:
            self._model = models.resnet34(pretrained=pretrained)

        elif arch == Arch.RESNET50:
            self._model = models.resnet50(pretrained=pretrained)

        elif arch == Arch.VGG11:
            self._model = models.vgg11_bn(pretrained=pretrained)

        elif arch == Arch.INCEPTION:
            self._model = models.inception_v3(pretrained=pretrained)

        else:
            raise ValueError(f'Unexpected type {arch}')

        self.arch = arch
        self._adopt_arch()

    def forward(self, inp: Tensor) -> Tensor:
        return self._model(inp)

    def classify(self,
                 inp: Union[Tensor, List[Tensor]]
                 ) -> Tuple[Tensor, Tensor]:
        self._model.eval()
        if isinstance(inp, list):
            probs = self._classify_tta(inputs=inp)
        else:
            probs = self._classify_simple(inp=inp)
        confidence, label = torch.max(probs, dim=1)
        return label, confidence

    def _classify_tta(self, inputs: List[Tensor]) -> Tensor:
        # todo
        bs, n_tta = inputs[0].shape[0], len(inputs)
        input_tensor = torch.cat(inputs)
        logits = self._model(input_tensor)

        # now calc averaged by augmentations logit for each sample in batch
        probs_avg = torch.zeros([bs, self._n_classes], dtype=torch.float)
        for i in range(bs):
            ii = np.arange(start=i, stop=n_tta * bs, step=bs)
            probs = softmax(logits[ii, :], dim=1)
            probs_avg[i, :] = torch.mean(probs, dim=0)
        return probs_avg

    def _classify_simple(self, inp: Tensor) -> Tensor:
        logits = self._model(inp)
        probs = softmax(logits, dim=1)
        return probs

    def _adopt_arch(self) -> None:
        if self.arch in [Arch.RESNET18, Arch.RESNET34, Arch.RESNET50, Arch.INCEPTION]:
            self._model.avgpool = nn.AdaptiveAvgPool2d(1)
            in_dim, out_dim = self._model.fc.in_features, self._n_classes
            self._model.fc = nn.Linear(in_dim, out_dim)

        elif self.arch in [Arch.VGG11]:
            fc = self._model.classifier[-1]
            in_dim, out_dim = fc.in_features, fc.out_features
            self._model.classifier[-1] = nn.Linear(in_dim, out_dim)

        else:
            raise ValueError(f'Unexpected architecture {self.arch}.')

    def save(self, path: Path, meta: Any) -> None:
        checkpoint = {
            'state_dict': self._model.state_dict(),
            'meta': meta
        }
        torch.save(checkpoint, path)
        logger.info(f'Model saved to {path}.')

    def load(self, path_to_ckpt: Path) -> Any:
        checkpoint = torch.load(path_to_ckpt)
        self._model.load_state_dict(checkpoint['state_dict'])
        meta = checkpoint['meta']
        return meta
