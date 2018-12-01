import logging

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import softmax


class Classifier:

    def __init__(self, arch, n_classes, pretrained):
        self.n_classes = n_classes
        self.logger = logging.getLogger()

        if arch == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif arch == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        else:
            # noinspection Annotator
            raise ValueError(f'Unexpected type {arch}')

        self.logger.info(f'Selected architecture: {arch}')

        self._adopt_num_classes()
        self._adopt_pooling()

    def classify(self, inp):
        self.model.eval()
        if isinstance(inp, list):
            logits = self._classify_tta(inp)
        else:
            logits = self.model(inp)
        probs = softmax(logits, dim=1)
        confidence, label = torch.max(probs, dim=1)
        return label, confidence

    def _classify_tta(self, tensor_list):
        bs, n_tta = tensor_list[0].shape[0], len(tensor_list)
        input_tensor = torch.cat(tensor_list)
        logits = self.model(input_tensor)

        # now calc averaged by augmentations logit for each sample in batch
        logits_avg = torch.zeros([bs, self.n_classes], dtype=torch.float)
        for i in range(bs):
            ii = np.arange(start=i, stop=n_tta * bs, step=bs)
            logits_avg[i, :] = torch.mean(logits[ii, :], dim=0)
        return logits_avg

    def classify_tta(self, image, tta):
        raise NotImplemented()

    def _adopt_num_classes(self):
        input_dim = self.model.fc.in_features
        output_dim = self.n_classes
        self.model.fc = nn.Linear(input_dim, output_dim)

    def _adopt_pooling(self):
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)

    def save(self, path, meta=None):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'meta': meta
        }
        torch.save(checkpoint, path)
        self.logger.info(f'Model saved to {path}.')

    def load(self, path_to_ckpt):
        checkpoint = torch.load(path_to_ckpt)
        self.model.load_state_dict(checkpoint['state_dict'])
        meta = checkpoint['meta']
        return meta
