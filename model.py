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
        elif arch == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
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
            probs = self._classify_tta(tensor_list=inp)
        else:
            probs = self._classify_simple(tensor=inp)
        confidence, label = torch.max(probs, dim=1)
        return label, confidence

    def _classify_tta(self, tensor_list):
        bs, n_tta = tensor_list[0].shape[0], len(tensor_list)
        input_tensor = torch.cat(tensor_list)
        logits = self.model(input_tensor)

        # now calc averaged by augmentations logit for each sample in batch
        probs_avg = torch.zeros([bs, self.n_classes], dtype=torch.float)
        for i in range(bs):
            ii = np.arange(start=i, stop=n_tta * bs, step=bs)
            probs = softmax(logits[ii, :], dim=1)
            probs_avg[i, :] = torch.mean(probs, dim=0)
        return probs_avg

    def _classify_simple(self, tensor):
        logits = self.model(tensor)
        probs = softmax(logits, dim=1)
        return probs

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
