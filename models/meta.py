import logging

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
            raise ValueError(f'Unexpected type {arch}')

        self.logger.info(f'\n Selected architecture: {arch}')

        self._adopt_num_classes()
        self._adopt_pooling()

    def classify(self, image):
        self.model.eval()
        logits = self.model(image)
        probs = softmax(logits)
        confidence, label = torch.max(probs, dim=1)
        return label, confidence

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
