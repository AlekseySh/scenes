from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.network import Classifier


class Trainer:

    def __init__(self,
                 classifier: Classifier,
                 train_set: Dataset,
                 test_set: Dataset,
                 loader_args: dict,
                 criterion,
                 optimizer,
                 device: torch.device,
                 n_epoch: int,
                 test_freq: int
                 ):

        self.classifier = classifier
        self.train_set = train_set
        self.test_set = test_set
        self.loader_args = loader_args
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.n_epoch = n_epoch
        self.test_freq = test_freq

        self.classifier.model.to(self.device)

    def train_epoch(self):
        loader = DataLoader(dataset=self.train_set,
                            shuffle=True,
                            **self.loader_args
                            )

        self.classifier.model.train()
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            self.optimizer.zero_grad()

            image = data['image'].to(self.device)
            label = data['label'].to(self.device)

            logits = self.classifier.model(image)
            loss = self.criterion(logits, label)
            loss.backward()
            self.optimizer.step()

    def test(self):
        loader = DataLoader(dataset=self.test_set,
                            shuffle=False,
                            **self.loader_args
                            )

        n_samples = len(loader.dataset)
        labels = np.zeros([n_samples, 1], np.int)
        preds = np.zeros_like(labels)
        confs = np.zeros_like(labels, dtype=np.float)
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            image = data['image'].to(self.device)
            label = data['label']

            pred, conf = self.classifier.classify(image=image)
            pred = pred.detach().cpu().numpy()
            conf = conf.detach().cpu().numpy()

            i_start = i * loader.batch_size
            i_stop = min(i_start + loader.batch_size, n_samples)

            # todo
            labels[i_start: i_stop] = np.reshape(label, [len(label), 1])
            preds[i_start: i_stop] = np.reshape(pred, [len(pred), 1])
            confs[i_start: i_stop] = np.reshape(conf, [len(conf), 1])

        # todo
        acc = np.sum(labels == preds) / n_samples
        return acc

    def train(self):
        for i in range(self.n_epoch):
            self.train_epoch()

            if i % self.test_freq == 0:
                acc = self.test()
                print(f'\n Accuracy: {acc}')
