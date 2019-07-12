from typing import List

import torch
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from hierarchical.hier_dataset import HierDataset
from hierarchical.hier_network import Classifier


class Trainer:
    _classifier: Classifier
    _train_set: HierDataset
    _test_set: HierDataset

    _bce: BCEWithLogitsLoss
    _device: torch.device
    _optimizer: torch.optim.Optimizer
    _writer: SummaryWriter
    _i_global: int

    def __init__(self,
                 classifier: Classifier,
                 train_set: HierDataset,
                 test_set: HierDataset
                 ):
        self._classifier = classifier
        self._train_set = train_set
        self._test_set = test_set

        self._bce = BCEWithLogitsLoss()
        self._device = torch.device('cuda:1')
        self._optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-2)

        self._writer = SummaryWriter()
        self._i_global = 0

        self._classifier.to(self._device)

    def train_epoch(self) -> None:
        loader = DataLoader(dataset=self._train_set, batch_size=128,
                            shuffle=True, num_workers=4)

        loader_tqdm = tqdm(loader)
        for images, one_hot_arr in loader_tqdm:
            self._optimizer.zero_grad()

            logits_list = self._classifier(images.to(self._device))

            losses = []
            for i, (logits, one_hot) in enumerate(zip(logits_list, one_hot_arr)):
                loss = self._bce(input=logits.to(self._device).type(torch.float32),
                                 target=one_hot.to(self._device).type(torch.float32)
                                 )
                loss.backward()
                losses.append(float(loss.detach().cpu()))

            self._optimizer.step()

            self.log_losses(loader_tqdm, losses)

    def train(self, n_epoch: int) -> None:
        for _ in range(n_epoch):
            self.train_epoch()

    def log_losses(self, loader_tqdm, losses: List[float]) -> None:
        postfix = {}
        for i, loss in enumerate(losses):
            name, val = f'loss{i}', round(loss, 3)
            self._writer.add_scalar(name, val, self._i_global)
            postfix[name] = val

        name, val = 'loss_sum', sum(losses)
        self._writer.add_scalar(name, val, self._i_global)
        postfix[name] = val

        loader_tqdm.set_postfix(postfix)
