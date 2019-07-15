from pathlib import Path
from typing import List, Tuple

import torch
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import OnlineAvg
from hierarchical.hier_dataset import HierDataset
from hierarchical.hier_metrics import calc_accuracy_batch
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
    _cur_metric: OnlineAvg

    def __init__(self,
                 classifier: Classifier,
                 train_set: HierDataset,
                 test_set: HierDataset,
                 board_dir: Path,
                 ):
        self._classifier = classifier
        self._train_set = train_set
        self._test_set = test_set

        self._bce = BCEWithLogitsLoss()
        self._device = torch.device('cuda:1')
        self._optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-2)

        self._writer = SummaryWriter(log_dir=board_dir)
        self._cur_metric = OnlineAvg()
        self._i_global = 0

        self._classifier.to(self._device)

    def data_loop(self) -> None:
        loader = DataLoader(dataset=self._train_set, batch_size=128,
                            shuffle=True, num_workers=4)

        loader_tqdm = tqdm(loader)
        for images, one_hot_arr in loader_tqdm:
            self._optimizer.zero_grad()

            logits_list = self._classifier(images.to(self._device))

            m = calc_accuracy_batch(logits_list=logits_list, one_hot_arr=one_hot_arr)
            print(m)

            losses = []
            for logits, one_hot in zip(logits_list, one_hot_arr):
                loss = self._bce(input=logits.to(self._device).type(torch.float32),
                                 target=one_hot.to(self._device).type(torch.float32)
                                 )
                loss.backward(retain_graph=True)
                losses.append(float(loss.detach().cpu()))

            self._optimizer.step()
            self._i_global += 1

            self.log_losses(loader_tqdm, losses=losses, levels=loader.dataset.levels)

        self._cur_metric = OnlineAvg()

    def train(self, n_epoch: int) -> None:
        for _ in range(n_epoch):
            self.data_loop()

    def log_losses(self, loader_tqdm, losses: List[float], levels: Tuple[int, ...]) -> None:
        postfix = {}
        for loss, level in zip(losses, levels):
            name, val = f'loss{level}', round(loss, 3)
            self._writer.add_scalar(name, val, self._i_global)
            postfix[name] = val

        name, val = 'loss_sum', sum(losses)
        self._writer.add_scalar(name, val, self._i_global)
        postfix[name] = val

        loader_tqdm.set_postfix(postfix)

    def log_metrics(self) -> None:
        return None
