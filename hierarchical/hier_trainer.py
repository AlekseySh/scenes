import logging
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import OnlineAvg
from hierarchical.hier_dataset import HierDataset
from hierarchical.hier_network import Classifier
from metrics import Calculator

logger = logging.getLogger(__name__)


class Mode(Enum):
    TRAIN = 'train'
    TEST = 'test'

    @property
    def s(self):
        return str(self)[5:].lower()


class Trainer:
    _classifier: Classifier
    _train_set: HierDataset
    _test_set: HierDataset

    _cross_entropy: CrossEntropyLoss
    _device: torch.device
    _optimizer: torch.optim.Optimizer
    _writer: SummaryWriter
    _i_global: int

    def __init__(self,
                 classifier: Classifier,
                 train_set: HierDataset,
                 test_set: HierDataset,
                 board_dir: Path,
                 ):
        self._classifier = classifier
        self._train_set = train_set
        self._test_set = test_set

        self._cross_entropy = CrossEntropyLoss()
        self._device = torch.device('cuda:1')
        self._optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-1)

        self._writer = SummaryWriter(log_dir=board_dir)
        self._cur_metric = OnlineAvg()
        self._cur_loss = OnlineAvg()
        self._i_global = 0

        self._classifier.to(self._device)

    def data_loop(self, mode: Mode) -> None:
        if mode == Mode.TRAIN:
            dataset, shuffle, drop_last = self._train_set, True, True
            self._classifier.set_train_mode()

        elif mode == Mode.TEST:
            dataset, shuffle, drop_last = self._test_set, False, False
            self._classifier.set_test_mode()

        else:
            raise ValueError(f'Unexpected mode {mode}.')

        loader = DataLoader(dataset=dataset, shuffle=shuffle, drop_last=drop_last,
                            batch_size=180, num_workers=4)

        loader_tqdm = tqdm(loader)
        acc_avg, loss_avg = OnlineAvg(), OnlineAvg()
        for images, enums_arr in loader_tqdm:

            if mode == Mode.TRAIN:
                self._optimizer.zero_grad()

            logits_list = self._classifier(images.to(self._device))

            losses, metrics = [], []
            for logits, enums in zip(logits_list, enums_arr):

                loss = self._cross_entropy(logits, enums.to(self._device))

                loss_v = float(loss.detach().cpu())
                acc = Calculator(preds=np.argmax(logits.detach().cpu().numpy(), axis=1),
                                 probs=np.argmax(softmax(logits, dim=1).detach().cpu().numpy(), axis=1),
                                 gts=enums.detach().cpu().numpy()
                                 ).calc()['accuracy_weighted']

                losses.append(loss_v)
                metrics.append(acc)

                if mode == Mode.TRAIN:
                    loss.backward(retain_graph=True)

            if mode == Mode.TRAIN:
                self._optimizer.step()
                self._i_global += 1

            acc_avg.update(np.mean(metrics))
            loss_avg.update(np.mean(losses))

            postf_loss = self.log_values(losses, loader.dataset.levels, tag=f'{mode.s}/loss')
            postf_metric = self.log_values(metrics, loader.dataset.levels, tag=f'{mode.s}/acc')
            loader_tqdm.set_postfix(ordered_dict={f'{mode.s}/acc_avg': acc_avg.avg,
                                                  f'{mode.s}/loss_avg': loss_avg.avg,
                                                  **postf_loss, **postf_metric})

        logger.info(f'Average loss: {round(loss_avg.avg, 3)}')
        logger.info(f'Average accuracy: {round(acc_avg.avg, 3)}')

    def train(self, n_epoch: int) -> None:
        for i in range(n_epoch):
            logger.info(f'\nEpoch {i + 1} / {n_epoch}:')

            mode = Mode.TRAIN
            self.data_loop(mode)

            mode = Mode.TEST
            self.data_loop(mode)

            logger.info('Epoch ended\n\n')

    def log_values(self,
                   values: List[float],
                   levels: Tuple[int, ...],
                   tag: str
                   ) -> Dict[str, float]:
        postfix = {}
        for loss, level in zip(values, levels):
            name, val = f'{tag}{level}', round(loss, 3)
            self._writer.add_scalar(name, val, self._i_global)
            postfix[name] = val

        name, val = f'{tag}_total', np.mean(values)
        self._writer.add_scalar(name, val, self._i_global)
        postfix[name] = val

        return postfix
