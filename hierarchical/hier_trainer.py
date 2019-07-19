import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, DefaultDict

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import OnlineAvg
from common import Stopper
from hierarchical.hier_dataset import HierDataset
from hierarchical.hier_network import Classifier
from metrics import Calculator

logger = logging.getLogger(__name__)


def round3(x: float) -> float:
    return round(x, 3)


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
    _device: torch.device

    _cross_entropy: CrossEntropyLoss
    _optimizer: torch.optim.Optimizer
    _writer: SummaryWriter
    _i_global: int

    def __init__(self,
                 classifier: Classifier,
                 train_set: HierDataset,
                 test_set: HierDataset,
                 board_dir: Path,
                 device: torch.device
                 ):
        self._classifier = classifier
        self._train_set = train_set
        self._test_set = test_set
        self._device = device

        self._cross_entropy = CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-1)

        self._writer = SummaryWriter(log_dir=board_dir)
        self._i_global = 0

        self._classifier.to(self._device)

    def data_loop(self, mode: Mode) -> float:
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
        avg_values = defaultdict(lambda: OnlineAvg())
        for images, enums_arr in loader_tqdm:
            logits_list = self._classifier(images.to(self._device))

            if mode == Mode.TRAIN:
                self._optimizer.zero_grad()

            for logits, enums, level in zip(logits_list, enums_arr, dataset.levels):
                # loss
                loss = self._cross_entropy(logits, enums.to(self._device))
                loss_tag, loss_v = f'{mode.s}/loss{level}', float(loss.detach().cpu())
                avg_values[loss_tag].update(loss_v)
                self._writer.add_scalar(loss_tag, loss_v, self._i_global)

                # metric
                acc = Calculator(preds=np.argmax(logits.detach().cpu().numpy(), axis=1),
                                 probs=np.argmax(softmax(logits, dim=1).detach().cpu().numpy(), axis=1),
                                 gts=enums.detach().cpu().numpy()
                                 ).calc()['accuracy_weighted']
                avg_values[f'{mode.s}/acc{level}'].update(acc)

                if mode == Mode.TRAIN:
                    loss.backward(retain_graph=True)

            if mode == Mode.TRAIN:
                self._optimizer.step()
                self._i_global += 1

            loader_tqdm.set_postfix(self.make_postfix(avg_values, mode))

        avg_acc_tag = f'{mode.s}/acc'
        avg_acc = avg_values[avg_acc_tag].avg
        self._writer.add_scalar(avg_acc_tag, avg_acc, self._i_global)

        return avg_acc

    def train(self, n_epoch: int) -> None:
        max_acc = 0
        stopper = Stopper(n_obs=15, delta=0.005)
        scheduler = CosineAnnealingLR(self._optimizer, T_max=n_epoch, eta_min=1e-3)

        for i in range(n_epoch):
            logger.info(f'\nEpoch {i + 1} / {n_epoch}:')

            self.data_loop(Mode.TRAIN)
            scheduler.step()
            self._writer.add_scalar('lr', scheduler.get_lr()[0], self._i_global)

            acc = self.data_loop(Mode.TEST)
            max_acc = acc if acc > max_acc else max_acc

            stopper.update(acc)
            if stopper.check_criterion():
                logger.info(f'Early stop by criterion. Reached {i} epoch of {n_epoch}')
                logger.info(f'Resulted accuracy: {max_acc}')
                break

    @staticmethod
    def make_postfix(avg_values: DefaultDict[str, OnlineAvg], mode: Mode) -> Dict[str, float]:
        postfix = {}
        losses, metrics = [], []
        for name, avg_val in avg_values.items():
            postfix[name] = avg_val.avg

            if 'loss' in name:
                losses.append(avg_val.avg)

            if 'acc' in name:
                metrics.append(avg_val.avg)

        postfix[f'{mode.s}/loss'] = np.mean(losses)
        postfix[f'{mode.s}/acc'] = np.mean(metrics)

        return postfix
