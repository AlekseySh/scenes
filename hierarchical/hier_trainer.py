import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import OnlineAvg
from hierarchical.hier_dataset import HierDataset_MultiLabel
from hierarchical.hier_metrics import calc_accuracy_batch
from hierarchical.hier_network import Classifier

logger = logging.getLogger(__name__)


class Trainer:
    _classifier: Classifier
    _train_set: HierDataset_MultiLabel
    _test_set: HierDataset_MultiLabel

    _bce: BCEWithLogitsLoss
    _device: torch.device
    _optimizer: torch.optim.Optimizer
    _writer: SummaryWriter
    _i_global: int
    _cur_metric: OnlineAvg

    def __init__(self,
                 classifier: Classifier,
                 train_set: HierDataset_MultiLabel,
                 test_set: HierDataset_MultiLabel,
                 board_dir: Path,
                 ):
        self._classifier = classifier
        self._train_set = train_set
        self._test_set = test_set

        self._bce = BCEWithLogitsLoss()
        self._device = torch.device('cuda:2')
        self._optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-1)

        self._writer = SummaryWriter(log_dir=board_dir)
        self._cur_metric = OnlineAvg()
        self._i_global = 0

        self._classifier.to(self._device)

    def data_loop(self) -> None:
        loader = DataLoader(dataset=self._train_set, batch_size=180,
                            shuffle=True, num_workers=4, drop_last=True)

        loader_tqdm = tqdm(loader)
        for images, one_hot_arr in loader_tqdm:
            self._optimizer.zero_grad()

            logits_list = self._classifier(images.to(self._device))

            losses = []
            gts, probs, preds = [], [], []
            for logits, one_hot in zip(logits_list, one_hot_arr):
                # loss = self._bce(input=logits.to(self._device).type(torch.float32),
                #                  target=one_hot.to(self._device).type(torch.float32)
                #                  )
                ind = one_hot.nonzero()[:, -1]
                # print(ind)
                loss = torch.nn.CrossEntropyLoss()(
                    logits.to(self._device), ind.to(self._device)
                )
                loss.backward(retain_graph=True)

                losses.append(float(loss.detach().cpu()))

                logits = logits.detach().cpu().numpy()
                preds.extend(np.argmax(logits, axis=1).tolist())
                gts.extend(ind.detach().cpu().numpy().tolist())

                probs.extend([1] * 180)

            from metrics import Calculator
            gts = np.array(gts)
            preds = np.array(preds)
            probs = np.array(probs)
            calc = Calculator(gts=gts, preds=preds, confidences=probs)
            print(calc.calc())

            self._optimizer.step()
            self._i_global += 1

            levels = loader.dataset.levels
            postf_loss = self.log_losses(losses=losses, levels=levels)
            postf_metric = self.log_metrics(logits_list=logits_list, levels=levels, one_hot_arr=one_hot_arr)
            # loader_tqdm.set_postfix(ordered_dict={**postf_loss, **postf_metric})

            self._cur_metric.update(postf_metric['acc_avg'])

        logger.info(f'\nAccuracy: {round(self._cur_metric.avg, 3)}')
        self._cur_metric.clear()

    def train(self, n_epoch: int) -> None:
        for i in range(n_epoch):
            self.data_loop()

            logger.info(f'Epoch {i + 1} / {n_epoch}')

    def log_losses(self, losses: List[float], levels: Tuple[int, ...]) -> Dict[str, float]:
        postfix = {}
        for loss, level in zip(losses, levels):
            name, val = f'loss{level}', round(loss, 3)
            self._writer.add_scalar(name, val, self._i_global)
            postfix[name] = val

        name, val = 'loss_sum', sum(losses)
        self._writer.add_scalar(name, val, self._i_global)
        postfix[name] = val

        return postfix

    def log_metrics(self,
                    logits_list: Tuple[Tensor, ...],
                    one_hot_arr: Tuple[Tensor, ...],
                    levels: Tuple[int, ...]
                    ) -> Dict[str, float]:

        acc_list = calc_accuracy_batch(logits_list=logits_list, one_hot_arr=one_hot_arr)

        postfix = {}
        for acc, level in zip(acc_list, levels):
            name, val = f'acc{level}', round(acc, 3)
            self._writer.add_scalar(name, val, self._i_global)
            postfix[name] = val

        name, val = 'acc_avg', round(sum(acc_list) / len(acc_list), 3)
        self._writer.add_scalar(name, val, self._i_global)
        postfix[name] = val

        return postfix
