import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from bidict import bidict
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from tqdm import tqdm

from common import OnlineAvg, Stopper
from datasets import ImagesDataset
from datasets import SIZE
from metrics import Calculator
from network import Classifier
from sun_data.utils import beutify_name

logger = logging.getLogger(__name__)


class Trainer:
    _classifier: Classifier
    _board_dir: Path
    _train_set: ImagesDataset
    _test_set: ImagesDataset
    _name_to_enum: bidict
    _device: torch.device
    _batch_size: int
    _num_workers: int

    _criterion: nn.Module
    _optimizer: Optimizer
    _writer: SummaryWriter

    def __init__(self,
                 classifier: Classifier,
                 board_dir: Path,
                 train_set: ImagesDataset,
                 test_set: ImagesDataset,
                 name_to_enum: bidict,
                 device: torch.device,
                 batch_size: int,
                 ):

        self._classifier = classifier
        self._board_dir = board_dir
        self._train_set = train_set
        self._test_set = test_set
        self._name_to_enum = name_to_enum
        self._device = device
        self._batch_size = batch_size

        self._num_workers = 4
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self._classifier.parameters(), lr=1e-4)
        self._writer = SummaryWriter(str(self._board_dir))

        self._i_global = 0
        self._classifier.to(self._device)

    def train_epoch(self) -> None:
        self._classifier.train()
        loader = DataLoader(dataset=self._train_set,
                            batch_size=self._batch_size,
                            num_workers=self._num_workers,
                            shuffle=True
                            )
        self._train_set.set_default_transforms()

        avg_loss = OnlineAvg()
        loader_tqdm = tqdm(loader, total=len(loader))
        for im, label in loader_tqdm:
            self._optimizer.zero_grad()

            logits = self._classifier(im.to(self._device))
            loss = self._criterion(logits, label.to(self._device))
            loss.backward()
            self._optimizer.step()

            avg_loss.update(loss.data)
            loss_val = round(float(avg_loss.avg.detach().cpu()), 4)
            loader_tqdm.set_postfix({'Avg loss': loss_val})

            self._writer.add_scalar('Loss', loss_val, self._i_global)
            self._i_global += 1

    def test(self, n_tta: int) -> Dict[str, float]:
        if n_tta != 0:
            self._test_set.set_test_transforms(n_augs=n_tta)
            batch_size_tta = int(self._batch_size / n_tta)

            def to_device(x_arr):
                return [x.to(self._device) for x in x_arr]
        else:
            batch_size_tta = self._batch_size
            self._test_set.set_default_transforms()

            def to_device(x):
                return x.to(self._device)

        loader = DataLoader(dataset=self._test_set,
                            batch_size=batch_size_tta,
                            num_workers=self._num_workers,
                            shuffle=False
                            )
        n_samples = len(loader.dataset)

        labels = np.zeros(n_samples, dtype=np.int)
        preds = np.zeros_like(labels)
        confs = np.zeros_like(labels, dtype=np.float)
        for i, (im, label) in tqdm(enumerate(loader), total=len(loader)):
            pred, conf = self._classifier.classify(to_device(im))

            pred = pred.detach().cpu().numpy()
            conf = conf.detach().cpu().numpy()

            i_start = i * loader.batch_size
            i_stop = min(i_start + loader.batch_size, n_samples)

            labels[i_start: i_stop] = label.numpy()
            preds[i_start: i_stop] = pred
            confs[i_start: i_stop] = conf

        mc = Calculator(gts=labels, preds=preds, confidences=confs)
        metrics = mc.calc()
        ii_worst = mc.find_worst_mistakes(n_worst=8)
        ii_best = mc.find_best_predicts(n_best=8)
        self.visualize(ii_worst, preds[ii_worst], 'Worst_mistakes')
        self.visualize(ii_best, preds[ii_best], 'Best_predicts')
        self._writer.add_scalar('Accuracy', metrics['accuracy'], self._i_global)
        return metrics

    def train(self,
              n_max_epoch: int,
              test_freq: int,
              n_tta: int,
              stopper: Stopper,
              ckpt_dir: Path
              ) -> float:
        best_ckpt_path = ckpt_dir / 'best.pth.tar'
        acc_max, best_epoch = 0, 0
        for i in range(n_max_epoch):
            # train
            logger.info(f'Epoch {i} from {n_max_epoch}')
            self.train_epoch()

            if i % test_freq == 0:
                # test
                acc = self.test(n_tta=0)['accuracy']

                # save model
                save_path = ckpt_dir / f'epoch{i}.pth.tar'
                self._classifier.save(save_path, meta={'acc': acc})
                logger.info(f'Accuracy: {acc}')

                if acc > acc_max:
                    acc_max, best_epoch = acc, i
                    self._classifier.save(best_ckpt_path, meta={'acc': acc})

                stopper.update(acc)
                if stopper.check_criterion():
                    logger.info(f'Training stoped by criterion. Reached {i} epoch of {n_max_epoch}')
                    break

        logger.info(f'Max metric {acc_max} reached at {best_epoch} epoch.')
        logger.info('Try improve this value with TTA:')

        self._classifier.load(best_ckpt_path)
        acc_tta = self.test(n_tta=n_tta)['accuracy']
        logger.info(f'Metric value with TTA: {acc_tta}')
        return max(acc_max, acc_tta)

    def visualize(self, indeces: List[int], labels_pred, board_tag: str) -> None:
        base_color = (0, 0, 0)
        gt_color = (0, 255, 0)
        err_color = (255, 0, 0)
        n_gt_samples, n_pred_samples = 2, 2

        layour_tensor = torch.zeros([0, 3, SIZE[0], SIZE[1]], dtype=torch.uint8)

        for (idx, label_pred) in zip(indeces, labels_pred):
            _, label_gt = self._test_set[idx]
            name_gt = beutify_name(self._name_to_enum.inv[label_gt])
            name_pred = beutify_name(self._name_to_enum.inv[label_pred])

            main_img = self._test_set.get_signed_image(idx=idx,
                                                       text=[f'pred: {name_pred}', f'gt: {name_gt}'],
                                                       color=base_color
                                                       )

            gt_imgs = self._test_set.draw_class_samples(
                n_samples=n_gt_samples, class_label=int(label_gt), color=gt_color)

            pred_color = gt_color if label_gt == label_pred else err_color
            pred_imgs = self._test_set.draw_class_samples(
                n_samples=n_pred_samples, class_label=int(label_pred), color=pred_color)

            layour_tensor = torch.cat(
                [layour_tensor, main_img.unsqueeze(dim=0), gt_imgs, pred_imgs], dim=0)

        grid = vutils.make_grid(tensor=layour_tensor,
                                nrow=n_gt_samples + n_pred_samples + 1,
                                normalize=False,
                                scale_each=True
                                )

        self._writer.add_image(img_tensor=grid, global_step=self._i_global, tag=board_tag)
