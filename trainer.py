import logging
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as t
from bidict import bidict
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.nn.functional import softmax
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from tqdm import tqdm

from common import OnlineAvg, Stopper, confusion_matrix_as_img, histogram_as_img
from datasets import ImagesDataset, SIZE
from metrics import Calculator
from network import Classifier, Arch
from sun_data.utils import beutify_name

logger = logging.getLogger(__name__)


class Mode(Enum):
    TRAIN = 'train'
    TEST = 'test'


class Trainer:
    _classifier: Classifier
    _board_dir: Path
    _train_set: ImagesDataset
    _test_set: ImagesDataset
    _name_to_enum: bidict
    _device: torch.device
    _batch_size: int
    _num_workers: int
    _use_train_aug: bool

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
                 n_workers: int,
                 use_train_aug: bool
                 ):

        self._classifier = classifier
        self._board_dir = board_dir
        self._train_set = train_set
        self._test_set = test_set
        self._name_to_enum = name_to_enum
        self._device = device
        self._batch_size = batch_size
        self._num_workers = n_workers
        self._use_train_aug = use_train_aug

        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self._classifier.parameters(), lr=1e-1)
        self._writer = SummaryWriter(str(self._board_dir))

        self._i_global = 0
        self._classifier.to(self._device)

    def train_epoch(self) -> float:
        self._classifier.train()
        if self._use_train_aug:
            self._train_set.set_train_transforms()
        else:
            self._train_set.set_default_transforms()

        loader = DataLoader(dataset=self._train_set,
                            batch_size=self._batch_size,
                            num_workers=self._num_workers,
                            shuffle=True, drop_last=True
                            )

        avg_loss = OnlineAvg()
        loader_tqdm = tqdm(loader, total=len(loader))

        gts: List[int] = []
        preds: List[int] = []
        probs: List[float] = []
        for im, label in loader_tqdm:
            self._optimizer.zero_grad()

            if self._classifier.arch == Arch.INCEPTION3:
                logits, aux_output = self._classifier(im.to(self._device))
                loss1 = self._criterion(logits, label.to(self._device))
                loss2 = self._criterion(aux_output, label.to(self._device))
                loss = loss1 + .4 * loss2

            else:
                logits = self._classifier(im.to(self._device))
                loss = self._criterion(logits, label.to(self._device))

            loss_data = loss.detach().cpu().numpy()
            self._writer.add_scalar('Loss', loss_data, self._i_global)

            loss.backward()
            self._optimizer.step()

            max_logits, ii_max = logits.max(dim=1)
            prob = softmax(max_logits, dim=0).detach().cpu().numpy().tolist()
            pred = ii_max.detach().cpu().numpy().tolist()

            gts.extend(label)
            preds.extend(pred)
            probs.extend(prob)

            avg_loss.update(loss_data)
            loader_tqdm.set_postfix({'Avg loss': round(float(avg_loss.avg), 4)})
            self._writer.add_scalar('Loss', loss_data, self._i_global)
            self._i_global += 1

        main_metric = self._log_metrics(gts, preds, probs, Mode.TRAIN, img_verbose=True)
        return main_metric

    def test(self, n_tta: int) -> float:
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

        loader = DataLoader(dataset=self._test_set, batch_size=batch_size_tta,
                            num_workers=self._num_workers, shuffle=False)

        gts: List[int] = []
        preds: List[int] = []
        probs: List[float] = []
        with torch.no_grad():
            for i, (im, label) in tqdm(enumerate(loader), total=len(loader)):
                pred, prob = self._classifier.classify(to_device(im))

                pred = pred.detach().cpu().numpy().tolist()
                prob = prob.detach().cpu().numpy().tolist()

                gts.extend(label)
                preds.extend(pred)
                probs.extend(prob)

        main_metric = self._log_metrics(gts, preds, probs, Mode.TEST, img_verbose=True)
        return main_metric

    def train(self,
              n_max_epoch: int,
              test_freq: int,
              n_tta: int,
              stopper: Stopper,
              ckpt_dir: Path
              ) -> float:

        self._visualize_hist()

        best_ckpt_path = ckpt_dir / 'best.pth.tar'
        acc_max: float = 0
        best_epoch: int = 0
        for i in range(n_max_epoch):
            # train
            logger.info(f'\n\nTrain. Epoch {i} from {n_max_epoch}')
            self.train_epoch()

            if i % test_freq == 0:
                # test
                acc = self.test(n_tta=0)

                # save model
                save_path = ckpt_dir / f'epoch{i}.pth.tar'
                self._classifier.save(save_path, meta={'acc_w': acc})

                if acc > acc_max:
                    acc_max, best_epoch = acc, i
                    self._classifier.save(best_ckpt_path, meta={'acc_w': acc})

                stopper.update(acc)
                if stopper.check_criterion():
                    logger.info(f'Stopped by criterion. Reached {i} epoch of {n_max_epoch}')
                    break

        logger.info(f'Max metric {acc_max} reached at {best_epoch} epoch.')

        if n_tta > 0:
            self._classifier, _ = Classifier.from_ckpt(best_ckpt_path)
            logger.info('Try improve this value with TTA:')
            acc_tta = self.test(n_tta=n_tta)
            logger.info(f'Metric value with TTA: {acc_tta}')
            return max(acc_max, acc_tta)

        else:
            return acc_max

    def _log_metrics(self,
                     gts: List[int],
                     preds: List[int],
                     probs: List[float],
                     mode: Mode,
                     img_verbose: bool = False
                     ) -> float:
        gts, preds, probs = np.array(gts), np.array(preds), np.array(probs)

        mc = Calculator(gts=gts, preds=preds, confidences=probs)
        metrics = mc.calc()

        for name, val in metrics.items():
            logger.info(f'{name}: {val}')
            self._writer.add_scalar(f'{mode}_{name}', val, self._i_global)

        if img_verbose:
            ii_worst, ii_best = mc.worst_errors(n_worst=8), mc.best_preds(n_best=8)
            self._visualize_preds(ids=ii_worst, enums_pred=preds[ii_worst], mode=mode, tag='worst_mistakes')
            self._visualize_preds(ids=ii_best, enums_pred=preds[ii_best], mode=mode, tag='best_predicts')
            self._visualize_confusion(preds=preds, gts=gts)

        main_metric = metrics['accuracy_weighted']
        return main_metric

    def _visualize_preds(self,
                         ids: np.ndarray,
                         enums_pred: np.ndarray,
                         mode: Mode,
                         tag: str
                         ) -> None:
        if len(ids) == 0:
            return

        if mode == Mode.TRAIN:
            dataset = self._train_set
        elif mode == Mode.TEST:
            dataset = self._test_set
        else:
            raise ValueError(f'Unexpected mode {mode}.')

        assert max(ids) < len(dataset)

        dataset.set_default_transforms()

        base_color, gt_color, err_color = (0, 0, 0), (0, 255, 0), (255, 0, 0)
        n_gt_samples, n_pred_samples = 2, 2

        layer_tensor = torch.zeros([0, 3, SIZE[0], SIZE[1]], dtype=torch.uint8)
        for (idx, enum_pred) in zip(ids, enums_pred):
            _, enum_gt = dataset[idx]
            name_gt = beutify_name(self._name_to_enum.inv[enum_gt])
            name_pred = beutify_name(self._name_to_enum.inv[enum_pred])

            main_img = dataset.get_signed_image(text=[f'pred: {name_pred}', f'gt: {name_gt}'],
                                                idx=idx, color=base_color)

            gt_imgs = dataset.draw_class_samples(
                n_samples=n_gt_samples, class_num=enum_gt, color=gt_color, text=[name_gt])

            pred_color = gt_color if enum_gt == enum_pred else err_color
            pred_imgs = dataset.draw_class_samples(
                n_samples=n_pred_samples, class_num=enum_pred, color=pred_color, text=[name_pred])

            layer_tensor = torch.cat(
                [layer_tensor, main_img.unsqueeze(dim=0), gt_imgs, pred_imgs], dim=0)

        grid = vutils.make_grid(tensor=layer_tensor, nrow=n_gt_samples + n_pred_samples + 1,
                                normalize=False, scale_each=False)

        self._writer.add_image(img_tensor=grid, global_step=self._i_global, tag=f'{mode}/{tag}')

    def _visualize_confusion(self, preds: np.ndarray, gts: np.ndarray) -> None:
        class_names = [self._name_to_enum.inv[num] for num in range(0, len(self._name_to_enum))]
        conf_mat = confusion_matrix_as_img(gts=gts, preds=preds, classes=class_names)
        self._writer.add_image(global_step=self._i_global, tag='Confusion_matrix.',
                               img_tensor=t.ToTensor()(conf_mat))

    def _visualize_hist(self) -> None:
        labels_enum = self._train_set.labels_enum.copy()
        labels_enum.extend(self._test_set.labels_enum.copy())
        names = [self._name_to_enum.inv[enum] for enum in labels_enum]
        histogram = histogram_as_img(names)
        self._writer.add_image(global_step=self._i_global, tag='Histogram.',
                               img_tensor=t.ToTensor()(histogram))
