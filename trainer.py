import logging
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchvision import utils as vutils
from tqdm import tqdm

from common import OnlineAvg
from datasets import SIZE
from metrics import Calculator
from model import Classifier
from sun_data.utils import get_mapping, beutify_name

logger = logging.getLogger(__name__)


class Stopper:

    def __init__(self, n_obs, delta):
        self.n_obs = n_obs
        self.delta = delta

        self.cur_val = None
        self.max_val = 0
        self.num_fails = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self._count_fails()
        self._update_max()

    def _count_fails(self):
        if self.cur_val - self.max_val <= self.delta:
            self.num_fails += 1
        else:
            self.num_fails = 0

    def check_criterion(self):
        is_stop = self.num_fails == self.n_obs
        return is_stop

    def _update_max(self):
        if self.max_val < self.cur_val:
            self.max_val = self.cur_val


class Trainer:

    def __init__(self,
                 classifier: Classifier,
                 work_dir: Path,
                 train_set: Dataset,
                 test_set: Dataset,
                 batch_size: int,
                 n_workers: int,
                 criterion,
                 optimizer,
                 device: torch.device,
                 test_freq: int
                 ):

        self.classifier = classifier
        self.work_dir = work_dir
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.test_freq = test_freq

        self.i_global = 0
        self.best_ckpt_path = self.work_dir / 'checkpoints' / 'best.pth.tar'
        self.writer = SummaryWriter(self.work_dir / 'board')

        self.classifier.model.to(self.device)

    def train_epoch(self):
        self.classifier.model.train()
        loader = DataLoader(dataset=self.train_set,
                            batch_size=self.batch_size,
                            num_workers=self.n_workers,
                            shuffle=True
                            )
        self.train_set.set_default_transforms()

        avg_loss = OnlineAvg()
        for data in tqdm(loader, total=len(loader)):
            self.optimizer.zero_grad()

            image = data['image'].to(self.device)
            label = data['label'].to(self.device)

            logits = self.classifier.model(image)
            loss = self.criterion(logits, label)
            loss.backward()
            self.optimizer.step()

            avg_loss.update(loss)
            self.i_global += 1
            self.writer.add_scalar('Loss', loss.data, self.i_global)

        logger.info(f'Loss: {avg_loss.avg}')
        self.writer.add_scalar('AvgLoss', avg_loss.avg, self.i_global)

    def test(self, n_tta: int):
        if n_tta != 0:
            self.test_set.set_test_transforms(n_augs=n_tta)
            batch_size = int(self.batch_size / n_tta)

            def to_device(x_arr):
                return [x.to(self.device) for x in x_arr]
        else:
            batch_size = self.batch_size
            self.test_set.set_default_transforms()

            def to_device(x):
                return x.to(self.device)

        loader = DataLoader(dataset=self.test_set,
                            batch_size=batch_size,
                            num_workers=self.n_workers,
                            shuffle=False
                            )
        n_samples = len(loader.dataset)

        labels = np.zeros(n_samples, np.int)
        preds = np.zeros_like(labels)
        confs = np.zeros_like(labels, dtype=np.float)

        for i, data in tqdm(enumerate(loader), total=len(loader)):
            im = to_device(data['image'])
            label = data['label'].numpy()

            pred, conf = self.classifier.classify(im)

            pred = pred.detach().cpu().numpy()
            conf = conf.detach().cpu().numpy()

            i_start = i * loader.batch_size
            i_stop = min(i_start + loader.batch_size, n_samples)

            labels[i_start: i_stop] = label
            preds[i_start: i_stop] = pred
            confs[i_start: i_stop] = conf

        mc = Calculator(gt=labels, pred=preds, score=confs)
        metrics = mc.calc()
        ii_worst = mc.find_worst_mistakes(n_worst=5)
        self.visualize_errors(ii_worst=ii_worst, labels_pred=preds[ii_worst])
        self.writer.add_scalar('Accuracy', metrics['accuracy'], self.i_global)
        return metrics

    def train(self, n_max_epoch: int, n_tta: int, stopper: Stopper):
        acc_max, best_epoch = 0, 0
        for i in range(n_max_epoch):
            # train
            logger.info(f'Epoch {i} from {n_max_epoch}')
            self.train_epoch()

            if i % self.test_freq == 0:
                # test
                acc = self.test(n_tta=0)['accuracy']

                # save model
                save_path = self.work_dir / 'checkpoints' / f'epoch{i}.pth.tar'
                self.classifier.save(save_path, meta={'acc': acc})
                logger.info(f'Accuracy: {acc}')

                if acc > acc_max:
                    acc_max, best_epoch = acc, i
                    self.classifier.save(self.best_ckpt_path, meta={'acc': acc})

                stopper.update(acc)
                if stopper.check_criterion():
                    logger.info(f'Training stoped by criterion. Reached {i} epoch of {n_max_epoch}')
                    break

        logger.info(f'Max metric {acc_max} reached at {best_epoch} epoch.')
        logger.info('Try improve this value with TTA:')

        self.classifier.load(self.best_ckpt_path)
        acc_tta = self.test(n_tta=n_tta)['accuracy']
        logger.info(f'Metric value with TTA: {acc_tta}')
        return max(acc_max, acc_tta)

    def visualize_errors(self, ii_worst, labels_pred):
        main_color = (0, 0, 0)
        n_gt_samples, gt_color = 2, (0, 255, 0)
        n_pred_samples, pred_color = 2, (255, 0, 0)

        name_to_enum = get_mapping('DomainToEnum')  # todo

        layour_tensor = torch.zeros([0, 3, SIZE[0], SIZE[1]], dtype=torch.uint8)

        for (ind, label_pred) in zip(ii_worst, labels_pred):
            label_gt = self.test_set[ind]['label']
            name_gt = beutify_name(name_to_enum.inv[label_gt])
            name_pred = beutify_name(name_to_enum.inv[label_pred])

            main_img = self.test_set.get_signed_image(idx=ind,
                                                      text=[f'pred: {name_pred}', f'gt: {name_gt}'],
                                                      color=main_color
                                                      )

            gt_imgs = self.test_set.draw_class_samples(
                n_samples=n_gt_samples, label=int(label_gt), color=gt_color)

            pred_imgs = self.test_set.draw_class_samples(
                n_samples=n_pred_samples, label=int(label_pred), color=pred_color)

            layour_tensor = torch.cat(
                [layour_tensor, main_img.unsqueeze(dim=0), gt_imgs, pred_imgs], dim=0)

        grid = vutils.make_grid(tensor=layour_tensor,
                                nrow=n_gt_samples + n_pred_samples + 1,
                                normalize=False,
                                scale_each=True
                                )

        self.writer.add_image('Worst_mistakes', grid, self.i_global)
