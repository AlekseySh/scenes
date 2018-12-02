import logging
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from metrics.classification import MetricsCalculator
from models.meta import Classifier
from utils.common import OnlineAvg, Stopper

logger = logging.getLogger(__name__)


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

    def test(self, use_tta: bool):
        if use_tta:
            n_tta = 8
            self.test_set.set_test_transforms(n_augs=n_tta)
            batch_size = int(self.batch_size / n_tta)
            def to_device(x_arr): return [x.to(self.device) for x in x_arr]
        else:
            batch_size = self.batch_size
            self.test_set.set_default_transforms()
            def to_device(x): return x.to(self.device)

        loader = DataLoader(dataset=self.test_set,
                            batch_size=batch_size,
                            num_workers=self.n_workers,
                            shuffle=False
                            )
        n_samples = len(loader.dataset)

        labels = np.zeros([n_samples, 1], np.int)
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

            labels[i_start: i_stop] = np.asmatrix(label).T
            preds[i_start: i_stop] = np.asmatrix(pred).T
            confs[i_start: i_stop] = np.asmatrix(conf).T

        mc = MetricsCalculator(gt=labels, pred=preds, score=confs)
        metrics = mc.calc()
        self.writer.add_scalar('Accuracy', metrics['accuracy'], self.i_global)
        return metrics

    def train(self, n_max_epoch):
        stopper = Stopper(n_observation=8, delta=0.005)
        acc_max, best_epoch = 0, 0
        for i in range(n_max_epoch):
            # train
            logger.info(f'Epoch {i} from {n_max_epoch}')
            self.train_epoch()

            if i % self.test_freq == 0:
                # test
                acc = self.test(use_tta=False)['accuracy']
                accccccc = self.test(use_tta=True)['accuracy']
                logger.info(f'accccc: {accccccc}')

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
        acc_tta = self.test(use_tta=True)['accuracy']
        logger.info(f'Metric value with TTA: {acc_tta}')
        return max(acc_max, acc_tta)
