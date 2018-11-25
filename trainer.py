import logging
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils.common import OnlineAvg, Stopper
from tensorboardX import SummaryWriter

from models.meta import Classifier
from metrics.classification import MetricsCalculator


class Trainer:

    def __init__(self,
                 classifier: Classifier,
                 work_dir: Path,
                 train_set: Dataset,
                 test_set: Dataset,
                 loader_args: dict,
                 criterion,
                 optimizer,
                 device: torch.device,
                 n_max_epoch: int,
                 test_freq: int
                 ):

        self.classifier = classifier
        self.work_dir = work_dir
        self.train_set = train_set
        self.test_set = test_set
        self.loader_args = loader_args
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.n_max_epoch = n_max_epoch
        self.test_freq = test_freq

        self.i_global = 0
        self.logger = logging.getLogger()
        self.writer = SummaryWriter(self.work_dir / 'board')
        self.classifier.model.to(self.device)

    def train_epoch(self):
        self.classifier.model.train()
        loader = DataLoader(dataset=self.train_set,
                            shuffle=True,
                            **self.loader_args
                            )
        avg_loss = OnlineAvg()
        for i, data in tqdm(enumerate(loader), total=len(loader)):
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

        self.logger.info(f'\n Loss: {avg_loss.avg}')
        self.writer.add_scalar('AvgLoss', avg_loss.avg, self.i_global)

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
            label = data['label'].numpy()

            pred, conf = self.classifier.classify(image=image)
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

    def train(self):
        stopper = Stopper(n_observation=5, delta=1)  # todo
        max_metric = 0
        for i in range(self.n_max_epoch):
            self.logger.info(f'\n Epoch {i} from {self.n_epoch}')
            self.train_epoch()

            if i % self.test_freq == 0:
                metrics = self.test()
                acc = metrics['accuracy']

                # save model
                save_path = self.work_dir / 'checkpoints' / f'epoch{i}.pth.tar'
                self.classifier.save(save_path, meta=metrics)
                self.logger.info(f'\n Accuracy: {round(acc, 3)}')

                if acc > max_metric:
                    max_metric = acc
                    save_path_best = save_path / 'checkpoints' / 'best.pth.tar'
                    self.classifier.save(save_path_best, meta=metrics)

                stopper.update(acc)
                if stopper.check_criterion():
                    break
        return max_metric
