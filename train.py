import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

from datasets.table import SceneDataset
from models.network import Classifier
from trainer import Trainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_data', dest='data_path', type=Path)
    parser.add_argument('-d', '--device', dest='device', default='cuda:0', type=torch.device)
    parser.add_argument('-a', '--architecture', dest='arch', default='resnet18', type=str)
    parser.add_argument('-e', '--n_epoch', dest='n_epoch', default=50, type=int)
    parser.add_argument('-f', '--test_freq', dest='test_freq', default=1, type=int)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else torch.device('cpu')
    total_set = SceneDataset(data_path=args.data_path, csv_name='small.csv')
    n_classes = total_set.get_num_classes()
    classifier = Classifier(arch=args.arch, n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.model.parameters())

    trainer = Trainer(classifier=classifier,
                      train_set=total_set,
                      test_set=total_set,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device,
                      n_epoch=args.n_epoch,
                      test_freq=args.test_freq
                      )
    trainer.train()
