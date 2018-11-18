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
    parser.add_argument('-t', '--path_to_tables', dest='tables_dir', type=Path)
    parser.add_argument('-d', '--device', dest='device', type=torch.device, default='cuda:0')
    parser.add_argument('-a', '--architecture', dest='arch', type=str, default='resnet18')
    parser.add_argument('-e', '--n_epoch', dest='n_epoch', type=int, default=100)
    parser.add_argument('-f', '--test_freq', dest='test_freq', type=int, default=1)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    train_set = SceneDataset(data_path=args.data_path, csv_path=args.tables_dir / 'Training_05.csv')
    test_set = SceneDataset(data_path=args.data_path, csv_path=args.tables_dir / 'Testing_05.csv')

    n_classes = train_set.get_num_classes()
    classifier = Classifier(arch=args.arch, n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.model.parameters())

    trainer = Trainer(classifier=classifier,
                      train_set=train_set,
                      test_set=test_set,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device,
                      n_epoch=args.n_epoch,
                      test_freq=args.test_freq
                      )
    trainer.train()
