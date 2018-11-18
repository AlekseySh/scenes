import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

from datasets.table import SceneDataset
from models.network import Classifier
from trainer import Trainer


def main(args):

    train_set = SceneDataset(data_path=args.data_path, csv_path=args.tables_dir / 'Training_05.csv')
    test_set = SceneDataset(data_path=args.data_path, csv_path=args.tables_dir / 'Testing_05.csv')

    n_classes = train_set.get_num_classes()
    classifier = Classifier(arch=args.arch, n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.model.parameters())
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    loader_args = {'batch_size': args.batch_size, 'num_workers': args.n_workers}

    trainer = Trainer(classifier=classifier,
                      train_set=train_set,
                      test_set=test_set,
                      loader_args=loader_args,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device,
                      n_epoch=args.n_epoch,
                      test_freq=args.test_freq
                      )
    trainer.train()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_data', dest='data_path', type=Path)
    parser.add_argument('-t', '--path_to_tables', dest='tables_dir', type=Path)
    parser.add_argument('-d', '--device', dest='device', type=torch.device, default='cuda:0')
    parser.add_argument('-a', '--architecture', dest='arch', type=str, default='resnet18')
    parser.add_argument('-e', '--n_epoch', dest='n_epoch', type=int, default=100)
    parser.add_argument('-f', '--test_freq', dest='test_freq', type=int, default=1)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('-w', '--n_workers', dest='n_workers', type=int, default=6)
    return parser


if __name__ == '__main__':

    arg_parser = get_parser()
    params = arg_parser.parse_args()

    main(args=params)
