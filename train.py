import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

from datasets.sun import SceneDataset
from models.meta import Classifier
from trainer import Trainer


def main(args):
    # make folds
    log_fold = args.work_dir / 'log'
    ckpt_fold = args.work_dir / 'checkpoints'
    board_fold = args.work_dir / 'board'
    for fold in [args.work_dir, log_fold, ckpt_fold, board_fold]:
        fold.mkdir(exist_ok=True)

    # logging
    log_file = log_fold / 'train.log'
    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])
    logging.info(f'Start train \n {args}')

    train_set = SceneDataset(data_path=args.data_path,
                             csv_path=args.tables_dir / args.train_table
                             )
    test_set = SceneDataset(data_path=args.data_path,
                            csv_path=args.tables_dir / args.test_table
                            )

    n_classes = train_set.get_num_classes()
    classifier = Classifier(args.arch, n_classes, args.pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.model.parameters())

    device = args.device if torch.cuda.is_available() else torch.device('cpu')
    logging.getLogger().info(f'Using device: {device}')

    trainer = Trainer(classifier=classifier,
                      work_dir=args.work_dir,
                      train_set=train_set,
                      test_set=test_set,
                      batch_size=args.batch_size,
                      n_workers=args.n_workers,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device,
                      test_freq=args.test_freq
                      )
    max_metric = trainer.train(n_max_epoch=args.n_max_epoch)
    return max_metric


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', dest='data_path', type=Path)
    parser.add_argument('-t', '--tables_dir', dest='tables_dir', type=Path)
    parser.add_argument('-w', '--work_dir', dest='work_dir', type=Path)
    parser.add_argument('-d', '--device', dest='device', type=torch.device, default='cuda:0')
    parser.add_argument('-a', '--arch', dest='arch', type=str, default='resnet18')
    parser.add_argument('-e', '--n_max_epoch', dest='n_max_epoch', type=int, default=100)
    parser.add_argument('-f', '--test_freq', dest='test_freq', type=int, default=1)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=256)
    parser.add_argument('-n', '--n_workers', dest='n_workers', type=int, default=6)
    parser.add_argument('-m', '--train_table', dest='train_table', type=str, default='Training_01.csv')
    parser.add_argument('-l', '--test_table', dest='test_table', type=str, default='Testing_01.csv')
    parser.add_argument('-g', '--pretrained', dest='pretrained', type=bool, default=True)
    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    main(args=arg_parser.parse_args())
