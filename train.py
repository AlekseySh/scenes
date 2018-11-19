import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

from datasets.table import SceneDataset
from models.meta import Classifier
from trainer import Trainer


def main(args):

    train_csv = args.tables_dir / f'Training_{args.split_id}.csv'
    test_csv = args.tables_dir /  f'Testing_{args.split_id}.csv'
    train_set = SceneDataset(args.data_path, train_csv)
    test_set = SceneDataset(args.data_path, test_csv)

    n_classes = train_set.get_num_classes()
    classifier = Classifier(arch=args.arch, n_classes=n_classes, pretrained=args.pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.model.parameters())

    device = args.device if torch.cuda.is_available() else torch.device('cpu')
    logging.getLogger().info(f'\n Using device: {device}')

    loader_args = {'batch_size': args.batch_size, 'num_workers': args.n_workers}

    trainer = Trainer(classifier=classifier,
                      work_dir=args.work_dir,
                      train_set=train_set,
                      test_set=test_set,
                      loader_args=loader_args,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device,
                      n_epoch=args.n_epoch,
                      test_freq=args.test_freq
                      )
    acc = trainer.train()
    return acc

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_data', dest='data_path', type=Path)
    parser.add_argument('-t', '--path_to_tables', dest='tables_dir', type=Path)
    parser.add_argument('-w', '--work_dir', dest='work_dir', type=Path)
    parser.add_argument('-d', '--device', dest='device', type=torch.device, default='cuda:0')
    parser.add_argument('-a', '--architecture', dest='arch', type=str, default='resnet18')
    parser.add_argument('-e', '--n_epoch', dest='n_epoch', type=int, default=100)
    parser.add_argument('-f', '--test_freq', dest='test_freq', type=int, default=1)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('-n', '--n_workers', dest='n_workers', type=int, default=6)
    parser.add_argument('-i', '--split_id', dest='split_id', type=str, default='01')
    parser.add_argument('-g', '--is_pretrained', dest='pretrained', type=bool, default=True)
    return parser


if __name__ == '__main__':

    arg_parser = get_parser()
    params = arg_parser.parse_args()

    # make folds
    log_fold = params.work_dir / 'log'
    ckpt_fold = params.work_dir / 'checkpoints'
    board_fold = params.work_dir / 'board'
    for fold in [params.work_dir, log_fold, ckpt_fold, board_fold]:
        fold.mkdir(exist_ok=True)

    # logging
    log_file = log_fold / 'train.log'
    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])
    logging.info(f'\n Start train \n {params} \n')

    main(args=params)
