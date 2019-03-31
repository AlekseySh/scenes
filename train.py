import logging
from argparse import Namespace, ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from common import beutify_args, Stopper
from datasets import ImagesDataset as ImSet
from network import Classifier
from sun_data.common import get_mapping
from sun_data.utils import get_split_csv_paths
from trainer import Trainer


def main(args: Namespace) -> float:
    # make folds
    work_dir = args.work_root / str(datetime.now())
    log_fold = work_dir / 'log'
    ckpt_fold = work_dir / 'checkpoints'
    board_fold = work_dir / 'board'
    for fold in [work_dir, log_fold, ckpt_fold, board_fold]:
        fold.mkdir(exist_ok=True)

    # logging
    log_file = log_fold / 'train.log'
    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])
    logger = logging.getLogger(__name__)
    logger.info(f'Params: \n{beutify_args(args)}')

    train_csv, test_csv = get_split_csv_paths(args.split_name)
    train_set = ImSet(data_fold=args.data_path, csv_path=train_csv)
    test_set = ImSet(data_fold=args.data_path, csv_path=test_csv)

    n_classes = train_set.get_num_classes()
    classifier = Classifier(args.arch, n_classes, args.pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    stopper = Stopper(args.n_stopper_obs, args.n_stopper_delta)

    if 'classic' in args.split_name:
        name_to_enum = get_mapping('NameToEnum')
    else:
        name_to_enum = get_mapping('DomainToEnum')

    trainer = Trainer(classifier=classifier,
                      work_dir=work_dir,
                      train_set=train_set,
                      test_set=test_set,
                      name_to_enum=name_to_enum,
                      batch_size=args.batch_size,
                      n_workers=args.n_workers,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=args.device,
                      test_freq=args.test_freq
                      )

    max_metric = trainer.train(n_max_epoch=args.n_max_epoch,
                               n_tta=args.n_tta,
                               stopper=stopper
                               )
    return max_metric


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=Path)
    parser.add_argument('-w', '--work_root', dest='work_root', type=Path)

    # with default values
    parser.add_argument('--split', dest='split_name', type=str, default='classic_01',
                        help='must be <classic_01>, <classis_02> ... <classic_10> or <domains>'
                        )
    parser.add_argument('--device', dest='device', type=torch.device, default='cuda:1')
    parser.add_argument('--arch', dest='arch', type=str, default='resnet18')
    parser.add_argument('--n_max_epoch', dest='n_max_epoch', type=int, default=5)
    parser.add_argument('--test_freq', dest='test_freq', type=int, default=1)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--n_tta', dest='n_tta', type=int, default=8)
    parser.add_argument('--n_workers', dest='n_workers', type=int, default=6)
    parser.add_argument('--n_stopper_obs', dest='n_stopper_obs', type=int, default=5)
    parser.add_argument('--n_stopper_delta', dest='n_stopper_delta', type=float, default=0.005)
    parser.add_argument('--pretrained', dest='pretrained', type=bool, default=True)
    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    main(args=arg_parser.parse_args())
