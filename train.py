import logging
from argparse import Namespace, ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch

from common import beutify_args, Stopper
from datasets import ImagesDataset as ImSet
from network import Classifier
from sun_data.common import get_mapping
from sun_data.utils import get_split_csv_paths
from trainer import Trainer

logger = logging.getLogger(__name__)


def main(args: Namespace) -> float:
    board_dir, ckpt_dir = setup_logging(args.log_dir)
    logger.info(f'Params: \n{beutify_args(args)}')

    train_csv, test_csv = get_split_csv_paths(args.split_name)
    train_set = ImSet(data_fold=args.data_path, csv_path=train_csv)
    test_set = ImSet(data_fold=args.data_path, csv_path=test_csv)

    n_classes = train_set.get_num_classes()
    classifier = Classifier(args.arch, n_classes, args.pretrained)
    stopper = Stopper(args.n_stopper_obs, args.n_stopper_delta)

    if 'classic' in args.split_name:
        name_to_enum = get_mapping('NameToEnum')
    else:
        name_to_enum = get_mapping('DomainToEnum')

    trainer = Trainer(classifier=classifier, board_dir=board_dir,
                      train_set=train_set, test_set=test_set,
                      name_to_enum=name_to_enum, device=args.device,
                      batch_size=args.batch_size)

    max_metric = trainer.train(n_max_epoch=args.n_max_epoch, test_freq=args.test_freq,
                               n_tta=args.n_tta, stopper=stopper, ckpt_dir=ckpt_dir)
    return max_metric


def setup_logging(log_dir: Path) -> Tuple[Path, Path]:
    experiment_dir = log_dir / str(datetime.now())
    ckpt_dir = experiment_dir / 'checkpoints'
    board_dir = experiment_dir / 'board'
    for fold in [experiment_dir, ckpt_dir, board_dir]:
        fold.mkdir(exist_ok=True, parents=True)

    log_file = experiment_dir / 'log.txt'
    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])
    return board_dir, ckpt_dir


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=Path)
    parser.add_argument('-w', '--log_dir', dest='log_dir', type=Path)

    # with default values
    parser.add_argument('--split', dest='split_name', type=str, default='classic_01',
                        help='must be <classic_01>, <classis_02> ... <classic_10> or <domains>'
                        )
    parser.add_argument('--device', dest='device', type=torch.device, default='cuda:1')
    parser.add_argument('--arch', dest='arch', type=str, default='resnet18')
    parser.add_argument('--n_max_epoch', dest='n_max_epoch', type=int, default=5)
    parser.add_argument('--test_freq', dest='test_freq', type=int, default=1)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=256)
    parser.add_argument('--n_tta', dest='n_tta', type=int, default=8)
    parser.add_argument('--n_workers', dest='n_workers', type=int, default=6)
    parser.add_argument('--n_stopper_obs', dest='n_stopper_obs', type=int, default=5)
    parser.add_argument('--n_stopper_delta', dest='n_stopper_delta', type=float, default=0.005)
    parser.add_argument('--pretrained', dest='pretrained', type=bool, default=True)
    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    main(args=arg_parser.parse_args())
