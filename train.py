import logging
import time
from argparse import Namespace, ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch

from common import beutify_args, Stopper, fix_seed
from datasets import ImagesDataset
from network import Classifier
from sun_data.utils import DataMode, load_data
from trainer import Trainer

logger = logging.getLogger(__name__)


def main(args: Namespace) -> float:
    start = time.time()
    board_dir, ckpt_dir = setup_logging(args.log_dir)
    logger.info(f'Params: \n{beutify_args(args)}')

    fix_seed(args.seed)

    (train_paths, train_names), (test_paths, test_names), name_to_enum = \
        load_data(args.data_mode)

    train_labels = [name_to_enum[name] for name in train_names]
    test_labels = [name_to_enum[name] for name in test_names]

    train_set = ImagesDataset(args.data_root, train_paths, train_labels)
    test_set = ImagesDataset(args.data_root, test_paths, test_labels)

    n_classes = len(name_to_enum)
    logger.info(f'Number of classes: {n_classes}.')

    classifier = Classifier(args.arch, n_classes, args.pretrained)
    stopper = Stopper(n_obs=args.n_stopper_obs, delta=args.n_stopper_delta)

    trainer = Trainer(classifier=classifier, board_dir=board_dir,
                      train_set=train_set, test_set=test_set,
                      name_to_enum=name_to_enum, device=args.device,
                      batch_size=args.batch_size, n_workers=args.n_workers,
                      use_train_aug=args.use_train_aug)

    max_metric = trainer.train(n_max_epoch=args.n_max_epoch, test_freq=args.test_freq,
                               n_tta=args.n_tta, stopper=stopper, ckpt_dir=ckpt_dir)

    logger.info(f'Elapsed time: {round((time.time() - start)/60, 3)} min.')
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
    parser.add_argument('-d', '--data_root', dest='data_root', type=Path)
    parser.add_argument('-w', '--log_dir', dest='log_dir', type=Path)
    parser.add_argument('--data_mode', dest='data_mode', type=DataMode,
                        default=DataMode.TAGS, help=f'One mode from {DataMode}.')

    parser.add_argument('--arch', dest='arch', type=str, default='resnet34')
    parser.add_argument('--use_train_aug', dest='use_train_aug', type=bool, default=True)
    parser.add_argument('--pretrained', dest='pretrained', type=bool, default=True)
    parser.add_argument('--n_max_epoch', dest='n_max_epoch', type=int, default=50)
    parser.add_argument('--test_freq', dest='test_freq', type=int, default=1)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=190)
    parser.add_argument('--n_tta', dest='n_tta', type=int, default=8)
    parser.add_argument('--n_workers', dest='n_workers', type=int, default=4)
    parser.add_argument('--device', dest='device', type=torch.device, default='cuda:2')
    parser.add_argument('--random_seed', dest='seed', type=int, default=42)

    parser.add_argument('--n_stopper_obs', dest='n_stopper_obs', type=int, default=500,
                        help='Number of epochs without metrics improving before stop.')
    parser.add_argument('--n_stopper_delta', dest='n_stopper_delta', type=float,
                        default=0.005, help='We assume that the metric improved only'
                                            'if it increased by a delta.')
    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    main(args=arg_parser.parse_args())
