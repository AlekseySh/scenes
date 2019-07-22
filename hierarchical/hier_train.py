from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple, Set, List

import torch

from common import fix_seed
from hierarchical.hier_dataset import HierDataset
from hierarchical.hier_network import Classifier
from hierarchical.hier_structure import Hierarchy
from hierarchical.hier_trainer import Trainer
from sun_data.utils import DataMode, load_data
from sun_data.utils import get_hierarchy_mappings
from train import setup_logging


def filter_data(paths: List[Path],
                classes: List[str],
                needed_classes: Set[str]
                ) -> Tuple[List[Path], List[str]]:
    paths_ret, classes_ret = [], []

    for path, cls in zip(paths, classes):

        if cls in needed_classes:
            paths_ret.append(path)
            classes_ret.append(cls)

    return paths_ret, classes_ret


def get_hier_data(data_root: Path,
                  data_mode: DataMode,
                  levels: Tuple[int, ...],
                  aug_degree: float
                  ) -> Tuple[HierDataset, HierDataset]:
    (train_paths, train_classes), (test_paths, test_classes), _ = load_data(data_mode)

    hierarchy = Hierarchy(mapping_files=get_hierarchy_mappings())
    classes_needed = hierarchy.get_intersected_classes()

    train_paths, train_classes = filter_data(train_paths, train_classes, classes_needed)
    test_paths, test_classes = filter_data(test_paths, test_classes, classes_needed)

    common_args = {'levels': levels, 'hierarchy': hierarchy, 'data_root': data_root}
    train_set = HierDataset(im_paths=train_paths, classes_bot=train_classes, **common_args)
    test_set = HierDataset(im_paths=test_paths, classes_bot=test_classes, **common_args)

    train_set.set_train_transforms(aug_degree=aug_degree)
    test_set.set_test_transforms()
    return train_set, test_set


def main(args: Namespace) -> None:
    assert 'classic' in str(args.data_mode).lower()

    fix_seed(args.random_seed)

    train_set, test_set = get_hier_data(data_root=args.data_root,
                                        data_mode=args.data_mode,
                                        levels=args.levels,
                                        aug_degree=args.aug_degree)

    level_sizes = train_set.hierarchy.get_levels_sizes(levels=args.levels)

    classifier = Classifier(level_sizes=level_sizes, splitted_heads=args.splitted_heads)

    board_dir, _ = setup_logging(log_dir=args.log_dir)
    trainer = Trainer(classifier=classifier, board_dir=board_dir,
                      train_set=train_set, test_set=test_set, device=args.device)

    trainer.train(n_epoch=args.n_epoch)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--data_root', dest='data_root', type=Path)
    parser.add_argument('--log_dir', dest='log_dir', type=Path)

    parser.add_argument('--levels', dest='levels', type=Tuple[int, ...], default=tuple([0, 1, 2]))
    parser.add_argument('--data_mode', dest='data_mode', type=DataMode,
                        default=DataMode.CLASSIC_01, help=f'One from classic modes.')

    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=100)
    parser.add_argument('--aug_degree', dest='aug_degree', type=float, default=2.5)
    parser.add_argument('--splitted_heads', dest='splitted_heads', type=bool, default=True)

    parser.add_argument('--device', dest='device', type=torch.device, default='cuda:1')
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=42)
    return parser


if __name__ == '__main__':
    main(args=get_parser().parse_args())
