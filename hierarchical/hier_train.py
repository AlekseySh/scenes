from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

from hierarchical.hier_dataset import HierDataset
from hierarchical.hier_network import Classifier
from hierarchical.hier_trainer import Trainer
from sun_data.utils import DataMode, load_data
from sun_data.utils import get_hierarchy_mappings
from train import setup_logging


def get_datasets(data_root: Path, data_mode: DataMode, levels: Tuple[int, ...]
                 ) -> Tuple[HierDataset, HierDataset]:
    (train_paths, train_names), (test_paths, test_names), _ = \
        load_data(data_mode)

    common_args = {
        'levels': levels,
        'hier_mappings': get_hierarchy_mappings(),
        'data_root': data_root
    }
    train_set = HierDataset(im_paths=train_paths, classes_bot=train_names,
                            **common_args)

    test_set = HierDataset(im_paths=test_paths, classes_bot=test_names,
                           **common_args)

    train_set.set_train_transforms(aug_degree=3)
    test_set.set_test_transforms()
    return train_set, test_set


def main(args: Namespace) -> None:
    assert 'classic' in str(args.data_mode).lower()
    train_set, test_set = get_datasets(data_root=args.data_root,
                                       data_mode=args.data_mode,
                                       levels=args.levels)

    level_sizes = train_set.hierarchy.get_levels_sizes(levels=args.levels)

    classifier = Classifier(level_sizes)

    board_dir, _ = setup_logging(log_dir=args.log_dir)
    trainer = Trainer(classifier=classifier, board_dir=board_dir,
                      train_set=train_set, test_set=test_set)

    trainer.train(n_epoch=args.n_epoch)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--data_root', dest='data_root', type=Path)
    parser.add_argument('--log_dir', dest='log_dir', type=Path)

    parser.add_argument('--levels', dest='levels', type=Tuple[int, ...], default=tuple([2]))
    parser.add_argument('--data_mode', dest='data_mode', type=DataMode,
                        default=DataMode.CLASSIC_01, help=f'One from classic modes.')

    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=100)
    return parser


if __name__ == '__main__':
    main(args=get_parser().parse_args())
