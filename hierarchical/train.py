from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

from hierarchical.hier_dataset import HierDataset
from hierarchical.hier_network import Classifier
from hierarchical.hier_trainer import Trainer
from sun_data.utils import DataMode, load_data
from sun_data.utils import get_hierarchy_mappings


def get_datasets(data_root: Path, data_mode: DataMode
                 ) -> Tuple[HierDataset, HierDataset]:
    (train_paths, train_names), (test_paths, test_names), _ = \
        load_data(data_mode)

    train_set = HierDataset(data_root=data_root, im_paths=train_paths,
                            classes_bot=train_names,
                            hierarchy_mappings=get_hierarchy_mappings())

    test_set = HierDataset(data_root=data_root, im_paths=test_paths,
                           classes_bot=test_names,
                           hierarchy_mappings=get_hierarchy_mappings())

    train_set.set_train_transforms()
    test_set.set_test_transforms()
    return train_set, test_set


def main(args: Namespace) -> None:
    assert 'classic' in str(args.data_mode).lower()
    train_set, test_set = get_datasets(data_root=args.data_root,
                                       data_mode=args.data_mode)

    level_sizes = train_set.hierarchy.get_level_sizes()
    classifier = Classifier(level_sizes)

    trainer = Trainer(classifier=classifier, train_set=train_set,
                      test_set=test_set)
    trainer.train()


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--data_root', dest='data_root', type=Path)

    parser.add_argument('--levels', dest='levels', type=Tuple[int, ...], default=[0, 1, 2])
    parser.add_argument('--data_mode', dest='data_mode', type=DataMode,
                        default=DataMode.CLASSIC_01, help=f'One from classic modes.')
    return parser


if __name__ == '__main__':
    main(args=get_parser().parse_args())
