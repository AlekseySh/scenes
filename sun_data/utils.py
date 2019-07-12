from enum import Enum
from pathlib import Path
from random import shuffle
from typing import List, Tuple, Dict, Union

import pandas as pd
from bidict import bidict

__all__ = ['DataMode', 'load_data', 'get_sun_names',
           'beutify_name', 'get_name_to_enum',
           'get_hierarchy_mappings']

FILES_DIR = Path(__file__).parent / 'files'
SPLITS_DIR = FILES_DIR / 'splits'

TLabeled = Tuple[List[Path], List[str]]


class DataMode(Enum):
    CLASSIC_01 = 'classic_01'
    CLASSIC_02 = 'classic_02'
    CLASSIC_03 = 'classic_03'
    CLASSIC_04 = 'classic_04'
    CLASSIC_05 = 'classic_05'
    CLASSIC_06 = 'classic_06'
    CLASSIC_07 = 'classic_07'
    CLASSIC_08 = 'classic_08'
    CLASSIC_09 = 'classic_09'
    CLASSIC_10 = 'classic_10'
    TAGS = 'tags'


def get_classic_modes() -> List[DataMode]:
    classic_list = [DataMode(f'classic_0{i}') for i in range(1, 10)]
    classic_list.append(DataMode('classic_10'))
    return classic_list


def load_data(mode: DataMode) -> Tuple[TLabeled, TLabeled, bidict]:
    train_paths, test_paths = get_split(mode=mode)
    train_names = names_from_paths(train_paths, mode=mode)
    test_names = names_from_paths(test_paths, mode=mode)

    train_data = train_paths, train_names
    test_data = test_paths, test_names

    name_to_enum = get_name_to_enum(mode)
    return train_data, test_data, name_to_enum


# SPLITS

def load_file(file_path: Path) -> List[Path]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    paths = [Path(line[:-1]) for line in lines]  # remove \n
    return paths


def get_split(mode: DataMode) -> Tuple[List[Path], List[Path]]:
    classic_modes = get_classic_modes()

    if mode in classic_modes:
        train_paths, test_paths = get_split_classic(mode)

    elif mode == DataMode.TAGS:
        paths = []
        for cur_mode in classic_modes:
            train_paths_cur, test_paths_cur = get_split_classic(cur_mode)
            paths.extend(train_paths_cur)
            paths.extend(test_paths_cur)

        paths = list(set(paths))  # remove dublicated paths
        shuffle(paths)
        n_train = int(0.8 * len(paths))
        train_paths, test_paths = paths[:n_train], paths[n_train:]

    else:
        raise ValueError(f'Unexpected data mode {mode}.')

    return train_paths, test_paths


def get_split_classic(mode: DataMode) -> Tuple[List[Path], List[Path]]:
    assert mode in get_classic_modes()

    split_num = str(mode).split('_')[-1]
    train_paths = load_file(SPLITS_DIR / f'Training_{split_num}.txt')
    test_paths = load_file(SPLITS_DIR / f'Testing_{split_num}.txt')
    return train_paths, test_paths


# MAPPINGS

def names_from_paths(im_paths: List[Path], mode: DataMode) -> List[str]:
    sun_names = [str(im_path.parent) for im_path in im_paths]
    if mode in get_classic_modes():
        names = sun_names

    elif mode == mode.TAGS:
        sun_to_tag = get_sun_to_tags_mapping()
        names = [sun_to_tag[name] for name in sun_names]

    else:
        raise ValueError(f'Unexpected mode {mode}.')

    return names


def get_sun_names(need_beutify: bool = False) -> List[str]:
    sun_paths = load_file(FILES_DIR / 'SunClassNames.txt')
    if need_beutify:
        sun_names = [beutify_name(name) for name in sun_paths]
    else:
        sun_names = [str(name) for name in sun_paths]
    return sun_names


def get_name_to_enum(mode: DataMode) -> bidict:
    if mode in get_classic_modes():
        sun_classes = get_sun_names(need_beutify=False)
        name_to_enum = bidict({str(name): i for i, name in enumerate(sun_classes)})

    elif mode == DataMode.TAGS:
        sun_to_tags = get_sun_to_tags_mapping()
        tags = sorted(set(sun_to_tags.values()))
        name_to_enum = bidict({tag: i for i, tag in enumerate(tags)})

    else:
        raise ValueError(f'Unexpected mode {mode}.')

    return name_to_enum


def get_sun_to_tags_mapping() -> Dict[str, str]:
    df = pd.read_csv(FILES_DIR / 'mapping.csv')
    sun_to_tag = {}
    for (sun_class, tag) in zip(df['raw_names'], df['tags']):
        tag = 'other' if pd.isna(tag) else tag
        sun_to_tag.update({sun_class: tag})
    return sun_to_tag


def get_hierarchy_mappings() -> List[Path]:
    hier_dir = FILES_DIR / 'hierarchy'
    return [hier_dir / 'level0.csv', hier_dir / 'level1.csv']


# RANDOM

def beutify_name(sun_name: Union[Path, str]) -> str:
    sun_special_words = [
        'indoor', 'outdoor', 'exterior', 'interior'
    ]

    if isinstance(sun_name, str):
        sun_name = Path(sun_name)

    beutified_name = sun_name.name
    if str(beutified_name) in sun_special_words:
        beutified_name = sun_name.parent.name
    return str(beutified_name)
