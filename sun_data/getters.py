import json
from pathlib import Path

import pandas as pd

_SUN_SPECIAL_WORDS = ['indoor', 'outdoor', 'exterior', 'interior']


def beutify_name(sun_name):
    name = Path(sun_name).name
    if name in _SUN_SPECIAL_WORDS:
        name = Path(sun_name).parent.name
    return name


def get_name_to_enum(mapping_type):
    avai_types = ['classic', 'domains']
    assert mapping_type in avai_types, \
        f'Unexpected type on mapping: {mapping_type}, type must be in {avai_types}'

    jpath = Path(__file__).parent / 'files' / f'{mapping_type}' / 'NameToEnum.json'

    with open(jpath) as j:
        name_to_enum = dict(json.load(j))

    return name_to_enum


def get_split_paths(split_name):
    """

    :param split_name: must be 'classic_01', 'classis_02' ... 'classic_10
                       or 'domains'

    :return: paths to tables for training and testing
    """
    assert 'classic' in split_name or split_name == 'domains'

    if 'classic' in split_name:
        i_split = split_name.split('_')[-1]
        train_name, test_name = f'Training_{i_split}.csv', f'Testing_{i_split}.csv'
        split_dir = Path(__file__).parent / 'files' / 'classis' / 'splits'
        train_p, test_p = split_dir / train_name, split_dir / test_name

    else:
        split_dir = Path(__file__).parent / 'files' / 'domains' / 'splits'
        train_p, test_p = split_dir / 'train.csv', split_dir / 'test.csv'

    return train_p, test_p


# CLASSIC

def get_names(need_beutify=True):
    file_path = Path(__file__).parent / 'files' / 'classic' / 'SunClassName.txt'
    names = pd.read_csv(file_path, header=None, names=['names'])['names']

    if need_beutify:
        names = [beutify_name(name) for name in names]

    return names


# DOMAIN BASED

def get_domains():
    file_path = Path(__file__).parent / 'files' / 'domains' / 'SunClassDomains.txt'
    with open(file_path, 'r') as f:
        names = f.readlines()

    names = [name.replace('\n', '') for name in names]
    return names
