import json
from pathlib import Path

import numpy as np
from bidict import bidict
from sklearn.preprocessing import LabelEncoder

_SUN_SPECIAL_WORDS = ['indoor', 'outdoor', 'exterior', 'interior']
_FILE_DIR = Path(__file__).parent / 'files'


def beutify_name(sun_name):
    name = Path(sun_name).name
    if name in _SUN_SPECIAL_WORDS:
        name = Path(sun_name).parent.name
    return name


def get_domains():
    file_path = _FILE_DIR / 'SunClassDomains.txt'
    with open(file_path, 'r') as f:
        names = f.readlines()

    names = [name.replace('\n', '') for name in names]
    return names


def get_mapping(a_to_b: str):
    jpath = _FILE_DIR / f'{a_to_b}.json'

    with open(jpath) as j:
        name_to_enum = bidict(json.load(j))

    return name_to_enum


def get_split_csv_paths(split_name):
    """

    :param split_name: must be 'classic_01', 'classis_02' ... 'classic_10
                       or 'domains'

    :return: paths to tables for training and testing
    """
    assert 'classic' in split_name or split_name == 'domains'

    if 'classic' in split_name:
        i_split = split_name.split('_')[-1]
        train_name, test_name = f'Training_{i_split}.csv', f'Testing_{i_split}.csv'
        train_csv_path = _FILE_DIR / 'split_classic' / train_name
        test_csv_path = _FILE_DIR / 'split_classic' / test_name

    else:
        train_csv_path = _FILE_DIR / 'split_domains' / 'train.csv'
        test_csv_path = _FILE_DIR / 'split_domains' / 'test.csv'

    return train_csv_path, test_csv_path


def save_domains_mappings():
    raw_names = np.array(get_names(need_beutify=False))
    domains = np.array(get_domains())

    w_exist = domains != ''
    domains_exist = domains[w_exist]

    # name to domain
    name_to_domain = dict(zip(raw_names[w_exist], domains_exist))
    name_to_domain_path = _FILE_DIR / 'NameToDomain.json'

    with open(name_to_domain_path, 'w') as j:
        json.dump(fp=j, obj=name_to_domain)

    # domain to enum
    enum_domains = LabelEncoder().fit_transform(domains_exist).tolist()
    domain_to_enum = dict(zip(domains_exist, enum_domains))
    domain_to_enum_path = _FILE_DIR / 'DomainToEnum.json'

    with open(domain_to_enum_path, 'w') as j:
        json.dump(fp=j, obj=domain_to_enum)


if __name__ == '__main__':
    save_domains_mappings()
