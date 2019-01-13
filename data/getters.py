import json
from pathlib import Path

import pandas as pd
from bidict import bidict

_SUN_SPECIAL_WORDS = ['indoor', 'outdoor', 'exterior', 'interior']


def get_name_to_enum():
    jpath = Path(__file__).parent / 'files' / 'SunNameToEnum.json'
    with open(jpath) as j:
        name_to_enum = bidict(json.load(j))
    return name_to_enum


def get_sun_names(need_beutify=True):
    file_path = Path(__file__).parent / 'files' / 'SunClassName.txt'
    names = pd.read_csv(file_path, header=None, names=['names'])['names']

    if need_beutify:
        names = [_beutify_name(name) for name in names]

    return names


def _beutify_name(sun_name):
    name = Path(sun_name).name
    if name in _SUN_SPECIAL_WORDS:
        name = Path(sun_name).parent.name
    return name
