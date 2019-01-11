import json
from pathlib import Path

import pandas as pd
from bidict import bidict


def get_name_enum_mapping():
    jpath = Path(__file__).parent / 'files' / 'SunNameToEnum.json'
    with open(jpath) as j:
        name_to_enum = bidict(json.load(j))
    return name_to_enum


def get_sun_tags():
    file_path = Path(__file__).parent / 'files' / 'SunClassName.txt'
    names_col = pd.read_csv(file_path, header=None, names=['names'])['names']
    names_np = names_col
    return names_np
