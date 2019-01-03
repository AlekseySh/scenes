import json
from pathlib import Path

from bidict import bidict


def get_name_enum_mapping():
    jpath = Path(__file__).parent.parent / 'data' / 'SunNameToEnum.json'
    with open(jpath) as j:
        name_to_enum = bidict(json.load(j))
    return name_to_enum
