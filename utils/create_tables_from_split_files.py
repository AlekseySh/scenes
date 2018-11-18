import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def table_from_split_file(file_path, class_to_enum):
    paths = pd.read_csv(file_path, header=None, names=['paths'])['paths']
    paths = np.array(paths)

    enums, names = [], []
    for path in paths:
        class_name = str(Path(path).parent)
        names.append(class_name)
        enums.append(class_to_enum[class_name])

    names, enums = np.array(names), np.array(enums)
    df = pd.DataFrame(data={
        'path': paths,
        'class_enum': enums,
        'class_name': names
    })
    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--files_path', dest='files_path', type=Path)
    parser.add_argument('-m', '--mapping_path', dest='mapping_path', type=Path)
    args = parser.parse_args()

    with open(args.mapping_path) as f:
        mapping = json.load(f)

    for p in args.files_path.glob('*.txt'):
        table = table_from_split_file(file_path=p, class_to_enum=mapping)
        table.to_csv(p.parent / f'{p.stem}.csv', index=False)

        print(f'For file {p.name} table was created and saved.')
