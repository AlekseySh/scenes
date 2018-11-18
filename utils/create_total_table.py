from pathlib import Path
import argparse
import json

import pandas as pd
import numpy as np


def create_table(data_path, class_to_enum):
    names, paths, enums = [], [], []
    for name in class_to_enum.keys():
        name = name[1:]  # remove first '/' in class name
        class_fold = data_path / name

        for path in list(class_fold.glob('*.jpg')):
            path = Path(name) / path.name
            enum = class_to_enum[name]

            names.append(name)
            paths.append(path)
            enums.append(enum)

    data_frame = pd.DataFrame(data={'path': np.array(paths),
                                    'class_name': np.array(names),
                                    'class_enum': np.array(enums)
                                    })
    return data_frame


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=Path)
    parser.add_argument('-m', '--mapping_path', dest='mapping_path', type=Path)
    args = parser.parse_args()

    with open(args.mapping_path, 'r') as f:
        mapping = json.load(f)

    df = create_table(args.data_path, class_to_enum=mapping)
    df.to_csv(args.data_path / 'total.csv', index=False)
