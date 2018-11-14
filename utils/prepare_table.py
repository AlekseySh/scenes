from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn import preprocessing


def get_class_names(data_path):
    file_path = data_path / 'ClassName.txt'
    names = pd.read_csv(file_path, header=None, names=['names'])
    names_np = np.array(names).squeeze()
    return names_np


def create_table(data_path, class_names_unique):
    class_names = []
    paths = []
    for name in class_names_unique:
        name = name[1:]  # remove first '/' in class name
        class_fold = data_path / name

        for path in list(class_fold.glob('*.jpg')):
            relative_path = Path(name) / path.name
            class_names.append(name)
            paths.append(relative_path)

    label_encoder = preprocessing.LabelEncoder()
    class_enums = label_encoder.fit_transform(class_names)

    data_frame = pd.DataFrame({'path': paths,
                               'class_name': class_names,
                               'class_enum': class_enums
                               })
    return data_frame


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=Path)
    args = parser.parse_args()

    class_names_unq = get_class_names(args.data_path)

    df = create_table(args.data_path, class_names_unq)
    df.to_csv(args.data_path / 'total.csv', index=False)
