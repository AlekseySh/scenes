from pathlib import Path

import pandas as pd


def get_sun_tags():
    file_path = Path(__file__).parent.parent / 'data' / 'SunClassName.txt'
    names_col = pd.read_csv(file_path, header=None, names=['names'])['names']
    names_np = names_col
    return names_np


if __name__ == '__main__':
    names = get_sun_tags()
    print(names)
