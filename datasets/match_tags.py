from pathlib import Path

import pandas as pd


def get_sun_tags():
    file_path = Path(__file__).parent.parent / 'data' / 'sun_tags.txt'
    names = pd.read_csv(file_path, header=None, names=['names'])['names']
    return names


if __name__ == '__main__':
    print(get_sun_tags())
