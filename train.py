import argparse
from pathlib import Path

from datasets.table import SceneDataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=Path)
    args = parser.parse_args()

    total_set = SceneDataset(data_path=args.data_path, csv_name='total.csv')

    print(len(total_set))
