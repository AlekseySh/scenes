import argparse
import json
from pathlib import Path

from data.common import table_from_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=Path)
    parser.add_argument('-m', '--mapping_path', dest='mapping_path', type=Path)
    args = parser.parse_args()

    with open(args.mapping_path, 'r') as f:
        mapping = json.load(f)

    df = table_from_directory(args.data_path, class_to_enum=mapping)
    df.to_csv(args.data_path / 'total.csv', index=False)
