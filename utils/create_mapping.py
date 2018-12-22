import argparse
import json
from pathlib import Path

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_names_file', dest='path_to_names', type=Path)
    args = parser.parse_args()

    save_path = args.path_to_names.parent / 'class_to_enum.json'

    content = pd.read_csv(args.path_to_names, header=None, names=['names'])
    class_to_enum = {name: i for i, name in enumerate(content['names'])}

    with open(save_path, 'w') as f:
        json.dump(class_to_enum, f)
