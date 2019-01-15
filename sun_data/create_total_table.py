import argparse
from pathlib import Path

from sun_data.common import tables_from_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', dest='data_dir', type=Path)
    args = parser.parse_args()

    df_train, df_test = tables_from_dir(args.data_dir)

    save_dir = Path(__file__).parent / 'files'
    df_train.to_csv(save_dir / 'train.csv', index=False)
    df_test.to_csv(save_dir / 'test.csv', index=False)


if __name__ == '__main__':
    main()
