import csv
from argparse import Namespace

from sun_data.utils import DataMode
from train import get_parser
from train import main as train_main


def main(args: Namespace) -> None:
    modes = [DataMode.CLASSIC_01, DataMode.CLASSIC_02,
             DataMode.CLASSIC_03, DataMode.CLASSIC_04,
             DataMode.CLASSIC_05, DataMode.CLASSIC_06,
             DataMode.CLASSIC_07, DataMode.CLASSIC_08,
             DataMode.CLASSIC_09, DataMode.CLASSIC_10]

    work_dir_root = args.work_root
    result_csv = work_dir_root / 'result.csv'

    for mode in modes:
        args.data_mode = mode
        args.work_root = work_dir_root / f'split_{mode}'

        max_metric = train_main(args)

        with open(result_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([mode, max_metric])


if __name__ == '__main__':
    parser = get_parser()
    main(args=parser.parse_args())
