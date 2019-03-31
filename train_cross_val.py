import csv
from argparse import Namespace

from train import get_parser
from train import main as train_main


def main(args: Namespace) -> None:
    split_ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    work_dir_root = args.work_root
    result_csv = work_dir_root / 'result.csv'

    for split_id in split_ids:
        args.split_name = f'classic_{split_id}'
        args.work_root = work_dir_root / f'split_{split_id}'

        max_metric = train_main(args)

        with open(result_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([split_id, max_metric])


if __name__ == '__main__':
    parser = get_parser()
    main(args=parser.parse_args())
