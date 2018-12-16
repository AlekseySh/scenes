import csv

from train import get_parser
from train import main as train_main


def main(args):
    split_ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    work_dir_root = args.work_dir
    result_csv = work_dir_root / 'result.csv'

    for split_id in split_ids:
        args.train_table = args.tables_dir / f'Training_{split_id}.csv'
        args.test_table = args.tables_dir / f'Testing_{split_id}.csv'
        args.work_dir = work_dir_root / f'split_{split_id}'

        max_metric = train_main(args)

        with open(result_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([split_id, max_metric])


if __name__ == '__main__':
    parser = get_parser()
    main(args=parser.parse_args())
