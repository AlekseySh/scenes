import argparse
from pathlib import Path

from data.common import resize_images_recursively

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_dir', dest='src_dir', type=Path)
    parser.add_argument('-d', '--dst_dir', dest='dst_dir', type=Path)
    parser.add_argument('-H', '--height', dest='height', type=int)
    parser.add_argument('-W', '--width', dest='width', type=int)
    args = parser.parse_args()

    resize_images_recursively(src_dir=args.src_dir,
                              dst_dir=args.dst_dir,
                              height=args.height,
                              width=args.width
                              )
