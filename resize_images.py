import argparse
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import image as mpimg
from scipy.misc import imresize
from tqdm import tqdm


def resize_inplace(im_dir: Path, height: int, width: int) -> None:
    im_paths = list(im_dir.glob('**/*.jpg'))
    for im_path in tqdm(im_paths):
        image = mpimg.imread(im_path)
        image_resized = imresize(arr=image, size=(height, width))
        mpimg.imsave(im_path, image_resized)


def get_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(description='Inplace image resizing.')
    parser.add_argument('--im_dir', dest='im_dir', type=Path)
    parser.add_argument('--height', dest='height', type=int, default=256)
    parser.add_argument('--width', dest='width', type=int, default=256)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    resize_inplace(im_dir=args.im_dir, height=args.height, width=args.width)
