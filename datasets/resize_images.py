import argparse
from pathlib import Path
from tqdm import tqdm

import matplotlib.image as mpimg
from scipy.misc import imresize


def resize_and_save(im_path: Path, height: int, width: int):
    image = mpimg.imread(im_path)
    imresize(arr=image, size=(height, width))
    mpimg.imsave(im_path, image)


def resize_images_recursively(image_dir: Path, height: int, width: int):
    im_paths = image_dir.glob('**/*.jpg')
    for path in tqdm(list(im_paths)):
        resize_and_save(im_path=path, height=height, width=width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--image_dir', dest='image_dir', type=Path)
    parser.add_argument('-H', '--height', dest='height', type=int)
    parser.add_argument('-W', '--width', dest='width', type=int)
    args = parser.parse_args()

    resize_images_recursively(image_dir=args.image_dir,
                              height=args.height,
                              width=args.width
                              )
