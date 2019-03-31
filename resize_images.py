import argparse
from pathlib import Path

from matplotlib import image as mpimg
from scipy.misc import imresize
from tqdm import tqdm


def resize_and_save(im_path_src: Path,
                    im_path_dst: Path,
                    height: int,
                    width: int
                    ) -> None:
    image = mpimg.imread(im_path_src)
    image_resized = imresize(arr=image, size=(height, width))
    mpimg.imsave(im_path_dst, image_resized)


def resize_images_recursively(src_dir: Path,
                              dst_dir: Path,
                              height: int,
                              width: int
                              ) -> None:
    im_paths = src_dir.glob('**/*.jpg')
    for im_path in tqdm(list(im_paths)):
        resize_and_save(im_path_src=im_path,
                        im_path_dst=dst_dir / im_path.name,
                        height=height,
                        width=width
                        )


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
