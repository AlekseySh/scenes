from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from scipy.misc import imresize
from tqdm import tqdm

from data.getters import get_name_to_enum


def table_from_split_file(file_path: Path):
    class_to_enum = get_name_to_enum()
    paths_raw = pd.read_csv(file_path, header=None, names=['paths'])['paths']
    paths_raw = np.array(paths_raw)

    enums, names, paths = [], [], []
    for path in paths_raw:
        class_name = str(Path(path).parent)
        paths.append(path[1:])
        names.append(class_name)
        enums.append(class_to_enum[class_name])

    names, enums, paths = np.array(names), np.array(enums), np.array(paths)
    df = pd.DataFrame(data={
        'path': paths,
        'class_enum': enums,
        'class_name': names
    })
    return df


def table_from_directory(data_path: Path):
    class_to_enum = get_name_to_enum()
    names, paths, enums = [], [], []
    for name in class_to_enum.keys():
        name = name[1:]  # remove first '/' in class name
        class_fold = data_path / name

        for path in list(class_fold.glob('*.jpg')):
            path = Path(name) / path.name
            enum = class_to_enum[name]

            names.append(name)
            paths.append(path)
            enums.append(enum)

    df = pd.DataFrame(data={
        'path': np.array(paths),
        'class_name': np.array(names),
        'class_enum': np.array(enums)
    })
    return df


def resize_and_save(im_path_src: Path,
                    im_path_dst: Path,
                    height: int,
                    width: int
                    ):
    image = mpimg.imread(im_path_src)
    image_resized = imresize(arr=image, size=(height, width))
    mpimg.imsave(im_path_dst, image_resized)


def resize_images_recursively(src_dir: Path,
                              dst_dir: Path,
                              height: int,
                              width: int
                              ):
    im_paths = src_dir.glob('**/*.jpg')
    for im_path in tqdm(list(im_paths)):
        resize_and_save(im_path_src=im_path,
                        im_path_dst=dst_dir / im_path.name,
                        height=height,
                        width=width
                        )
