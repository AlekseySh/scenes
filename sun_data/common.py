from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sun_data.utils import get_mapping


def table_from_split_file(file_path: Path):
    name_to_enum = get_mapping('NameToEnum')

    paths_raw = pd.read_csv(file_path, header=None, names=['paths'])['paths']
    paths_raw = np.array(paths_raw)

    enums, names, paths = [], [], []
    for path in paths_raw:
        class_name = str(Path(path).parent)
        paths.append(path[1:])  # del '/' from name
        names.append(class_name)
        enums.append(name_to_enum[class_name])

    names, enums, paths = np.array(names), np.array(enums), np.array(paths)
    df = pd.DataFrame(data={
        'path': paths,
        'class_enum': enums,
        'class_name': names
    })
    return df


def domains_tables_from_dir(data_path: Path):
    name_to_domain = get_mapping('NameToDomain')
    domain_to_enum = get_mapping('DomainToEnum')

    class_names, paths, enums = [], [], []
    for name in name_to_domain.keys():

        domain = name_to_domain[name]
        enum = domain_to_enum[domain]

        class_fold = data_path / name[1:]  # del '/' from name

        for path in list(class_fold.glob('*.jpg')):
            path = Path(name[1:]) / path.name

            class_names.append(domain)
            paths.append(path)
            enums.append(enum)

    df = pd.DataFrame(data={
        'path': np.array(paths),
        'class_name': np.array(class_names),
        'class_enum': np.array(enums)
    })
    df_train, df_test = train_test_split(df, test_size=0.2)

    save_dir = Path(__file__).parent / 'split_domains' / 'files'
    df_train.to_csv(save_dir / 'train.csv', index=False)
    df_test.to_csv(save_dir / 'test.csv', index=False)


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


if __name__ == '__main__':
    domains_tables_from_dir(Path('/data/sun/images/256_size'))
