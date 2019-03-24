import logging
from pathlib import Path

import PIL
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as t
from torch.utils.data import Dataset

from common import put_text_to_image
from sun_data.utils import beutify_name

STD = (0.229, 0.224, 0.225)
MEAN = (0.485, 0.456, 0.406)
SIZE = (256, 256)

logger = logging.getLogger(__name__)
logger.info(f'Used image size: {SIZE}')


class ImagesDataset(Dataset):

    def __init__(self, data_fold: Path, csv_path: Path):
        super(Dataset, self).__init__()

        self.data_fold = data_fold
        self.csv_path = csv_path

        self.transforms = None

        self.df = pd.read_csv(self.csv_path)

    def __getitem__(self, idx):
        pil_image = self.__read_pil_image(idx)
        label = self.df['class_enum'][idx]
        data = {
            'image': self.transforms(pil_image),
            'label': label,
        }
        return data

    def __len__(self):
        return len(self.df)

    def get_signed_image(self, idx, color, text=None):
        if text is None:
            text = [beutify_name(self.df['class_name'][idx])]

        image = np.array(self.__read_pil_image(idx))
        image = put_text_to_image(image, strings=text, color=color)
        image = t.ToTensor()(image)
        image_tensor = torch.tensor(255 * image, dtype=torch.uint8)
        return image_tensor

    def draw_class_samples(self, n_samples, label, color):
        layout = torch.zeros([n_samples, 3, SIZE[0], SIZE[1]], dtype=torch.uint8)
        ii_class = np.squeeze(np.nonzero(self.df['class_enum'] == label))
        ii_sampels = np.random.choice(ii_class, size=n_samples)
        for i, ind in enumerate(ii_sampels):
            layout[i, :, :, :] = self.get_signed_image(ind, color)
        return layout

    def __read_pil_image(self, idx):
        path = self.data_fold / self.df['path'][idx]
        pil_image = PIL.Image.open(path).convert('RGB')
        return pil_image

    def get_num_classes(self):
        return len(set(self.df['class_enum']))

    def set_default_transforms(self):
        self.transforms = get_default_transforms()

    def set_train_transforms(self):
        self.transforms = get_train_transforms()

    def set_test_transforms(self, n_augs: int):
        self.transforms = get_test_transforms(n_augs=n_augs)


def get_default_transforms():
    transforms = t.Compose([t.Resize(size=SIZE),
                            t.ToTensor(),
                            t.Normalize(mean=MEAN, std=STD)]
                           )
    return transforms


def get_train_transforms():
    transforms = t.Compose([t.Resize(size=SIZE),
                            get_random_transforms(),
                            t.ToTensor(),
                            t.Normalize(mean=MEAN, std=STD)]
                           )
    return transforms


def get_test_transforms(n_augs):
    # Test Time Augmentation (TTA) aproach
    rand_transforms = get_random_transforms()
    default_transforms = t.Compose([t.ToTensor(), t.Normalize(mean=MEAN, std=STD)])
    augs = t.Compose([
        t.Resize(size=SIZE),
        t.Lambda(lambda image: [rand_transforms(image) for _ in range(n_augs)]),
        t.Lambda(lambda images: [default_transforms(image) for image in images])
    ])
    return augs


def get_random_transforms():
    crop_k = 0.9
    degree = 10
    color_k = 0.3
    apply_prob = 0.5

    crop_sz = (int(crop_k * SIZE[0]), int(crop_k * SIZE[1]))

    aug_list = [
        t.functional.hflip,
        t.Compose([t.RandomCrop(size=crop_sz), t.Resize(size=SIZE)]),
        t.RandomRotation(degrees=(-degree, degree)),
        t.ColorJitter(brightness=color_k, contrast=color_k, saturation=color_k)
    ]
    rand_transforms = t.RandomOrder([t.RandomApply([aug], p=apply_prob) for aug in aug_list])
    return rand_transforms
