import logging
from pathlib import Path

import PIL
import pandas as pd
import torchvision.transforms as t
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SceneDataset(Dataset):

    def __init__(self, data_path: Path, csv_path: Path):
        super(Dataset, self).__init__()

        self.data_path = data_path
        self.csv_path = csv_path
        self.transforms = None

        self.df = pd.read_csv(self.csv_path)

    def __getitem__(self, idx):
        path = self.data_path / self.df['path'][idx]
        pil_image = PIL.Image.open(path).convert('RGB')
        im = self.transforms(pil_image)
        label = self.df['class_enum'][idx]
        data = {'image': im,
                'label': label
                }
        return data

    def __len__(self):
        return len(self.df)

    def get_num_classes(self):
        return len(set(self.df['class_enum']))

    def set_default_transforms(self):
        self.transforms = get_default_transforms()

    def set_train_transforms(self):
        self.transforms = get_train_transforms()

    def set_test_transforms(self, n_augs: int):
        self.transforms = get_test_transforms(n_augs=n_augs)


std = (0.229, 0.224, 0.225)
mean = (0.485, 0.456, 0.406)
size = (128, 128)


def get_default_transforms():
    transforms = t.Compose([t.Resize(size=size),
                            t.ToTensor(),
                            t.Normalize(mean=mean, std=std)]
                           )
    return transforms


def get_train_transforms():
    transforms = t.Compose([t.Resize(size=size),
                            get_random_transforms(),
                            t.ToTensor(),
                            t.Normalize(mean=mean, std=std)]
                           )
    return transforms


def get_test_transforms(n_augs):
    # Test Time Augmentation (TTA) aproach
    rand_transforms = get_random_transforms()
    default_transforms = t.Compose([t.ToTensor(), t.Normalize(mean=mean, std=std)])
    augs = t.Compose([
        t.Resize(size=size),
        t.Lambda(lambda image: [rand_transforms(image) for _ in range(n_augs)]),
        t.Lambda(lambda images: [default_transforms(image) for image in images])
    ])
    return augs


def f_identity(image): return image


def get_random_transforms():
    crop_k = 0.9
    degree = 10
    color_k = 0.3
    apply_prob = 0.5

    crop_sz = (int(crop_k * size[0]), int(crop_k * size[1]))

    aug_list = [
        t.functional.hflip,
        t.Compose([t.RandomCrop(size=crop_sz), t.Resize(size=size)]),
        t.RandomRotation(degrees=(-degree, degree)),
        t.ColorJitter(brightness=color_k, contrast=color_k, saturation=color_k)
    ]
    rand_transforms = t.RandomOrder([t.RandomApply([aug], p=apply_prob) for aug in aug_list])
    return rand_transforms
