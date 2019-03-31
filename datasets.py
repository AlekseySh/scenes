import logging
from pathlib import Path
from typing import Tuple, List

import PIL
import numpy as np
import torch
import torchvision.transforms as t
from PIL.Image import Image as TImage
from torch import Tensor
from torch.utils.data import Dataset

from common import put_text_to_image

STD = (0.229, 0.224, 0.225)
MEAN = (0.485, 0.456, 0.406)
SIZE = (256, 256)

logger = logging.getLogger(__name__)
logger.info(f'Using image size: {SIZE}')


class ImagesDataset(Dataset):
    _data_root: Path
    _im_paths: List[Path]
    _labels: List[int]
    _transforms: t.Compose

    def __init__(self,
                 data_root: Path,
                 im_paths: List[Path],
                 labels: List[int]
                 ):
        assert len(im_paths) == len(labels)

        super().__init__()
        self._data_root = data_root
        self._im_paths = im_paths
        self._labels = labels
        self._transforms = None

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        assert self._transforms is not None

        im_path, label = self._im_paths[idx], self._labels[idx]

        if str(im_path).startswith('/'):
            im_path = Path(str(im_path)[1:])

        im_tensor = self._transforms(_read_pil(self._data_root / im_path))
        return im_tensor, label

    def __len__(self) -> int:
        return len(self._im_paths)

    def set_default_transforms(self) -> None:
        self._transforms = get_default_transforms()

    def set_train_transforms(self) -> None:
        self._transforms = get_train_transforms()

    def set_test_transforms(self, n_augs: int) -> None:
        self._transforms = get_test_transforms(n_tta=n_augs)

    # VISUALISATION

    def get_signed_image(self,
                         idx: int,
                         color: Tuple[int, int, int],
                         text: List[str]
                         ) -> Tensor:
        image = np.array(_read_pil(self._im_paths[idx]))
        image = put_text_to_image(image=image, strings=text, color=color)
        image = t.ToTensor()(image)
        image_tensor = torch.tensor(255 * image, dtype=torch.uint8)
        return image_tensor

    def draw_class_samples(self,
                           n_samples: int,
                           class_label: int,
                           text: List[str],
                           color: Tuple[int, int, int]
                           ) -> Tensor:
        layout = torch.zeros([n_samples, 3, SIZE[0], SIZE[1]], dtype=torch.uint8)
        ii_class = np.squeeze(np.nonzero(np.array(self._labels) == class_label))
        ii_sampels = np.random.choice(ii_class, size=n_samples)
        for i, ind in enumerate(ii_sampels):
            layout[i, :, :, :] = self.get_signed_image(idx=ind, color=color, text=text)
        return layout


def get_default_transforms() -> t.Compose:
    transforms = t.Compose([t.Resize(size=SIZE),
                            t.ToTensor(),
                            t.Normalize(mean=MEAN, std=STD)]
                           )
    return transforms


def get_train_transforms() -> t.Compose:
    transforms = t.Compose([t.Resize(size=SIZE),
                            get_random_transforms(),
                            t.ToTensor(),
                            t.Normalize(mean=MEAN, std=STD)]
                           )
    return transforms


def get_test_transforms(n_tta: int) -> t.Compose:
    # Test Time Augmentation (TTA) aproach
    rand_transforms = get_random_transforms()
    default_transforms = t.Compose([t.ToTensor(), t.Normalize(mean=MEAN, std=STD)])
    augs = t.Compose([
        t.Resize(size=SIZE),
        t.Lambda(lambda image: [rand_transforms(image) for _ in range(n_tta)]),
        t.Lambda(lambda images: [default_transforms(image) for image in images])
    ])
    return augs


def get_random_transforms() -> t.RandomOrder:
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


def _read_pil(im_path: Path) -> TImage:
    pil_image = PIL.Image.open(im_path).convert('RGB')
    return pil_image
