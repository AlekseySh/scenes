import logging
import resource
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

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))

STD = (0.229, 0.224, 0.225)
MEAN = (0.485, 0.456, 0.406)
SIZE = (256, 256)  # 299 for inception, 224 others

logger = logging.getLogger(__name__)


class ImagesDataset(Dataset):
    _data_root: Path
    _im_paths: List[Path]
    _labels_enum: List[int]
    _transforms: t.Compose

    def __init__(self,
                 data_root: Path,
                 im_paths: List[Path],
                 labels_enum: List[int]
                 ):
        assert len(im_paths) == len(labels_enum)

        super().__init__()
        self._data_root = data_root
        self._im_paths = im_paths
        self._labels_enum = labels_enum
        self._transforms = None

        logger.info(f'Dataset created with size {len(self)}')
        logger.info(f'Using image size: {SIZE}')
        assert str(SIZE[0]) in str(data_root)  # todo remove it

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        assert self._transforms is not None

        im_tensor = self._transforms(self._read_pil(idx))
        label = self._labels_enum[idx]
        return im_tensor, label

    def __len__(self) -> int:
        return len(self._im_paths)

    def _read_pil(self, idx: int) -> TImage:
        local_path = self._im_paths[idx]

        if str(local_path).startswith('/'):
            local_path = Path(str(local_path)[1:])

        abs_path = self._data_root / local_path

        with PIL.Image.open(abs_path) as im:
            pil_image = im.convert('RGB')
            assert pil_image.size == SIZE, \
                'Assumed, that stored images already resized.'
        return pil_image

    def set_default_transforms(self) -> None:
        self._transforms = _get_default_transf()

    def set_train_transforms(self, aug_degree: float) -> None:
        self._transforms = _get_train_transf(aug_degree)

    def set_test_transforms(self, n_augs: int, aug_degree: float) -> None:
        self._transforms = _get_test_transf(n_augs, aug_degree)

    @property
    def labels_enum(self) -> List[int]:
        return self._labels_enum

    # VISUALISATION

    def get_signed_image(self,
                         idx: int,
                         color: Tuple[int, int, int],
                         text: List[str]
                         ) -> Tensor:
        image = np.array(self._read_pil(idx).resize(SIZE))
        image = put_text_to_image(image=image, strings=text, color=color)
        image_tensor = (255 * t.ToTensor()(image)).type(torch.uint8)
        return image_tensor

    def draw_class_samples(self,
                           n_samples: int,
                           class_num: int,
                           text: List[str],
                           color: Tuple[int, int, int]
                           ) -> Tensor:
        layout = torch.zeros([n_samples, 3, SIZE[0], SIZE[1]], dtype=torch.uint8)
        ii_class = np.nonzero(np.array(self._labels_enum) == class_num)[0]
        ii_sampels = np.random.choice(ii_class, size=n_samples)
        for i, ind in enumerate(ii_sampels):
            layout[i, :, :, :] = self.get_signed_image(idx=ind, color=color, text=text)
        return layout


def _get_default_transf() -> t.Compose:
    transforms = t.Compose([t.ToTensor(), t.Normalize(mean=MEAN, std=STD)])
    return transforms


def _get_train_transf(aug_degree: float) -> t.Compose:
    transforms = t.Compose([
        _get_rand_transf(aug_degree),
        t.Resize(size=SIZE),
        t.ToTensor(),
        t.Normalize(mean=MEAN, std=STD)]
    )
    return transforms


def _get_test_transf(n_tta: int, aug_degree: float) -> t.Compose:
    # Test Time Augmentation (TTA) aproach
    transforms = t.Compose([
        t.Lambda(lambda image: [_get_rand_transf(aug_degree)(image) for _ in range(n_tta)]),
        t.Lambda(lambda images: [t.Resize(size=SIZE)(image) for image in images]),
        t.Lambda(lambda images: [_get_default_transf()(image) for image in images])
    ])
    return transforms


def _get_rand_transf(k: float) -> t.RandomOrder:
    assert k > 0

    crop_k = 1 - k * 0.1
    crop_sz = (int(crop_k * SIZE[0]), int(crop_k * SIZE[1]))

    aug_list = [
        t.functional.hflip,
        t.RandomCrop(size=crop_sz),
        t.RandomAffine(degrees=k * 10,
                       translate=(0.1 * k, 0.1 * k),
                       scale=(1 - 0.1 * k, 1 + 0.1 * k),
                       shear=k * 5,
                       fillcolor=0
                       ),
        t.ColorJitter(brightness=0.1 * k,
                      contrast=0.1 * k,
                      saturation=0.1 * k,
                      hue=0.1 * k
                      )
    ]
    transforms = t.RandomOrder([t.RandomApply([aug], p=0.4 + 0.1 * k) for aug in aug_list])
    return transforms
