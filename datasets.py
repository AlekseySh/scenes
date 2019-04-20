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
logger.info(f'Using image size: {SIZE}')


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
            im.close()
        return pil_image

    def set_default_transforms(self) -> None:
        self._transforms = get_default_transforms()

    def set_train_transforms(self) -> None:
        self._transforms = get_train_transforms()

    def set_test_transforms(self, n_augs: int) -> None:
        self._transforms = get_test_transforms(n_tta=n_augs)

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
    transforms = t.Compose([
        t.Resize(size=SIZE),
        t.Lambda(lambda image: [rand_transforms(image) for _ in range(n_tta)]),
        t.Lambda(lambda images: [default_transforms(image) for image in images])
    ])
    return transforms


def get_random_transforms(k: int = 1) -> t.RandomOrder:
    crop_k = k * 0.9
    degree = k * 10
    color_k = k * 0.3
    apply_prob = k * 0.5

    crop_sz = (int(crop_k * SIZE[0]), int(crop_k * SIZE[1]))

    aug_list = [
        t.functional.hflip,
        t.Compose([t.RandomCrop(size=crop_sz), t.Resize(size=SIZE)]),
        t.RandomRotation(degrees=(-degree, degree)),
        t.ColorJitter(brightness=color_k, contrast=color_k, saturation=color_k)
    ]
    rand_transforms = t.RandomOrder([t.RandomApply([aug], p=apply_prob) for aug in aug_list])
    return rand_transforms
