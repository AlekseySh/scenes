from pathlib import Path
from typing import List, Optional, Tuple

import torchvision.transforms as t
from torch import Tensor
from torch.utils.data import Dataset

from dataset import read_pil, get_train_transf, get_default_transf
from hierarchical.hier_structure import Hierarchy


class HierDataset(Dataset):
    _data_root: Path
    _im_paths: List[Path]
    _classes_bot: List[str]
    hierarchy: Hierarchy
    levels: Tuple[int, ...]

    _transforms: Optional[t.Compose]

    def __init__(self,
                 data_root: Path,
                 im_paths: List[Path],
                 classes_bot: List[str],
                 hierarchy: Hierarchy,
                 levels: Tuple[int, ...]
                 ):
        assert len(im_paths) == len(classes_bot)

        self._data_root = data_root
        self._im_paths = im_paths
        self._classes_bot = classes_bot
        self.levels = levels
        self.hierarchy = hierarchy

        self._transforms = None

    def __getitem__(self, idx: int) -> Tuple[Tuple[int, ...], Tuple[Tensor, ...]]:
        if self._transforms is None:
            raise ValueError('Set transforms before using.')

        pil_image = read_pil(self._data_root, self._im_paths[idx])
        im_tensor = self._transforms(pil_image)

        enums = self.hierarchy.get_enums(
            levels=self.levels, cls=self._classes_bot[idx])

        return im_tensor, enums

    def __len__(self) -> int:
        return len(self._im_paths)

    def set_train_transforms(self, aug_degree: float) -> None:
        self._transforms = get_train_transf(aug_degree)

    def set_test_transforms(self) -> None:
        self._transforms = get_default_transf()
