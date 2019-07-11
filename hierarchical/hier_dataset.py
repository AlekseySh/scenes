from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torchvision.transforms as t
from torch import Tensor
from torch.utils.data import Dataset

from dataset import read_pil, get_train_transf, get_default_transf
from sun_data.utils import get_sun_names


class Hierarchy:
    n_levels: int

    _hier_sizes: Dict[int, int]
    _hier_mappings: Dict[int, Dict[str, np.ndarray]]

    def __init__(self, hier_mapping_files: List[Path]):
        self.n_levels = len(hier_mapping_files) + 1

        self._hier_sizes = {}
        self._hier_mappings = {}

        for i_level, file in enumerate(hier_mapping_files):
            self._parse_mapping(file, i_level)

        self._set_bottom_level()

    def get_one_hot(self, i_level: int, class_name: str) -> np.ndarray:
        return self._hier_mappings[i_level][class_name]

    def get_level_size(self, i_level: int) -> int:
        return self._hier_sizes[i_level]

    def get_level_sizes(self) -> Tuple[int, ...]:
        sizes = [self.get_level_size(i) for i in range(self.n_levels)]
        return tuple(sizes)

    def _set_bottom_level(self) -> None:
        classes_to_one_hot = {}
        bot_classes = get_sun_names()
        n_classes = len(bot_classes)
        for i, cls in enumerate(sorted(bot_classes)):  # todo
            classes_to_one_hot[cls] = Hierarchy.code_one_hot(n_classes, i)

        i_level_bot = len(self._hier_mappings)
        self._hier_mappings[i_level_bot] = classes_to_one_hot
        self._hier_sizes[i_level_bot] = n_classes

    def _parse_mapping(self, file_path: Path, i_level: int) -> None:
        df = pd.read_csv(file_path, index_col=None)
        classes = list(df['category'].values)  # bot classes

        df = df.drop(columns=['category'])
        class_to_onehot = {cls: row for cls, row in
                           zip(classes, df.values.astype(np.bool))}

        self._hier_mappings[i_level] = class_to_onehot
        self._hier_sizes[i_level] = len(df.columns)

    @staticmethod
    def code_one_hot(n_classes: int, i_hot: int) -> np.ndarray:
        one_hot = np.zeros(n_classes, dtype=np.bool)
        one_hot[i_hot] = True
        return one_hot


class HierDataset(Dataset):
    _data_root: Path
    _im_paths: List[Path]
    _classes_bot: List[str]
    hierarchy: Hierarchy
    _transforms: Optional[t.Compose]

    def __init__(self,
                 data_root: Path,
                 im_paths: List[Path],
                 classes_bot: List[str],
                 hierarchy_mappings: List[Path]
                 ):
        assert len(im_paths) == len(classes_bot)

        self._data_root = data_root
        self._im_paths = im_paths
        self._classes_bot = classes_bot
        self.hierarchy = Hierarchy(hierarchy_mappings)

        self._transforms = None

    def __getitem__(self, idx: int) -> Tensor:
        if self._transforms is None:
            raise ValueError('Set transforms before using.')

        pil_image = read_pil(self._data_root, self._im_paths[idx])
        im_tensor = self._transforms(pil_image)
        return im_tensor

    def __len__(self) -> int:
        return len(self._im_paths)

    def set_train_transforms(self, aug_degree: float = 1.0) -> None:
        self._transforms = get_train_transf(aug_degree)

    def set_test_transforms(self) -> None:
        self._transforms = get_default_transf()
