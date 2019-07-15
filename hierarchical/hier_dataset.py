from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchvision.transforms as t
from torch import Tensor
from torch.utils.data import Dataset

from dataset import read_pil, get_train_transf, get_default_transf
from sun_data.utils import get_sun_names


class Hierarchy:
    n_levels: int

    _hier_sizes: Dict[int, int]
    _hier_mappings: Dict[int, Dict[str, Tensor]]

    def __init__(self, hier_mapping_files: List[Path]):
        self._hier_sizes = {}
        self._hier_mappings = {}

        for i_level, file in enumerate(hier_mapping_files):
            self._parse_mapping(file, i_level)

        self._set_bottom_level()

        self.n_levels = len(hier_mapping_files) + 1

    def get_one_hot(self, i_level: int, class_name: str) -> Tensor:
        return self._hier_mappings[i_level][class_name]

    def get_one_hot_arr(self, levels: Tuple[int, ...], class_name: str
                        ) -> Tuple[torch.Tensor, ...]:
        one_hot_arr = tuple([self.get_one_hot(i, class_name) for i in levels])
        return one_hot_arr

    def get_level_size(self, i_level: int) -> int:
        return self._hier_sizes[i_level]

    def get_level_sizes(self, levels: Tuple[int, ...]) -> Tuple[int, ...]:
        sizes = [self.get_level_size(l) for l in levels]
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
        class_to_onehot = {cls: torch.tensor(row) for cls, row in
                           zip(classes, df.values)}

        self._hier_mappings[i_level] = class_to_onehot
        self._hier_sizes[i_level] = len(df.columns)

    @staticmethod
    def code_one_hot(n_classes: int, i_hot: int) -> Tensor:
        one_hot = torch.zeros(n_classes)
        one_hot[i_hot] = True
        return one_hot


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
                 hier_mappings: List[Path],
                 levels: Tuple[int, ...]
                 ):
        assert len(im_paths) == len(classes_bot)

        self._data_root = data_root
        self._im_paths = im_paths
        self._classes_bot = classes_bot
        self.levels = levels
        self.hierarchy = Hierarchy(hier_mappings)

        self._transforms = None

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        if self._transforms is None:
            raise ValueError('Set transforms before using.')

        pil_image = read_pil(self._data_root, self._im_paths[idx])
        im_tensor = self._transforms(pil_image)

        one_hot = self.hierarchy.get_one_hot_arr(
            levels=self.levels, class_name=self._classes_bot[idx])

        return im_tensor, one_hot

    def __len__(self) -> int:
        return len(self._im_paths)

    def set_train_transforms(self, aug_degree: float = 1.0) -> None:
        self._transforms = get_train_transf(aug_degree)

    def set_test_transforms(self) -> None:
        self._transforms = get_default_transf()
