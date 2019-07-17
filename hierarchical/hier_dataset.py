from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as t
from torch import Tensor
from torch.utils.data import Dataset

from dataset import read_pil, get_train_transf, get_default_transf
from sun_data.utils import get_sun_names


# Multi label task
class HierarchyMultiLabel:
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


# Single label task

class Hierarchy:
    n_levels: int

    _levels_sizes: Dict[int, int]
    _mappings: Dict[int, Dict[str, int]]

    def __init__(self, mapping_files: Tuple[Path, ...]):
        self.n_levels = 0
        self._levels_sizes = {}
        self._mappings = {}

        for level, file in enumerate(mapping_files):
            self._parse_mapping_file(level=level, file=file)
            self.n_levels += 1

        self._set_bottom_level()
        self.n_levels += 1

    def _parse_mapping_file(self, level: int, file: Path) -> None:
        df = pd.read_csv(file, index_col=None)
        bot_classes = list(df['category'].values)

        df = df.drop(columns=['category'])

        class_to_enum = {}
        for cls, row in zip(bot_classes, df.values):
            class_to_enum[cls] = Hierarchy.one_hot_to_enum(row)

        self._mappings[level] = class_to_enum
        self._levels_sizes[level] = len(class_to_enum)

    def _set_bottom_level(self) -> None:
        cls_to_enum = {}
        bot_classes = get_sun_names()

        for i, cls in enumerate(sorted(bot_classes)):
            cls_to_enum[cls] = i

        bot_level = len(self._mappings)
        self._mappings[bot_level] = cls_to_enum
        self._levels_sizes[bot_level] = len(bot_classes)

    def get_enums(self, cls: str, levels: Tuple[int, ...]) -> Tuple[int, ...]:
        enums = [self._mappings[level][cls] for level in levels]
        return tuple(enums)

    def get_levels_sizes(self, levels: Tuple[int, ...]) -> Tuple[int, ...]:
        sizes = [len(self._mappings[level]) for level in levels]
        return tuple(sizes)

    @staticmethod
    def one_hot_to_enum(one_hot: np.ndarray) -> int:
        n_labels = sum(one_hot)
        if n_labels != 1:
            raise ValueError(f'Mapping file contains row with {n_labels} labels.')

        inds_nonzero = np.nonzero(one_hot)
        enum = int(inds_nonzero[0])
        return enum


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
