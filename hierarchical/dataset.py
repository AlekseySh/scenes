from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torchvision.transforms as t
from torch import Tensor
from torch.utils.data import Dataset

from dataset import read_pil, get_train_transf, get_default_transf
from sun_data.utils import get_sun_names


class Hierarchy:
    n_levels: int
    _hier_mappings: Dict[int, Dict[str, np.ndarray]]

    def __init__(self, hier_mapping_files: List[Path]):
        self._hier_mappings = {}
        for i_level, file in enumerate(hier_mapping_files):
            self._hier_mappings[i_level + 1] = Hierarchy.parse_mapping(file)

        # additional mapping for classes from bot level
        classes_to_one_hot = {}
        bot_classes = get_sun_names()
        n_bot = len(bot_classes)
        for i, cls in enumerate(sorted(bot_classes)):  # todo
            classes_to_one_hot[cls] = Hierarchy.code_one_hot(n_bot, i)

        i_level_bot = len(hier_mapping_files) + 1
        self._hier_mappings[i_level_bot] = classes_to_one_hot

    def get_one_hot(self, i_level: int, class_name: str) -> np.ndarray:
        return self._hier_mappings[i_level][class_name]

    @staticmethod
    def parse_mapping(file_path: Path) -> Dict[str, np.ndarray]:
        df = pd.read_csv(file_path, index_col=None)
        classes = list(df['category'].values)  # bot classes

        df = df.drop(columns=['category'])
        class_to_onehot = {cls: row for cls, row in
                           zip(classes, df.values.astype(np.bool))}

        return class_to_onehot

    @staticmethod
    def code_one_hot(n_classes: int, i_hot: int) -> np.ndarray:
        one_hot = np.zeros(n_classes, dtype=np.bool)
        one_hot[i_hot] = True
        return one_hot


class HierDataset(Dataset):
    _data_root: Path
    _im_paths: List[Path]
    _classes_bot: List[str]
    _hierarchy: Hierarchy
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
        self._hierarchy = Hierarchy(hierarchy_mappings)

        self._transforms = None

    def __getitem__(self, idx: int) -> Tensor:
        pil_image = read_pil(self._data_root, self._im_paths[idx])
        im_tensor = self._transforms(pil_image)
        return im_tensor

    def __len__(self) -> int:
        return len(self._im_paths)

    def set_train_transforms(self, aug_degree: float = 1.0) -> None:
        self._transforms = get_train_transf(aug_degree)

    def set_test_transforms(self) -> None:
        self._transforms = get_default_transf()
