import random
from argparse import Namespace
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor


def fix_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def beutify_args(args: Namespace) -> str:
    text = ''
    for arg in vars(args):
        string = f'{arg}: {getattr(args, arg)} \n'
        text += string
    return text


def put_text_to_image(image: np.ndarray,
                      strings: List[str],
                      str_height: int = 25,
                      color: Tuple[int, int, int] = (0, 0, 0),
                      ) -> np.ndarray:
    image = image.astype(np.uint8)

    for i, string in enumerate(strings):
        image = cv2.putText(img=image,
                            org=(5, (i + 1) * str_height),
                            text=string,
                            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1,
                            thickness=2,
                            color=color
                            )
    return image


class OnlineAvg:
    avg: Union[np.ndarray, Tensor, float]
    n: int

    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, new_x: Union[np.ndarray, Tensor, float]) -> None:
        self.n += 1
        self.avg = (self.avg * (self.n - 1) + new_x) / self.n


class Stopper:
    _n_obs: int
    _delta: float
    _cur_val: float
    _max_val: float
    _num_fails: int

    def __init__(self, n_obs: int, delta: float):
        self._n_obs = n_obs
        self._delta = delta

        self._cur_val = 0
        self._max_val = 0
        self._num_fails = 0

    def update(self, cur_val: float) -> None:
        self._cur_val = cur_val
        self._count_fails()
        self._update_max()

    def _count_fails(self) -> None:
        if self._cur_val - self._max_val <= self._delta:
            self._num_fails += 1
        else:
            self._num_fails = 0

    def check_criterion(self) -> bool:
        is_stop = self._num_fails == self._n_obs
        return is_stop

    def _update_max(self) -> None:
        if self._max_val < self._cur_val:
            self._max_val = self._cur_val
