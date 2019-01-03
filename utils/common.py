import cv2
import numpy as np

from typing import List


def args_to_text(args):
    text = ''
    for arg in vars(args):
        string = f'{arg}: {getattr(args, arg)} \n'
        text += string
    return text


def put_text_to_image(image: np.ndarray,
                      strings: List[str],
                      str_height: int = 25,
                      color: List[int] = (0, 0, 0),
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

    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, new_x):
        self.n += 1
        self.avg = (self.avg * (self.n - 1) + new_x) / self.n


class Stopper:

    def __init__(self, n_obs, delta):
        self.n_obs = n_obs
        self.delta = delta

        self.cur_val = None
        self.max_val = 0
        self.num_fails = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self._count_fails()
        self._update_max()

    def _count_fails(self):
        if self.cur_val - self.max_val <= self.delta:
            self.num_fails += 1
        else:
            self.num_fails = 0

    def check_criterion(self):
        is_stop = self.num_fails == self.n_obs
        return is_stop

    def _update_max(self):
        if self.max_val < self.cur_val:
            self.max_val = self.cur_val
