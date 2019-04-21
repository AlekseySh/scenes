import random
from argparse import Namespace
from typing import List, Tuple, Union

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL.Image import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from seaborn import countplot
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
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


# VISUALISATION

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


def canvas_to_img(canvas: FigureCanvas) -> PIL.Image:
    canvas.draw()
    string, (width, height) = canvas.print_to_buffer()
    img = np.fromstring(string, np.uint8).reshape((height, width, 4))
    pil_image = PIL.Image.fromarray(img, mode='RGBA')
    return pil_image


def confusion_matrix_as_img(gts: np.ndarray,
                            preds: np.ndarray,
                            classes: List[str]
                            ) -> Image:
    font = 20
    conf_mat = confusion_matrix(y_true=gts, y_pred=preds)
    classes = np.array(classes)[unique_labels(gts, preds)]

    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    fig = Figure(figsize=(16, 16))
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set(xticks=np.arange(conf_mat.shape[1]),
           yticks=np.arange(conf_mat.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='Ground Truth', xlabel='Predicted')

    ax.xaxis.label.set_fontsize(font)
    ax.yaxis.label.set_fontsize(font)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=font)
    plt.setp(ax.get_yticklabels(), fontsize=font)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i,
                    format(conf_mat[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if conf_mat[i, j] > conf_mat.max() / 2. else "black",
                    size=font
                    )

    fig.tight_layout()
    pil_image = canvas_to_img(canvas)
    return pil_image


def histogram_as_img(categories: List[str]) -> Image:
    font = 20

    fig = Figure(figsize=(16, 16))
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    df = pd.DataFrame({'category': categories})
    countplot(ax=ax, x='category', color='b',
              data=df, order=df['category'].value_counts().index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=font)

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(1.2 * font)

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font)

    fig.tight_layout()
    pil_image = canvas_to_img(canvas)
    return pil_image
