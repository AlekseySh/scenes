from typing import List

import cv2
import numpy as np


def beutify_args(args):
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
