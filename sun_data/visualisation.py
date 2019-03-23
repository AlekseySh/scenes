import colorsys
import random
from typing import List, Tuple

import cv2
import numpy as np


def mask_center(bin_mask: np.ndarray) -> (int, int):
    assert np.sum(bin_mask) > 0

    moments = cv2.moments(bin_mask.astype(np.uint8))
    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return x, y


def random_colors(n: int,
                  bright: bool = True
                  ) -> List[Tuple[float, float, float]]:
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image: np.ndarray,
                      masks: np.ndarray,
                      labels: List[str]
                      ) -> np.ndarray:
    num_masks = masks.shape[2]
    im_masked = image.astype(np.uint8).copy()

    colors = random_colors(num_masks)
    for i in range(num_masks):
        xc, yc = mask_center(masks[:, :, i])
        im_masked = apply_mask(im_masked,
                               masks[:, :, i],
                               colors[i]
                               )
        im_masked = cv2.putText(
            img=im_masked.astype(np.uint8),
            org=(xc, yc),
            text=labels[i],
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            fontScale=1,
            thickness=1,
            color=(0, 0, 0),
        )
    return im_masked


def apply_mask(image: np.ndarray,
               mask: np.ndarray,
               color: List,
               alpha: float = 0.5
               ) -> np.ndarray:
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            (1 - alpha) * image[:, :, c] + alpha * color[c] * 255,
            image[:, :, c],
        )
    return image
