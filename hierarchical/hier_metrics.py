from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Sigmoid


def calc_accuracy_batch(logits_list: Tuple[Tensor, ...],
                        one_hot_arr: Tuple[Tensor, ...],
                        th: float = 0.5
                        ) -> Tuple[float, ...]:
    acc_list = []
    for logits, one_hot in zip(logits_list, one_hot_arr):
        logits = logits.clone().cpu()
        one_hot = one_hot.clone().cpu()

        assert logits.shape == one_hot.shape

        preds = Sigmoid()(logits) > th
        equals_mat = preds.type(torch.bool) == one_hot.type(torch.bool)
        equals = equals_mat.type(torch.float32).prod(dim=1)

        assert equals.shape == logits.shape[0:1], f'{equals.shape}, {logits.shape}'

        acc = float(torch.mean(equals))
        acc_list.append(acc)

    return acc_list
