from typing import Dict

import numpy as np


class Calculator:
    _gts: np.ndarray
    _preds: np.ndarray
    _confidences: np.ndarray
    samples_num: int

    def __init__(self, gts: np.ndarray, preds: np.ndarray, probs: np.ndarray):
        assert gts.shape == preds.shape, f'{gts.shape}, {preds.shape}'
        assert gts.shape == probs.shape, f'{gts.shape}, {probs.shape}'

        self._gts = gts
        self._preds = preds
        self._confidences = probs
        self.samples_num = len(self._gts)

    def calc(self) -> Dict[str, float]:
        metrics = {
            'accuracy': round(calc_accuracy(self._gts, self._preds), 4),
            'accuracy_weighted': round(calc_accuracy_weighted(self._gts, self._preds), 4)
        }
        return metrics

    def worst_errors(self, n_worst: int) -> np.ndarray:
        ii_mistakes = np.nonzero(self._preds != self._gts)[0]
        probs = self._confidences[ii_mistakes]
        ii_worst = ii_mistakes[np.argsort(probs)][-n_worst:]
        return ii_worst

    def best_preds(self, n_best: int) -> np.ndarray:
        ii_correct = np.nonzero(self._preds == self._gts)[0]
        probs = self._confidences[ii_correct]
        ii_best = ii_correct[np.argsort(probs)][-n_best:]
        return ii_best


def calc_accuracy(gts: np.ndarray, preds: np.ndarray) -> float:
    assert gts.shape == preds.shape
    acc = np.sum(gts == preds) / len(gts)
    return float(acc)


def calc_accuracy_weighted(gts: np.ndarray, preds: np.ndarray) -> float:
    assert gts.shape == preds.shape
    acc_list = []
    for label in np.unique(gts):
        w_label = gts == label
        acc = calc_accuracy(gts=gts[w_label], preds=preds[w_label])
        acc_list.append(acc)
    return np.mean(acc_list)
