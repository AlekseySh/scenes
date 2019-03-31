import numpy as np
from typing import Dict


class Calculator:
    _gts: np.ndarray
    _preds: np.ndarray
    _confidences: np.ndarray

    def __init__(self, gts: np.ndarray, preds: np.ndarray, confidences: np.ndarray):
        assert gts.shape == preds.shape
        assert gts.shape == confidences.shape

        self._gts = gts
        self._preds = preds
        self._confidences = confidences
        self.samples_num = len(self._gts)

    def calc(self) -> Dict[str, float]:
        acc = np.sum(self._gts == self._preds) / self.samples_num
        metrics = {'accuracy': acc}
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
