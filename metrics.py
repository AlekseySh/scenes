import numpy as np


class Calculator:

    def __init__(self, gt, pred, score):
        assert gt.shape == pred.shape
        assert gt.shape == score.shape

        self.gt = gt
        self.pred = pred
        self.score = score
        self.samples_num = len(self.gt)

    def calc(self):
        acc = np.sum(self.gt == self.pred) / self.samples_num
        metrics = {
            'accuracy': acc
        }
        return metrics

    def find_worst_mistakes(self, n_worst: int) -> np.ndarray:
        ii_mistakes = np.nonzero(self.pred != self.gt)[0]
        probs = self.score[ii_mistakes]
        ii_worst = ii_mistakes[np.argsort(probs)][-n_worst:]
        return ii_worst

    def find_best_predicts(self, n_best: int) -> np.ndarray:
        ii_correct = np.nonzero(self.pred == self.gt)[0]
        probs = self.score[ii_correct]
        ii_best = ii_correct[np.argsort(probs)][-n_best:]
        return ii_best
