import numpy as np


class MetricsCalculator:

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

    def find_worst_mistakes(self, n_worst: int):
        ii_mistakes = np.nonzero(self.pred != self.gt)
        probs = self.score[ii_mistakes]
        ii_worst = np.argsort(probs)[-n_worst:]
        return ii_worst
