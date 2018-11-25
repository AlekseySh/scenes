import numpy as np


class MetricsCalculator:

    def __init__(self, gt, pred, score):
        assert gt.shape == pred.shape
        assert gt.shape == score.shape

        self.gt = gt
        self.pred = pred
        self.score = score
        self.num = len(self.gt)

    def calc(self):
        acc = np.sum(self.gt == self.pred) / self.num
        metrics = {
            'accuracy': acc
        }
        return metrics
