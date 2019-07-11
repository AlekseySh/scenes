from torch.utils.data import DataLoader
from tqdm import tqdm

from hierarchical.hier_dataset import HierDataset
from hierarchical.hier_network import Classifier


class Trainer:
    _classifier: Classifier
    _train_set: HierDataset
    _test_set: HierDataset

    def __init__(self,
                 classifier: Classifier,
                 train_set: HierDataset,
                 test_set: HierDataset
                 ):
        self._classifier = classifier
        self._train_set = train_set
        self._test_set = test_set

    def train(self) -> None:
        loader = DataLoader(dataset=self._train_set, batch_size=16,
                            shuffle=False, num_workers=4)

        for batch in tqdm(loader):
            logits_list = self._classifier(batch)

            [print(l.shape) for l in logits_list]
            break
