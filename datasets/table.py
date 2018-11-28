import PIL

from torch.utils.data import Dataset
import torchvision.transforms as t
import pandas as pd


class SceneDataset(Dataset):

    def __init__(self, data_path, csv_path, transforms=None):
        super(Dataset, self).__init__()

        self.data_path = data_path
        self.csv_path = csv_path

        if transforms is None:
            self._set_default_transforms()
        else:
            self.transforms = transforms

        self.df = pd.read_csv(self.csv_path)

    def __getitem__(self, idx):
        path = self.data_path / self.df['path'][idx]
        image = _load_img_to_tensor(path, self.transforms)
        label = self.df['class_enum'][idx]
        data = {'image': image,
                'label': label
                }
        return data

    def __len__(self):
        return len(self.df)

    def _set_default_transforms(self):
        std = (0.229, 0.224, 0.225)
        mean = (0.485, 0.456, 0.406)
        size = (128, 128)
        transforms = t.Compose([t.Resize(size=size),
                                t.ToTensor(),
                                t.Normalize(mean=mean, std=std)]
                               )
        self.transforms = transforms

    def get_num_classes(self):
        return len(set(self.df['class_enum']))


def _load_img_to_tensor(path, transforms):
    pil_image = PIL.Image.open(path).convert('RGB')
    tensor_image = transforms(pil_image)
    return tensor_image
