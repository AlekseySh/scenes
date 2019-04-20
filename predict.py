from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import PIL
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import put_text_to_image
from datasets import ImagesDataset
from network import Classifier
from sun_data.utils import get_name_to_enum, DataMode


def predict_arr(model: Classifier,
                im_paths: List[Path],
                im_dir: Path,
                batch_size: int
                ) -> List[int]:
    dataset = ImagesDataset(data_root=im_dir,
                            im_paths=im_paths,
                            labels_enum=[1] * len(im_paths)
                            )
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4
                        )
    labels: List[int] = []
    for _, im in tqdm(loader):
        label, _ = model.classify(im)
        labels.extend(label.detach().cpu().numpy().tolist())

    return labels


def sign_and_save(im_paths: List[Path], names: List[str], save_dir: Path) -> None:
    for im_path, name in zip(im_paths, names):
        pil_image = PIL.Image.open(im_path).convert('RGB')
        image_signed = put_text_to_image(image=np.array(pil_image),
                                         strings=[name]
                                         )
        image_signed = PIL.Image.fromarray(image_signed).convert('RGB')
        image_signed.save(save_dir / im_path.name)


def main(args: Namespace) -> None:
    model = Classifier.from_ckpt(args.ckpt_path)

    im_paths = list(args.im_dir.glob('**/*.jpg'))
    labels = predict_arr(model, im_paths, args.im_dir, args.batch_size)

    name_to_enum = get_name_to_enum(DataMode.TAGS)
    names = [name_to_enum.inv[label] for label in labels]

    sign_and_save(im_paths, names, args.save_dir)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Classify images from <im_dir> by model '
                                        'loaded from <ckpt_path> and save result'
                                        'as images with putted tag name to <save_dir>.'
                            )
    parser.add_argument('--im_dir', dest='im_dir', type=Path)
    parser.add_argument('--save_dir', dest='save_dir', type=Path)
    parser.add_argument('--ckpt_path', dest='ckpt_path', type=Path)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=190)
    return parser


if __name__ == '__main__':
    params = get_parser().parse_args()
    main(params)
