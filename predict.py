from argparse import ArgumentParser, Namespace
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple

import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import put_text_to_image
from dataset import ImagesDataset
from network import Classifier
from sun_data.utils import get_name_to_enum, DataMode


def predict_arr(model: Classifier,
                im_paths: List[Path],
                device: torch.device,
                batch_size: int,
                size: int
                ) -> Tuple[List[int], List[float]]:
    model.to(device)
    dataset = ImagesDataset(data_root=Path('/'),
                            im_paths=im_paths,
                            labels_enum=[1] * len(im_paths)
                            )
    dataset.set_defaukt_transforms_with_resize((size, size))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4
                        )
    labels: List[int] = []
    probs: List[float] = []
    for im, _ in tqdm(loader):
        label, prob = model.classify(im.to(device))
        labels.extend(label.detach().cpu().tolist())
        probs.extend(prob.detach().cpu().tolist())

    return labels, probs


def sign_and_save(im_paths: List[Path],
                  names: List[str],
                  probs: List[float],
                  save_dir: Path
                  ) -> None:
    for im_path, name, prob in tqdm(zip(im_paths, names, probs), total=len(probs)):
        image = np.array(PIL.Image.open(im_path).convert('RGB'))
        image_signed = put_text_to_image(image=image, strings=[name], color=(0, 255, 0))
        image_signed = PIL.Image.fromarray(image_signed).convert('RGB')
        image_signed.save(save_dir / f'{round(prob, 3)}_{im_path.name}')


def main(args: Namespace) -> None:
    model, _ = Classifier.from_ckpt(args.ckpt_path)

    im_paths = list(args.im_dir.glob('**/*.jpg'))
    labels, probs = predict_arr(model=model, im_paths=im_paths, size=args.size,
                                device=args.device, batch_size=args.batch_size)

    name_to_enum = get_name_to_enum(DataMode.TAGS)
    names = [name_to_enum.inv[label] for label, prob in zip(labels, probs)]

    if args.drop_other:
        ii_not_other = [i for i, name in enumerate(names) if name != 'other']
        n_drop = len(probs) - len(ii_not_other)
        print(f'{n_drop} other-predicts will be dropped.')
        names = itemgetter(*ii_not_other)(names)
        im_paths = itemgetter(*ii_not_other)(im_paths)
        probs = itemgetter(*ii_not_other)(probs)

    sign_and_save(im_paths=im_paths, names=names, probs=probs, save_dir=args.save_dir)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Classify images from <im_dir> by model '
                                        'loaded from <ckpt_path> and save result'
                                        'as images with putted tag name to <save_dir>.'
                            )
    parser.add_argument('--im_dir', dest='im_dir', type=Path)
    parser.add_argument('--save_dir', dest='save_dir', type=Path)
    parser.add_argument('--ckpt_path', dest='ckpt_path', type=Path)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=160)
    parser.add_argument('--device', dest='device', type=torch.device, default='cuda:3')
    parser.add_argument('--size', dest='size', type=int, default=512)
    parser.add_argument('--drop_other', dest='drop_other', type=int, default=True)
    return parser


if __name__ == '__main__':
    params = get_parser().parse_args()
    main(params)
