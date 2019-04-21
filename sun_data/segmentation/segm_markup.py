import gzip
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import numpy as np
import xmltodict
from PIL import Image, ImageDraw
from tqdm import tqdm


def main(data_dir: Path, save_dir: Path) -> None:
    annot_paths = list(data_dir.glob('**/*.xml'))

    for annot_path in tqdm(annot_paths):
        try:
            im_path = annot_to_im_path(annot_path)
            im = Image.open(im_path).convert('RGB')
            width, height = im.width, im.height

            with open(annot_path, 'r') as f:
                content_str = f.read()

            masks, labels, folder = xml_to_mask(content_str, width, height)

            cur_dir = save_dir / folder
            cur_dir.mkdir(exist_ok=True, parents=True)
            save_path = cur_dir / f'{annot_path.stem}.zpkl'
            with gzip.open(save_path, 'wb') as f:
                data = {'masks': masks, 'labels': labels}
                pickle.dump(data, f, protocol=3)
        except Exception:
            print(f'Broken path {annot_path}')


def xml_to_mask(xml_content: str,
                width: int,
                height: int
                ) -> Tuple[np.ndarray, List[str], str]:
    data = xmltodict.parse(xml_content)

    folder = Path(data['annotation']['folder'])

    masks, labels = [], []
    for obj in data['annotation']['object']:
        pts = obj['polygon']['pt']
        polygon = [(int(pt['x']), int(pt['y'])) for pt in pts]
        mask = polygon_to_mask(polygon, width, height)
        masks.append(mask)

        labels.append(obj['name'])

    masks_stacked = np.stack(masks).astype(np.bool)
    masks_stacked = np.transpose(masks_stacked, (1, 2, 0))

    return masks_stacked, labels, folder


def polygon_to_mask(polygon: List[Tuple[int, int]],
                    width: int,
                    height: int
                    ) -> np.ndarray:
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)
    return mask


def annot_to_im_path(annot_path: Path) -> Path:
    im_path = str(annot_path)
    im_path = im_path.replace('Annotations', 'Images')
    im_path = im_path.replace('.xml', '.jpg')
    return Path(im_path)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', type=Path)
    parser.add_argument('--save_dir', dest='save_dir', type=Path)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(data_dir=args.data_dir, save_dir=args.save_dir)
