import os
import json
import gzip
import argparse
from tqdm import tqdm
from PIL import Image

from torchvision import transforms


def main(args):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
        transforms.ToPILImage()])

    train_folder = os.path.join(args.mscoco_dir, 'train2014')
    valid_folder = os.path.join(args.mscoco_dir, 'val2014')

    crops_folder = os.path.join(args.mscoco_dir, 'guesswhat_crops'
                                + '_transformed' if args.transform else '')
    if not os.path.exists(crops_folder):
        os.mkdir(crops_folder)

    splits = ['train', 'valid', 'test']
    files = [os.path.join(args.data_dir, 'guesswhat.' + split + '.jsonl.gz')
             for split in splits]

    with tqdm(total=155280) as pbar:
        for file in files:
            with gzip.open(file, 'r') as file:
                for json_game in file:
                    pbar.update(1)
                    game = json.loads(json_game.decode("utf-8"))
                    crop_filename = str(game['id']) + '.jpg'
                    crop_path = os.path.join(crops_folder, crop_filename)
                    if os.path.exists(crop_path):
                        continue
                    image_file = game['image']['file_name']

                    # get bounding box
                    for oi, obj in enumerate(game['objects']):
                        if obj['id'] == game['object_id']:
                            x1 = obj['bbox'][0]
                            y1 = obj['bbox'][1]
                            x2 = x1 + obj['bbox'][2]
                            y2 = y1 + obj['bbox'][3]

                            break

                    # load and crop image
                    if 'train' in image_file:
                        folder = train_folder
                    elif 'val' in image_file:
                        folder = valid_folder
                    else:
                        raise
                    image_path = os.path.join(folder, game['image']['file_name'])
                    crop = Image.open(image_path).crop((x1, y1, x2, y2))

                    if args.transform:
                        crop = transform(crop)

                    crop.save(crop_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='../data')
    parser.add_argument('-mc', '--mscoco_dir', type=str,
                        default='/Users/timbaumgartner/MSCOCO')
    parser.add_argument('-t', '--transform', action='store_true')


    args = parser.parse_args()
    main(args)
