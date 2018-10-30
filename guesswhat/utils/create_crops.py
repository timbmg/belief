import os
import json
import gzip
import argparse
from PIL import Image


def main(args):

    train_folder = os.path.join(args.mscoco_dir, 'train2014')
    valid_folder = os.path.join(args.mscoco_dir, 'val2014')

    crops_folder = os.path.join(args.mscoco_dir, 'guesswhat_crops')
    if not os.path.exists(crops_folder):
        os.mkdir(crops_folder)

    splits = ['train', 'valid', 'test']
    files = [os.path.join(args.data_dir, 'guesswhat.' + split + '.jsonl.gz')
             for split in splits]

    for file in files:
        with gzip.open(file, 'r') as file:
            for json_game in file:
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

                crop.save(crop_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='../data')
    parser.add_argument('-mc', '--mscoco_dir', type=str,
                        default='/Users/timbaumgartner/MSCOCO')

    args = parser.parse_args()
    main(args)
