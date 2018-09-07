import os
import gzip
import h5py
import json
import torch
import argparse
from PIL import Image
from multiprocessing import cpu_count
from torchvision.models import vgg16
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.no_grad()


class ImageDataset(Dataset):

    def __init__(self, data_dir, mscoco_dir, splits=['train', 'valid', 'test']):

        self.train_coco = os.path.join(mscoco_dir, 'train2014')
        self.valid_coco = os.path.join(mscoco_dir, 'val2014')

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        self.data = list()
        for split in splits:
            file = os.path.join(data_dir, 'guesswhat.' + split + '.jsonl.gz')
            with gzip.open(file, 'r') as file:

                for json_game in file:
                    game = json.loads(json_game.decode("utf-8"))
                    self.data += [game['image']['file_name']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.train_coco if 'train' in self.data[idx]
                                else self.valid_coco, self.data[idx])
        img = Image.open(img_path).convert('RGB')
        proc_img = self.transform(img)

        return {
            'image_file_name': self.data[idx],
            'processed_image': proc_img
        }


def main(args):

    h5_file = h5py.File(os.path.join(args.data_dir, "vgg_fc8.hdf5"), "w")
    dset = None

    vgg = vgg16(pretrained=True)
    vgg.eval()
    vgg.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        vgg = torch.nn.DataParallel(vgg)

    data_loader = DataLoader(ImageDataset(args.data_dir, args.mscoco_dir),
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    id2file = list()
    for i, sample in enumerate(data_loader):

        id2file += sample['image_file_name']
        fc8 = vgg(sample['processed_image'].to(device)).cpu().detach().numpy()

        if dset is not None:
            idx = dset.shape[0]
            dset.resize((idx+fc8.shape[0], 1000))
        else:
            idx = 0
            dset = h5_file.create_dataset("vgg_fc8",
                                          (fc8.shape[0], 1000),
                                          maxshape=(None, 1000),
                                          dtype='f')

        dset[idx:] = fc8

        if i % 100 == 0:
            print("{}/{}".format(i, len(data_loader)))

    mapping = dict()
    for file in id2file:
        mapping[file] = len(mapping)

    json.dump(mapping, open(os.path.join(args.data_dir, 'imagefile2id.json'),
                            'w'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-mc', '--mscoco_dir', type=str,
                        default='/Users/timbaumgartner/MSCOCO')
    parser.add_argument('-nw', '--num_workers', type=int, default=2)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)

    args = parser.parse_args()
    args.num_workers = min(cpu_count(), args.num_workers)

    main(args)
