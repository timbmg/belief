import os
import gzip
import h5py
import json
import torch
import argparse
import torchvision
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from multiprocessing import cpu_count
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.no_grad()


class ImageDataset(Dataset):

    def __init__(self, data_dir, mscoco_dir, split):

        self.train_coco = os.path.join(mscoco_dir, 'train2014')
        self.valid_coco = os.path.join(mscoco_dir, 'val2014')

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        self.data = set()
        file = os.path.join(data_dir, 'guesswhat.' + split + '.jsonl.gz')
        with gzip.open(file, 'r') as file:

            for json_game in file:
                game = json.loads(json_game.decode("utf-8"))
                self.data.add(game['image']['file_name'])

        self.data = list(self.data)

    def __len__(self):
        #return 100
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

    resnet = torchvision.models.resnet152(pretrained=True).to(device)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-3])
    resnet.eval()
    torch.no_grad()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        resnet = torch.nn.DataParallel(resnet)

    print(resnet)

    h5_file = h5py.File(os.path.join(
        args.data_dir, "resnet152_block3.hdf5"), "w")
    id2file = defaultdict(list)

    for split in ['train', 'valid', 'test']:
        grp = h5_file.create_group(split)
        dset = None

        data_loader = DataLoader(
            ImageDataset(args.data_dir, args.mscoco_dir, split),
            batch_size=args.batch_size,
            num_workers=args.num_workers)

        with tqdm(total=len(data_loader), desc=split, unit='batch') as pbar:
            for i, sample in enumerate(data_loader):

                id2file[split] += sample['image_file_name']
                block3 = resnet(sample['processed_image'].to(device))\
                    .cpu().detach().numpy()

                if dset is not None:
                    idx = dset.shape[0]
                    dset.resize((idx+block3.shape[0], 1024, 14, 14))
                else:
                    idx = 0
                    dset = grp.create_dataset("resnet152_block3",
                                              (block3.shape[0], 1024, 14, 14),
                                              maxshape=(None, 1024, 14, 14),
                                              dtype='f')

                dset[idx:] = block3

                pbar.update(1)
                # if i % 100 == 0:
                #     print("{}/{}".format(i, len(data_loader)))

    mapping = defaultdict(dict)
    for split in ['train', 'valid', 'test']:
        for id, file in enumerate(id2file[split]):
            mapping[split][file] = id
    mapping = dict(mapping)

    json.dump(mapping, open(os.path.join(
        args.data_dir, 'resnet_block3_imagefile2id.json'), 'w'))


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
