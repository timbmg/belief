import os
import torch
import argparse
import datetime
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import QGen
from utils import Vocab, CategoryVocab, QuestionerDataset, eval_epoch


def main(args):

    ts = datetime.datetime.now().timestamp()

    logger = SummaryWriter('exp/qgen/baseline_{}'.format(ts))
    logger.add_text('args', str(args))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocab(os.path.join(args.data_dir, 'vocab.csv'), args.min_occ)
    category_vocab = CategoryVocab(os.path.join(args.data_dir,
                                                'categories.csv'))

    data_loader = OrderedDict()
    splits = ['train', 'valid']
    for split in splits:
        file = os.path.join(args.data_dir, 'guesswhat.' + split + '.jsonl.gz')
        data_loader[split] = DataLoader(
            dataset=QuestionerDataset(file, vocab, category_vocab, True),
            batch_size=args.batch_size,
            shuffle=split == 'train',
            collate_fn=QuestionerDataset.get_collate_fn(device))

    model = QGen(len(vocab), args.word_embedding_dim, args.num_visual_features,
                 args.visual_embedding_dim, args.hidden_size).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    forward_kwargs_mapping = {
        'dialogue': 'source_dialogue',
        'dialogue_lengths': 'dialogue_lengths',
        'additional_features': 'image_featuers'}
    target_kwarg = 'target_dialogue'

    best_val_loss = 1e9
    for epoch in range(args.epochs):
        train_loss, _ = eval_epoch(model, data_loader['train'],
                                   forward_kwargs_mapping, target_kwarg,
                                   loss_fn, optimizer)

        valid_loss, _ = eval_epoch(model, data_loader['valid'],
                                   forward_kwargs_mapping, target_kwarg,
                                   loss_fn)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            model.save()

        logger.add_scalar('train_loss', train_loss, epoch)
        logger.add_scalar('valid_loss', valid_loss, epoch)

        print(("Epoch {:2d}/{:2d} Train Loss {:06.3f} Vaild Loss {:06.3f}")
              .format(epoch, args.epochs, train_loss, valid_loss))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)

    parser.add_argument('-d', '--data-dir', type=str, default='data')
    parser.add_argument('-c', '--coco-dir', type=str)

    parser.add_argument('-ep', '--epochs', type=int, default=30)
    parser.add_argument('-bs', '--batch-size', type=int, default=32)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001)
    parser.add_argument('-mo', '--min-occ', type=int, default=3)

    parser.add_argument('-we', '--word-embedding-dim', type=int, default=512)
    parser.add_argument('-nv', '--num-visual-features', type=int, default=1000)
    parser.add_argument('-ve', '--visual-embedding-dim', type=int,
                        default=512)
    parser.add_argument('-hs', '--hidden-size', type=int, default=1024)

    args = parser.parse_args()

    main(args)
