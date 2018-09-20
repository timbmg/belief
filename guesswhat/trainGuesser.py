import os
import torch
import argparse
import datetime
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import Guesser
from utils import Vocab, CategoryVocab, QuestionerDataset, eval_epoch


def main(args):

    ts = datetime.datetime.now().timestamp()

    logger = SummaryWriter(os.path.join('exp/guesser/',
                                        '{}_{}'.format(args.exp_name, ts)))
    logger.add_text('exp_name', args.exp_name)
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

    model = Guesser(len(vocab), args.word_embedding_dim, len(category_vocab),
                    args.category_embedding_dim, args.hidden_size,
                    args.mlp_hidden).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    forward_kwargs_mapping = {
        'dialogue': 'source_dialogue',
        'dialogue_lengths': 'dialogue_lengths',
        'object_categories': 'object_categories',
        'object_bboxes': 'object_bboxes'}
    target_kwarg = 'target_id'

    best_val_acc = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = eval_epoch(model, data_loader['train'],
                                           forward_kwargs_mapping,
                                           target_kwarg, loss_fn, optimizer)

        valid_loss, valid_acc = eval_epoch(model, data_loader['valid'],
                                           forward_kwargs_mapping,
                                           target_kwarg, loss_fn)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            model.save(os.path.join('bin', 'guesser_{}_{}.pt'
                                    .format(args.exp_name, ts)))

        logger.add_scalar('train_loss', train_loss, epoch)
        logger.add_scalar('valid_loss', valid_loss, epoch)
        logger.add_scalar('train_acc', train_acc, epoch)
        logger.add_scalar('valid_acc', valid_acc, epoch)

        print(("Epoch {:2d}/{:2d} Train Loss {:07.4f} Vaild Loss {:07.4f} " +
               "Train Acc {:07.4f} Vaild Acc {:07.4f}")
              .format(epoch, args.epochs, train_loss, valid_loss,
                      train_acc*100, valid_acc*100))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-d', '--data-dir', type=str, default='data')
    parser.add_argument('-exp', '--exp-name', type=str, required=True)

    parser.add_argument('-ep', '--epochs', type=int, default=20)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001)
    parser.add_argument('-mo', '--min-occ', type=int, default=3)

    parser.add_argument('-we', '--word-embedding-dim', type=int, default=512)
    parser.add_argument('-ce', '--category-embedding-dim', type=int,
                        default=256)
    parser.add_argument('-hs', '--hidden-size', type=int, default=512)
    parser.add_argument('-mh', '--mlp-hidden', type=int, default=512)

    args = parser.parse_args()

    main(args)
