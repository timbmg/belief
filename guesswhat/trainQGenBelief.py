import os
import torch
import argparse
from tensorboardX import SummaryWriter
from collections import OrderedDict
from torch.utils.data import DataLoader

from models import QGen, Guesser, QGenBelief
from utils import Vocab, CategoryVocab, QuestionerDataset, eval_epoch


def main(args):

    logger = SummaryWriter('exp/qgenbelief')
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
            dataset=QuestionerDataset(file, vocab, category_vocab, True,
                                      unroll_dialogue=True),
            batch_size=args.batch_size,
            shuffle=split == 'train',
            collate_fn=QuestionerDataset.get_collate_fn(device))

    guesser = Guesser.load(device)
    qgen = QGen(len(vocab), args.word_embedding_dim, args.num_visual_features,
                args.visual_embedding_dim, args.hidden_size).to(device)
    model = QGenBelief(qgen, guesser).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    forward_kwargs_mapping = {
        'source_questions': 'source_questions',
        'question_lengths': 'question_lengths',
        'visual_features': 'image_featuers',
        'unrolled_dialogue': 'unrolled_dialogue',
        'cumulative_lengths': 'cumulative_lengths',
        'num_questions': 'num_questions',
        'object_categories': 'object_categories',
        'object_bboxes': 'object_bboxes'}
    target_kwarg = 'target_questions'

    best_val_loss = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = eval_epoch(model, data_loader['train'],
                                           forward_kwargs_mapping,
                                           target_kwarg, loss_fn, optimizer)

        valid_loss, valid_acc = eval_epoch(model, data_loader['valid'],
                                           forward_kwargs_mapping,
                                           target_kwarg, loss_fn)

        if valid_loss > best_val_loss:
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
