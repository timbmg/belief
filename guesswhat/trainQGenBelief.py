import os
import torch
import argparse
import datetime
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import QGen, Guesser, QGenBelief
from utils import Vocab, CategoryVocab, QuestionerDataset, eval_epoch


def main(args):

    ts = datetime.datetime.now().timestamp()

    logger = SummaryWriter(os.path.join('exp/qgenbelief/',
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
            dataset=QuestionerDataset(file, vocab, category_vocab, True,
                                      cumulative_dialogue=True),
            batch_size=args.batch_size,
            shuffle=split == 'train',
            collate_fn=QuestionerDataset.get_collate_fn(device))

    guesser = Guesser.load(device, file=args.guesser_file)
    qgen = QGen(len(vocab), args.word_embedding_dim, args.num_visual_features,
                args.visual_embedding_dim, args.hidden_size,
                args.category_embedding_dim).to(device)
    model = QGenBelief(qgen, guesser, args.category_embedding_dim,
                       args.object_embedding_setting,
                       args.object_probs_setting,
                       args.train_guesser_setting).to(device)
    print(model)
    logger.add_text('model', str(model))

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.train_guesser_setting:
        optimizer_guesser = torch.optim.Adam(guesser.parameters(),
                                             lr=args.learning_rate_guesser)
        optimizer = [optimizer, optimizer_guesser]

    forward_kwargs_mapping = {
        'dialogue': 'source_dialogue',
        'dialogue_lengths': 'dialogue_lengths',
        'visual_features': 'image_featuers',
        'cumulative_dialogue': 'cumulative_dialogue',
        'cumulative_lengths': 'cumulative_lengths',
        'num_questions': 'num_questions',
        'object_categories': 'object_categories',
        'object_bboxes': 'object_bboxes',
        'num_objects': 'num_objects'}
    target_kwarg = 'target_dialogue'

    best_val_loss = 1e9
    for epoch in range(args.epochs):
        train_loss, train_acc = eval_epoch(model, data_loader['train'],
                                           forward_kwargs_mapping,
                                           target_kwarg, loss_fn, optimizer)

        valid_loss, valid_acc = eval_epoch(model, data_loader['valid'],
                                           forward_kwargs_mapping,
                                           target_kwarg, loss_fn)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            model.save(os.path.join('bin', 'qgenbelief_{}_{}.pt'
                                    .format(args.exp_name, ts)))

        logger.add_scalar('train_loss', train_loss, epoch)
        logger.add_scalar('valid_loss', valid_loss, epoch)

        print(("Epoch {:2d}/{:2d} Train Loss {:07.4f} Vaild Loss {:07.4f}")
              .format(epoch, args.epochs, train_loss, valid_loss))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-d', '--data-dir', type=str, default='data')
    parser.add_argument('-exp', '--exp-name', type=str, required=True)
    parser.add_argument('-g', '--guesser-file', type=str,
                        default='bin/guesser.pt')

    # Hyperparameter
    parser.add_argument('-ep', '--epochs', type=int, default=15)
    parser.add_argument('-bs', '--batch-size', type=int, default=32)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001)
    parser.add_argument('-lrg', '--learning-rate-guesser', type=float,
                        default=0.00001)
    parser.add_argument('-mo', '--min-occ', type=int, default=3)
    parser.add_argument('-we', '--word-embedding-dim', type=int, default=512)
    parser.add_argument('-ce', '--category-embedding-dim', type=int,
                        default=512)
    parser.add_argument('-nv', '--num-visual-features', type=int, default=1000)
    parser.add_argument('-ve', '--visual-embedding-dim', type=int,
                        default=512)
    parser.add_argument('-hs', '--hidden-size', type=int, default=1024)

    # Settings
    parser.add_argument('-oe', '--object-embedding-setting',
                        choices=['learn-emb', 'from-guesser'],
                        default='learn-emb',
                        help='Determines, which object embeddings to use. '
                        + 'Either a new embedding matrix is learned '
                        + 'end-2-end, or the object embedding (including the '
                        + 'bbox information) form the guesser is used.')
    parser.add_argument('-op', '--object-probs-setting',
                        choices=['guesser-probs', 'uniform',
                                 'guesser-all-categories'],
                        default='guesser-probs',
                        help='Determines how the probabilites from the '
                        + 'guesser are used. If uniform, the probabilites are '
                        + 'not used and all object emebddings are simply '
                        + 'averaged. If guesser-probs, a weighted average '
                        + 'with the guesser probabilites over the object '
                        + 'embeddings is used. If guesser-all-categories, '
                        + 'the guesser returns probabilites for all MS COCO '
                        + 'categoires and these are used for the weighted '
                        + 'average.')
    parser.add_argument('-tg', '--train-guesser-setting', action='store_true',
                        help='If true, the guesser is also updated during '
                        + 'the training process.')

    args = parser.parse_args()

    main(args)
