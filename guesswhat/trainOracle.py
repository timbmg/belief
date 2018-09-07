import os
import torch
import argparse
from collections import OrderedDict
from torch.utils.data import DataLoader

from models import Oracle
from utils import Vocab, CategoryVocab, OracleDataset


def main(args):

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
            dataset=OracleDataset(file, vocab, category_vocab, True),
            batch_size=args.batch_size,
            shuffle=split == 'train',
            collate_fn=OracleDataset.get_collate_fn(device))

    model = Oracle(len(vocab), args.word_embedding_dim, len(category_vocab),
                   args.category_embedding_dim, args.hidden_size,
                   args.mlp_hidden)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        for split in splits:

            if split == 'train':
                model.train()
                torch.enable_grad()
            else:
                model.eval()
                torch.no_grad()
                
            for bi, batch in enumerate(data_loader[split]):

                logits = model(
                    question=batch['question'],
                    question_lengths=batch['question_lengths'],
                    object_categories=batch['target_category'],
                    object_bboxes=batch['target_bbox'])

                loss = loss_fn(logits, batch['target_answer'])

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)

    parser.add_argument('-d', '--data-dir', type=str, default='data')
    parser.add_argument('-c', '--coco-dir', type=str)

    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-bs', '--batch-size', type=int, default=32)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-mo', '--min-occ', type=int, default=3)

    parser.add_argument('-we', '--word-embedding-dim', type=int, default=256)
    parser.add_argument('-ce', '--category-embedding-dim', type=int,
                        default=256)
    parser.add_argument('-hs', '--hidden-size', type=int, default=512)
    parser.add_argument('-mh', '--mlp-hidden', type=int, default=512)

    args = parser.parse_args()

    main(args)
