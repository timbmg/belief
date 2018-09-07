import os
import torch
import argparse
from collections import OrderedDict
from torch.utils.data import DataLoader

from models import Guesser
from utils import Vocab, CategoryVocab, QuestionerDataset


def main(args):

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
                    dialogue=batch['source_dialogue'],
                    dialogue_lengths=batch['dialogue_lengths'],
                    object_categories=batch['object_categories'],
                    object_bboxes=batch['object_bboxes'])

                loss = loss_fn(logits, batch['target_id'])

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, default='data')
    parser.add_argument('-c', '--coco-dir', type=str)

    parser.add_argument('-ep', '--epochs', type=int, default=20)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-mo', '--min-occ', type=int, default=3)

    parser.add_argument('-we', '--word-embedding-dim', type=int, default=512)
    parser.add_argument('-ce', '--category-embedding-dim', type=int,
                        default=256)
    parser.add_argument('-hs', '--hidden-size', type=int, default=512)
    parser.add_argument('-mh', '--mlp-hidden', type=int, default=512)

    args = parser.parse_args()

    main(args)
