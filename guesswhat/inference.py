import os
import torch
import argparse
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader

from models import QGen, Guesser, Oracle, QGenBelief, GenerationWrapper
from utils import Vocab, CategoryVocab, InferenceDataset
from utils.eval import accuarcy


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocab(os.path.join(args.data_dir, 'vocab.csv'), 3)
    category_vocab = CategoryVocab(os.path.join(args.data_dir,
                                                'categories.csv'))

    data_loader = OrderedDict()
    splits = (['train'] if args.train_set else list()) + ['valid'] + \
             (['test'] if args.test_set else list())
    for split in splits:
        file = os.path.join(args.data_dir, 'guesswhat.' + split + '.jsonl.gz')
        data_loader[split] = DataLoader(
            dataset=InferenceDataset(file, vocab, category_vocab),
            batch_size=args.batch_size,
            collate_fn=InferenceDataset.get_collate_fn(device))

    if not args.belief_state:
        qgen = QGen.load(device, file=args.qgen_file)
    else:
        qgen = QGenBelief.load(device, file=args.qgen_file)
    guesser = Guesser.load(device, file=args.guesser_file)
    oracle = Oracle.load(device, file=args.oracle_file)

    generation_wrapper = GenerationWrapper(qgen, guesser, oracle)

    torch.no_grad()
    for split in splits:

        total_acc = list()
        for iteration, sample in enumerate(data_loader[split]):
            return_dict = generation_wrapper.generate(
                sample, vocab, args.strategy, args.max_num_questions, device,
                args.belief_state)

            acc = accuarcy(return_dict['object_logits'], sample['target_id'])
            total_acc += [acc]
            print(np.mean(total_acc))

        print("{} Accuracy {}".format(split.upper(), np.mean(total_acc)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-belief', '--belief-state', action='store_true')

    # Dataset Settings
    parser.add_argument('-d', '--data_dir', type=str, default='data')

    # Experiment Settings
    parser.add_argument('-mq', '--max_num_questions', type=int, default=5)
    parser.add_argument('-s', '--strategy', choices=['greedy', 'sampling'],
                        default='greedy')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-nw', '--num_workers', type=int, default=2)
    parser.add_argument('-train', '--train_set', action='store_true')
    parser.add_argument('-test', '--test_set', action='store_true')

    # .pt files
    parser.add_argument('-oracle', '--oracle-file', type=str,
                        default='bin/oracle.pt')
    parser.add_argument('-guesser', '--guesser-file', type=str,
                        default='bin/guesser.pt')
    parser.add_argument('-qgen', '--qgen-file', type=str, required=True)

    args = parser.parse_args()

    main(args)
