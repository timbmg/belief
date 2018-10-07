import os
import torch
import argparse
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader

from models import QGen, Guesser, Oracle, QGenBelief
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

    torch.no_grad()
    belief_kwargs = dict()

    for split in splits:

        total_acc = list()
        for iteration, sample in enumerate(data_loader[split]):

            batch_size = sample['source_dialogue'].size(0)
            dialogue = sample['source_dialogue'].clone()
            dialogue_lengths = dialogue.new_ones(batch_size)
            visual_features = sample['image_featuers']
            input = torch.LongTensor(batch_size).fill_(vocab['<sos>'])\
                .to(device).unsqueeze(1)

            if args.belief_state:
                belief_kwargs['dialogue'] = dialogue
                belief_kwargs['dialogue_lengths'] = dialogue_lengths
                belief_kwargs['object_categories'] = \
                    sample['object_categories']
                belief_kwargs['object_bboxes'] = sample['object_bboxes']
                belief_kwargs['num_objects'] = sample['num_objects']

            questions_lengths, h, c, return_dict = qgen.inference(
                input=input,
                visual_features=visual_features,
                end_of_question_token=vocab['<eoq>'],
                hidden=None,
                strategy=args.strategy,
                return_keys=['generations'],
                **belief_kwargs)

            for _ in range(1, args.max_num_questions+1):

                # add question to dialogue
                dialogue = append_to_padded_sequence(
                    padded_sequence=dialogue,
                    sequence_lengths=dialogue_lengths,
                    appendix=return_dict['generations'],
                    appendix_lengths=questions_lengths)
                dialogue_lengths += questions_lengths

                # get answers
                answer_logits = oracle.forward(
                    question=return_dict['generations'],
                    question_lengths=questions_lengths,
                    object_categories=sample['target_category'],
                    object_bboxes=sample['target_bbox']
                    )
                answers = answer_logits.topk(1)[1].long()
                answers = answer_class_to_token(answers, vocab.w2i)

                # add answers to dialogue
                dialogue = append_to_padded_sequence(
                    padded_sequence=dialogue,
                    sequence_lengths=dialogue_lengths,
                    appendix=answers,
                    appendix_lengths=answers.new_ones(answers.size(0)))
                dialogue_lengths += 1

                if args.belief_state:
                    # update dialogue with new q/a pair
                    belief_kwargs['dialogue'] = dialogue
                    belief_kwargs['dialogue_lengths'] = dialogue_lengths

                # ask next question
                questions_lengths, h, c, return_dict = qgen.inference(
                    input=answers,
                    visual_features=visual_features,
                    end_of_question_token=vocab.w2i['<eoq>'],
                    hidden=(h, c),
                    strategy=args.strategy,
                    return_keys=['generations'],
                    **belief_kwargs)

            object_logits = guesser(
                dialogue=dialogue,
                dialogue_lengths=dialogue_lengths,
                object_categories=sample['object_categories'],
                object_bboxes=sample['object_bboxes'],
                num_objects=sample['num_objects'])

            acc = accuarcy(object_logits, sample['target_id'])
            total_acc += [acc]
            # print(np.mean(total_acc))

        print("{} Accuracy {}".format(split.upper(), np.mean(total_acc)))


def append_to_padded_sequence(padded_sequence, sequence_lengths, appendix,
                              appendix_lengths):

    max_length = torch.max(sequence_lengths + appendix_lengths)
    sequence = padded_sequence.new_zeros((padded_sequence.size(0), max_length))

    for si in range(len(padded_sequence)):
        new_length = sequence_lengths[si].item() + appendix_lengths[si].item()
        sequence[si, :new_length] = torch.cat(
            (padded_sequence[si, :sequence_lengths[si]],
             appendix[si, :appendix_lengths[si]]))

    return sequence


def answer_class_to_token(answers, w2i):

    yes_mask = answers == 0
    no_mask = answers == 1
    na_mask = answers == 2

    answers.masked_fill_(yes_mask, w2i['<yes>'])
    answers.masked_fill_(no_mask, w2i['<no>'])
    answers.masked_fill_(na_mask, w2i['<n/a>'])

    return answers


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
