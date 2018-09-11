import os
import torch
import argparse
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import QGen, Guesser, Oracle
from utils import Vocab, CategoryVocab, InferenceDataset
from utils.eval import accuarcy


def main(args):

    logger = SummaryWriter('exp/inference/baseline')

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

    qgen = QGen.load(device)
    guesser = Guesser.load(device)
    oracle = Oracle.load(device)

    torch.no_grad()

    for split in splits:

        for iteration, sample in enumerate(data_loader[split]):

            batch_size = sample['source_dialogue'].size(0)
            dialogue = sample['source_dialogue'].clone()
            dialogue_lengths = dialogue.new_zeros(batch_size)
            additional_features = sample['image_featuers']
            input = torch.LongTensor(batch_size).fill_(vocab['<sos>'])\
                .to(device).unsqueeze(1)

            running_idx = torch.LongTensor(list(range(batch_size))).to(device)
            mask_current = torch.ones(batch_size).byte().to(device)

            # get first question
            questions, questions_lengths, h, c = qgen.inference(
                input=input,
                additional_features=additional_features,
                end_of_question_token=vocab['<eoq>'],
                hidden=None,
                strategy=args.strategy
            )

            for qi in range(1, args.max_num_questions+1):

                # add question to dialogue
                dialogue = append_to_padded_sequence(
                    padded_sequence=dialogue,
                    sequence_lengths=dialogue_lengths,
                    appendix=questions,
                    appendix_lengths=questions_lengths,
                    mask_current=mask_current
                    )

                dialogue_lengths[running_idx] += questions_lengths

                # get answers
                answer_logits = oracle.forward(
                    question=questions,
                    question_lengths=questions_lengths,
                    object_categories=sample['target_category'][running_idx],
                    object_bboxes=sample['target_bbox'][running_idx]
                    )
                answers = answer_logits.topk(1)[1].long()
                answers = answer_class_to_token(answers, vocab.w2i)

                # add answers to dialogue
                dialogue = append_to_padded_sequence(
                    padded_sequence=dialogue,
                    sequence_lengths=dialogue_lengths,
                    appendix=answers,
                    appendix_lengths=answers.new_ones(answers.size(0)),
                    mask_current=mask_current
                    )
                dialogue_lengths[running_idx] += 1

                # ask next question
                questions, questions_lengths, \
                    h[:, running_idx], c[:, running_idx] = qgen.inference(
                        input=answers,
                        additional_features=additional_features[running_idx],
                        end_of_question_token=vocab.w2i['<eoq>'],
                        hidden=(h[:, running_idx], c[:, running_idx]),
                        strategy=args.strategy)

            object_logits = guesser(
                dialogue=dialogue,
                dialogue_lengths=dialogue_lengths,
                object_categories=sample['object_categories'],
                object_bboxes=sample['object_bboxes'])

            acc = accuarcy(object_logits, sample['target_id'])
            print(acc)
            raise


def append_to_padded_sequence(padded_sequence, sequence_lengths, appendix,
                              appendix_lengths, mask_current):

    assert mask_current.sum().item() == appendix.size(0)

    sequence = list()
    lengths = list()

    # get the max length of the new sequences
    appendix_lengths_padded = \
        appendix_lengths.new_zeros(padded_sequence.size(0))
    appendix_lengths_padded.masked_scatter_(mask_current, appendix_lengths)
    appended_sequences_lengths = sequence_lengths + appendix_lengths_padded
    max_length = torch.max(appended_sequences_lengths)

    mi = 0
    for si in range(padded_sequence.size(0)):

        # if dialogue is still running, add item from appendix
        if mask_current[si] == 1:
            # remove padding from padded_sequence; remove padding from appendix
            # concate both
            sequence.append(
                torch.cat((padded_sequence[si, :sequence_lengths[si]],
                           appendix[mi, :appendix_lengths[mi]]), dim=0))
            mi += 1
        else:
            sequence.append(padded_sequence[si, :sequence_lengths[si]])

        lengths.append(len(sequence[-1]))

        # pad new sequence up to max_length
        pad = sequence[-1].new_zeros((max_length-lengths[-1],
                                     *list(sequence[-1].size()[1:])))
        sequence[-1] = torch.cat((sequence[-1], pad))

    sequence = torch.stack(sequence)

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

    args = parser.parse_args()

    main(args)
