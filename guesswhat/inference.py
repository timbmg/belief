import os
import json
import torch
import argparse
import numpy as np
from collections import OrderedDict, defaultdict
from torch.utils.data import DataLoader

from models import QGen, Guesser, Oracle, QGenBelief, GenerationWrapper
from utils import Vocab, CategoryVocab, InferenceDataset
from utils.eval import accuarcy, multi_target_accuracy


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocab(os.path.join(args.data_dir, 'vocab.csv'), 3)
    category_vocab = CategoryVocab(os.path.join(args.data_dir,
                                                'categories.csv'))

    data_loader = OrderedDict()
    splits = (['train'] if args.train_set else list()) + ['valid'] + \
             (['test'] if args.test_set else list())
    ds_kwargs = dict()
    if args.mrcnn_belief or args.mrcnn_guesser:
        ds_kwargs['mrcnn_objects'] = True
        ds_kwargs['mrcnn_settings'] = \
            {'filter_category': True, 'skip_below_05': True}
    for split in splits:
        file = os.path.join(args.data_dir, 'guesswhat.' + split + '.jsonl.gz')
        data_loader[split] = DataLoader(
            dataset=InferenceDataset(file, vocab, category_vocab, **ds_kwargs),
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
    save_data = defaultdict(dict)
    for split in splits:

        total_acc = list()
        multi_target_acc = list()
        for iteration, sample in enumerate(data_loader[split]):
            return_dict = generation_wrapper.generate(
                sample, vocab, args.strategy, args.max_num_questions, device,
                args.belief_state, args.mrcnn_belief, args.mrcnn_guesser,
                return_keys=['generations', 'log_probs', 'hidden_states',
                             'mask', 'object_probs'])

            if args.mrcnn_belief and not args.mrcnn_guesser:
                target_id = sample['gt_target_id']
            else:
                target_id = sample['target_id']

            acc = accuarcy(return_dict['object_logits'], target_id)
            total_acc += [acc]
            # print(np.mean(total_acc))

            if args.mrcnn_guesser:
                multi_target_acc += \
                    [multi_target_accuracy(return_dict['object_logits'],
                                           sample['multi_target_mask'])]

            if args.save_data:
                for i, game_id in enumerate(sample['game_id']):
                    game_id = game_id.item()
                    no = sample['num_objects'][i].item()
                    dl = torch.sum(return_dict['mask'], 1)[i].item()

                    save_data[game_id]['split'] = split
                    save_data[game_id]['dialogue'] = ' '.join(vocab.decode(
                        return_dict['dialogue'][i].tolist()[:dl+6]))
                    save_data[game_id]['object_probs'] = \
                        [op[:no] for op in return_dict['object_probs'][i].tolist()]
                    save_data[game_id]['object_categories'] = \
                        category_vocab.decode(
                            sample['object_categories'][i].tolist())[:no]
                    save_data[game_id]['bbox'] = \
                        sample['orignal_bboxes'][i].tolist()[:no]
                    save_data[game_id]['target_id'] = \
                        sample['target_id'][i].item()
                    save_data[game_id]['prediction'] = \
                        int(np.argmax(save_data[game_id]['object_probs'][-1]))

                    save_data[game_id]['success'] = \
                        bool(return_dict['object_logits'][i].topk(1)[1] == sample['target_id'][i])
                    save_data[game_id]['image_width'] = sample['image_width'][i].item()
                    save_data[game_id]['image_height'] = sample['image_height'][i].item()
                    save_data[game_id]['image_file'] = sample['image'][i]
                    save_data[game_id]['image_url'] = sample['image_url'][i]
                break

            if args.save_data:
                json.dump(save_data, open('generations/{}.json'.format(
                    os.path.basename(os.path.normpath(args.qgen_file))), 'w'))

        print("{} Accuracy {}".format(split.upper(), np.mean(total_acc)))

        if args.mrcnn_guesser:
            total_acc = np.mean(total_acc) \
                * len(data_loader[split].dataset) / \
                (len(data_loader[split].dataset)
                 + data_loader[split].dataset.skipped_datapoints)
            print("{} Total Split Accuracy {}"
                  .format(split.upper(), total_acc))

            print("{} Multi Target Accuracy {}"
                  .format(split.upper(), np.mean(multi_target_acc)))
            multi_target_acc = np.mean(multi_target_acc) \
                * len(data_loader[split].dataset) / \
                (len(data_loader[split].dataset)
                 + data_loader[split].dataset.skipped_datapoints)
            print("{} Multi Target Total Split Accuracy {}"
                  .format(split.upper(), multi_target_acc))

        if args.save_data:
            json.dump(save_data, open('generations/{}.json'.format(os.path.basename(os.path.normpath(args.qgen_file))), 'w'))

def save_data(sample, return_dict):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-save', '--save-data', action='store_true')

    parser.add_argument('-belief', '--belief-state', action='store_true')
    parser.add_argument('-mrcnn-belief', '--mrcnn-belief', action='store_true',
                        help='Indicates whether the guesser used in the ' +
                        'QGenBelief is based on mrcnn or not.')
    parser.add_argument('-mrcnn-guesser', '--mrcnn-guesser',
                        action='store_true', help='Indicates wheather the ' +
                        'guesser used for guessing the object at the end of ' +
                        'the game is based on mrcnn or not.')
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
