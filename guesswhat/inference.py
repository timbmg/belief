import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
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

    if not args.belief_state:
        qgen = QGen.load(device, file=args.qgen_file)
    else:
        qgen = QGenBelief.load(device, file=args.qgen_file)
    guesser = Guesser.load(device, file=args.guesser_file)
    oracle = Oracle.load(device, file=args.oracle_file)
    generation_wrapper = GenerationWrapper(qgen, guesser, oracle)

    ds_kwargs = dict()
    load_vgg_features = qgen.visual_representation == 'vgg'
    if load_vgg_features:
        ds_kwargs['load_vgg_features'] = True

    load_resnet_features = 'resnet' in qgen.visual_representation
    if load_resnet_features:
        ds_kwargs['load_resnet_features'] = True

    if args.mrcnn_belief or args.mrcnn_guesser:
        ds_kwargs['mrcnn_objects'] = True
        ds_kwargs['mrcnn_settings'] = \
            {'filter_category': True, 'skip_below_05': True}

    data_loader = OrderedDict()
    splits = [args.split]
    for split in splits:
        file = os.path.join(args.data_dir, 'guesswhat.' + split + '.jsonl.gz')
        data_loader[split] = DataLoader(
            dataset=InferenceDataset(
                split, file, vocab, category_vocab, args.data_dir, **ds_kwargs),
            batch_size=args.batch_size,
            collate_fn=InferenceDataset.get_collate_fn(device))

    torch.no_grad()
    dump_data = defaultdict(dict)
    for split in splits:

        total_acc = list()
        multi_target_acc = list()
        with tqdm(total=len(data_loader[split]), desc=split, unit='batches') as pbar:
            for iteration, sample in enumerate(data_loader[split]):
                return_dict = generation_wrapper.generate(
                    sample, vocab, args.strategy, args.max_num_questions,
                    device, args.belief_state, args.mrcnn_belief,
                    args.mrcnn_guesser,
                    return_keys=['generations', 'log_probs', 'hidden_states',
                                 'mask', 'object_probs'])

                if args.mrcnn_belief and not args.mrcnn_guesser:
                    target_id = sample['gt_target_id']
                else:
                    target_id = sample['target_id']

                acc = accuarcy(return_dict['object_logits'], target_id)
                total_acc += [acc]

                if args.mrcnn_guesser:
                    multi_target_acc += \
                        [multi_target_accuracy(return_dict['object_logits'],
                                               sample['multi_target_mask'])]

                if args.save_data:
                    dump_data = save_data(sample, dump_data, return_dict,
                                          split, vocab, category_vocab,
                                          args.max_num_questions)

                pbar.update(1)

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
            json.dump(dump_data, open(
                'analysis/{}_{}.json'.format(
                    os.path.basename(os.path.normpath(args.qgen_file)),
                    split), 'w'))


def save_data(sample, save_data, return_dict, split, vocab, category_vocab,
              max_num_questions):
    for i, game_id in enumerate(sample['game_id']):
        game_id = game_id.item()
        no = sample['num_objects'][i].item()
        dl = torch.sum(return_dict['mask'], 1)[i].item()

        save_data[game_id]['split'] = split
        save_data[game_id]['dialogue'] = ' '.join(vocab.decode(
            return_dict['dialogue'][i].tolist()[:dl+max_num_questions+1]))
        save_data[game_id]['guesser_probs'] = \
            [op[:no] for op in return_dict['guesser_probs'][i].tolist()]
        if len(return_dict['belief_probs']) > 0:
            save_data[game_id]['belief_probs'] = \
                [op[:no] for op in return_dict['belief_probs'][i].tolist()]
        save_data[game_id]['object_categories'] = \
            category_vocab.decode(
                sample['object_categories'][i].tolist())[:no]
        save_data[game_id]['bbox'] = \
            sample['orignal_bboxes'][i].tolist()[:no]
        save_data[game_id]['target_id'] = \
            sample['target_id'][i].item()
        save_data[game_id]['prediction'] = \
            int(np.argmax(save_data[game_id]['guesser_probs'][-1]))

        save_data[game_id]['success'] = \
            bool(return_dict['object_logits'][i].topk(1)[1] ==
                 sample['target_id'][i])
        save_data[game_id]['image_width'] = \
            sample['image_width'][i].item()
        save_data[game_id]['image_height'] = \
            sample['image_height'][i].item()
        save_data[game_id]['image_file'] = sample['image'][i]
        save_data[game_id]['image_url'] = sample['image_url'][i]

    return save_data


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
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-nw', '--num_workers', type=int, default=2)
    parser.add_argument('-sp', '--split', choices=['train', 'valid', 'test'], default='valid')

    # .pt files
    parser.add_argument('-oracle', '--oracle-file', type=str,
                        default='bin/oracle.pt')
    parser.add_argument('-guesser', '--guesser-file', type=str,
                        default='bin/guesser.pt')
    parser.add_argument('-qgen', '--qgen-file', type=str, required=True)

    args = parser.parse_args()

    main(args)
