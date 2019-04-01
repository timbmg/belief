import os
import time
import torch
import argparse
import datetime
import numpy as np
from tensorboardX import SummaryWriter
from collections import OrderedDict
from torch.utils.data import DataLoader

from models import QGen, Guesser, Oracle, QGenBelief, MLP, GenerationWrapper
from utils import Vocab, CategoryVocab, InferenceDataset, Optimizer
from utils.eval import accuarcy


def main(args):

    ts = datetime.datetime.now().timestamp()

    logger = SummaryWriter(os.path.join('exp/qgen_rl/',
                                        '{}_{}'.format(args.exp_name, ts)))
    logger.add_text('exp_name', args.exp_name)
    logger.add_text('args', str(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocab(os.path.join(args.data_dir, 'vocab.csv'), 3)
    category_vocab = CategoryVocab(os.path.join(args.data_dir,
                                                'categories.csv'))

    data_loader = OrderedDict()
    splits = ['train', 'valid'] + (['test'] if args.test_set else list())
    for split in splits:
        file = os.path.join(args.data_dir, 'guesswhat.' + split + '.jsonl.gz')
        data_loader[split] = DataLoader(
            dataset=InferenceDataset(split, file, vocab, category_vocab,
                                     new_object=split == 'train',
                                     load_vgg_features=True),
            batch_size=args.batch_size,
            collate_fn=InferenceDataset.get_collate_fn(device),
            shuffle=split == 'train')

    if not args.belief_state:
        qgen = QGen.load(device, file=args.qgen_file)
    else:
        qgen = QGenBelief.load(device, file=args.qgen_file)
    guesser = Guesser.load(device, file=args.guesser_file)
    oracle = Oracle.load(device, file=args.oracle_file)

    generation_wrapper = GenerationWrapper(qgen, guesser, oracle)

    baseline = MLP(
        sizes=[qgen.hidden_size, args.baseline_hidden_size, 1],
        activation='relu', final_activation='relu', bias=[True, False])\
        .to(device)

    baseline_loss_fn = torch.nn.MSELoss(reduction='sum')
    baseline_optimizer = Optimizer(torch.optim.SGD, baseline.parameters(),
                                   lr=args.baseline_lr)
    qgen_optimizer = Optimizer(torch.optim.SGD, qgen.parameters(),
                               lr=args.qgen_lr)

    split2strat = {'train': args.train_strategy, 'valid': args.eval_strategy,
                   'test': args.eval_strategy}

    best_val_acc = 0
    for epoch in range(args.epochs):

        for split in splits:

            if split == 'train':
                qgen.train()
                baseline.train()
                torch.enable_grad()
            else:
                qgen.eval()
                baseline.eval()
                torch.no_grad()

            total_acc = list()
            for iteration, sample in enumerate(data_loader[split]):

                return_dict = generation_wrapper.generate(
                    sample, vocab, split2strat[split], args.max_num_questions,
                    device, args.belief_state,
                    return_keys=['mask', 'object_logits', 'hidden_states',
                                 'log_probs', 'generations'])

                mask = return_dict['mask']
                object_logits = return_dict['object_logits']
                hidden_states = return_dict['hidden_states']
                log_probs = return_dict['log_probs']

                acc = accuarcy(object_logits,  sample['target_id'])
                total_acc += [acc]

                mask = mask.float()

                rewards = torch.eq(object_logits.topk(1)[1].view(-1),
                                   sample['target_id'].view(-1)).float()
                rewards = rewards.unsqueeze(1).repeat(1, mask.size(1))
                rewards *= mask


                print("dialogue", return_dict['dialogue'][0], return_dict['dialogue'].size())
                #print("log_probs", log_probs, log_probs.size())
                #print("mask", mask, mask.size())
                #print("rewards", rewards, rewards.size())

                baseline_preds = baseline(hidden_states.detach_()).squeeze(2)
                baseline_preds *= mask
                baseline_loss = baseline_loss_fn(
                    baseline_preds.view(-1), rewards.view(-1)) \
                    / baseline_preds.size(0)

                log_probs *= mask
                baseline_preds = baseline_preds.detach()
                policy_gradient_loss = torch.sum(
                    log_probs * (rewards - baseline_preds), dim=1)
                print(policy_gradient_loss)
                policy_gradient_loss = -torch.mean(policy_gradient_loss)
                print()
                raise
                # policy_gradient_loss = - torch.sum(log_probs) / torch.sum(mask)
                #print(policy_gradient_loss_old.item(), policy_gradient_loss.item())

                if split == 'train':
                    qgen_optimizer.optimize(
                        policy_gradient_loss, clip_norm_args=[args.clip_value])
                    baseline_optimizer.optimize(
                        baseline_loss, clip_norm_args=[args.clip_value])

                logger.add_scalar('{}_accuracy'.format(split), acc,
                                  iteration + len(data_loader[split])*epoch)

                logger.add_scalar('{}_reward'.format(split),
                                  torch.mean(rewards).item(),
                                  iteration + len(data_loader[split])*epoch)

                logger.add_scalar('{}_bl_loss'.format(split),
                                  baseline_loss.item(),
                                  iteration + len(data_loader[split])*epoch)

                logger.add_scalar('{}_pg_loss'.format(split),
                                  policy_gradient_loss.item(),
                                  iteration + len(data_loader[split])*epoch)

            model_saved = False
            if split == 'valid':
                if np.mean(total_acc) > best_val_acc:
                    best_val_acc = np.mean(total_acc)
                    qgen.save(file='bin/qgen_rl_{}_{}.pt'
                                   .format(args.exp_name, ts),
                              accuarcy=np.mean(total_acc))
                    model_saved = True

            logger.add_scalar('epoch_{}_accuracy'.format(split),
                              np.mean(total_acc), epoch)

            print("Epoch {:3d}: {} Accuracy {:5.3f} {}"
                  .format(epoch, split.upper(), np.mean(total_acc)*100,
                          '*' if model_saved else ''))
        print("-"*50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-exp', '--exp-name', type=str, required=True)
    parser.add_argument('-ep', '--epochs', type=int, default=100)

    # Dataset Settings
    parser.add_argument('-d', '--data_dir', type=str, default='data')

    # Experiment Settings
    parser.add_argument('-mq', '--max_num_questions', type=int, default=8)
    parser.add_argument('-mt', '--max_question_tokens', type=int, default=12)
    parser.add_argument('-tstg', '--train-strategy',
                        choices=['greedy', 'sampling'], default='sampling')
    parser.add_argument('-estg', '--eval-strategy',
                        choices=['greedy', 'sampling'], default='greedy')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-test', '--test_set', action='store_true')

    # .pt files
    parser.add_argument('-oracle', '--oracle-file', type=str,
                        default='bin/oracle.pt')
    parser.add_argument('-guesser', '--guesser-file', type=str,
                        default='bin/guesser.pt')
    parser.add_argument('-qgen', '--qgen-file', type=str, required=True)
    parser.add_argument('-belief', '--belief-state', action='store_true')

    # Hyperparameter
    parser.add_argument('-blh', '--baseline_hidden_size', type=int,
                        default=128)
    parser.add_argument('-blr', '--baseline_lr', type=float, default=0.001)
    parser.add_argument('-qlr', '--qgen_lr', type=float, default=0.001)
    parser.add_argument('-gc', '--clip_value', type=float, default=5)

    args = parser.parse_args()

    main(args)
