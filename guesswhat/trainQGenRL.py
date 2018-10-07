import os
import time
import torch
import argparse
import datetime
import numpy as np
from tensorboardX import SummaryWriter
from collections import OrderedDict
from torch.utils.data import DataLoader

from models import QGen, Guesser, Oracle, QGenBelief, MLP
from utils import Vocab, CategoryVocab, InferenceDataset, Optimizer
from utils.eval import accuarcy


def main(args):

    ts = datetime.datetime.now().timestamp()

    logger = SummaryWriter(os.path.join('exp/qgen_rl/',
                                        '{}_{}'.format(args.exp_name, ts)))
    logger.add_text('exp_name', args.exp_name)
    logger.add_text('args', str(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
            dataset=InferenceDataset(file, vocab, category_vocab,
                                     new_object=split == 'train'),
            batch_size=args.batch_size,
            collate_fn=InferenceDataset.get_collate_fn(device),
            shuffle=split == 'train')

    if not args.belief_state:
        qgen = QGen.load(device, file=args.qgen_file)
    else:
        qgen = QGenBelief.load(device, file=args.qgen_file)
    guesser = Guesser.load(device, file=args.guesser_file)
    oracle = Oracle.load(device, file=args.oracle_file)

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

    belief_kwargs = dict()

    best_val_acc = 0
    for epoch in range(args.epochs):

        for split in splits:

            total_acc = list()
            for iteration, sample in enumerate(data_loader[split]):

                # t1 = time.time()

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
                    max_tokens=args.max_question_tokens,
                    strategy=split2strat[split],
                    **belief_kwargs)

                hidden_states = torch.Tensor().to(device)
                mask = torch.ByteTensor().to(device)
                log_probs = torch.Tensor().to(device)

                for _ in range(1, args.max_num_questions+1):

                    # add question to dialogue
                    dialogue = append_to_padded_sequence(
                        padded_sequence=dialogue,
                        sequence_lengths=dialogue_lengths,
                        appendix=return_dict['generations'],
                        appendix_lengths=questions_lengths)
                    dialogue_lengths += questions_lengths

                    # save hidden states
                    hidden_states = torch.cat(
                        (hidden_states, return_dict['hidden_states']), dim=1)
                    mask = torch.cat((mask, return_dict['mask']), dim=1)
                    log_probs = torch.cat(
                        (log_probs, return_dict['log_probs']), dim=1)

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
                        max_tokens=args.max_question_tokens,
                        strategy=split2strat[split],
                        **belief_kwargs)

                object_logits = guesser(
                    dialogue=dialogue,
                    dialogue_lengths=dialogue_lengths,
                    object_categories=sample['object_categories'],
                    object_bboxes=sample['object_bboxes'],
                    num_objects=sample['num_objects'])

                acc = accuarcy(object_logits,  sample['target_id'])
                total_acc += [acc]

                mask = mask.float()

                rewards = torch.eq(object_logits.topk(1)[1].view(-1),
                                   sample['target_id'].view(-1)).float()
                rewards = rewards.unsqueeze(1).repeat(1, mask.size(1))
                rewards *= mask

                # cum_rewards = torch.cumsum(rewards, dim=1)
                # cum_rewards *= mask

                baseline_preds = baseline(hidden_states.detach_()).squeeze(2)
                baseline_preds *= mask
                baseline_loss = baseline_loss_fn(baseline_preds.view(-1),
                                                 rewards.view(-1)) / batch_size

                log_probs *= mask
                baseline_preds = baseline_preds.detach()
                policy_gradient_loss = torch.sum(
                    -log_probs * (rewards - baseline_preds), dim=1)
                policy_gradient_loss = torch.mean(policy_gradient_loss)

                # rewards = rewards.masked_select(mask)
                #
                # baseline_preds = baseline(hidden_states.detach_()).squeeze(2)
                # baseline_preds = baseline_preds.masked_select(mask)
                # baseline_loss = baseline_loss_fn(baseline_preds, rewards)\
                #     / batch_size
                #
                # log_probs = log_probs.masked_select(mask)
                # policy_gradient_loss = torch.sum(
                #     log_probs * (rewards - baseline_preds.detach_())) \
                #     / batch_size

                # print(baseline_loss.item())
                # print(policy_gradient_loss.item())

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

                acc = accuarcy(object_logits, sample['target_id'])
                total_acc += [acc]
                # if iteration % 10 == 0:
                #     print(("Iter {:4d} T: {:5.2f} Avg Acc: {:6.3f} " +
                #            "BL Loss {:6.3f} PG Loss {:6.3f}")
                #           .format(iteration, time.time()-t1,
                #                   np.mean(total_acc) * 100,
                #                   baseline_loss.item(),
                #                   policy_gradient_loss.item()))
                #     t1 = time.time()

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
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-exp', '--exp-name', type=str, required=True)
    parser.add_argument('-ep', '--epochs', type=int, default=100)

    # Dataset Settings
    parser.add_argument('-d', '--data_dir', type=str, default='data')

    # Experiment Settings
    parser.add_argument('-mq', '--max_num_questions', type=int, default=5)
    parser.add_argument('-mt', '--max_question_tokens', type=int, default=12)
    parser.add_argument('-tstg', '--train-strategy',
                        choices=['greedy', 'sampling'], required=True)
    parser.add_argument('-vstg', '--eval-strategy',
                        choices=['greedy', 'sampling'], required=True)
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