import os
import torch
import argparse
import datetime
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import Oracle, filmed_resnet50, MultiHopFiLM
from utils import Vocab, CategoryVocab, OracleDataset, eval_epoch


def main(args):
    print(args)

    if not args.eval:
        ts = datetime.datetime.now().timestamp()
        logger = SummaryWriter(os.path.join('exp/oracle/',
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

    # if args.use_film:
    #     filmed_resnet = filmed_resnet50(langugae_embedding_size=args.hidden_size)
    #     # filmwrapper = torch.nn.DataParallel(filmwrapper, device_ids=[0, 1, 2, 3])
    # else:
    #     filmed_resnet = None

    #global_film = MultiHopFiLM(128, args.hidden_size, 128)
    global_film = None
    crop_film = None

    data_loader = OrderedDict()
    if not args.eval:
        splits = ['train', 'valid']
    else:
        splits = ['valid', 'test']

    for split in splits:
        file = os.path.join(args.data_dir, 'guesswhat.' + split + '.jsonl.gz')
        data_loader[split] = DataLoader(
            dataset=OracleDataset(
                file, vocab, category_vocab, True,
                load_crops=args.use_film,
                crops_folder=args.crops_folder,
                global_features=args.global_features,
                global_mapping=args.global_mapping,
                crop_features=args.crop_features,
                crop_mapping=args.crop_mapping),
            batch_size=args.batch_size,
            shuffle=split == 'train',
            collate_fn=OracleDataset.get_collate_fn(device))

    if not args.eval:
        model = Oracle(len(vocab), args.word_embedding_dim, len(category_vocab),
                       args.category_embedding_dim, args.hidden_size,
                       args.mlp_hidden, global_film, crop_film).to(device)
    else:
        model = Oracle.load(device, file=args.bin)

    loss_fn = torch.nn.CrossEntropyLoss()

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    forward_kwargs_mapping = {
        'question': 'question',
        'question_lengths': 'question_lengths',
        #'question_mask': 'question_mask',
        'object_categories': 'target_category',
        'object_bboxes': 'target_bbox'}
        #'global_features': 'global_features',
        #'crop_features': 'crop_features'}
    if args.use_film:
        forward_kwargs_mapping['crop'] = 'crop'
    target_kwarg = 'target_answer'

    best_val_acc = 0

    for epoch in range(args.epochs):
        if not args.eval:
            train_loss, train_acc = eval_epoch(model, data_loader['train'],
                                               forward_kwargs_mapping,
                                               target_kwarg, loss_fn, optimizer,
                                               clip_norm_args=[args.clip_value])

        valid_loss, valid_acc = eval_epoch(model, data_loader['valid'],
                                           forward_kwargs_mapping,
                                           target_kwarg, loss_fn)
        if args.eval:
            test_loss, test_acc = eval_epoch(model, data_loader['test'],
                                             forward_kwargs_mapping,
                                             target_kwarg, loss_fn)

            print("Valid Loss {:07.4f} Valid Acc {:07.4f}".format(
                valid_loss, valid_acc))
            print("Test Loss {:07.4f} Test Acc {:07.4f}".format(
                test_loss, test_acc))

            break

        else:
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                model.save(os.path.join('bin', 'oracle_{}_{}.pt'
                                        .format(args.exp_name, ts)))

            logger.add_scalar('train_loss', train_loss, epoch)
            logger.add_scalar('valid_loss', valid_loss, epoch)
            logger.add_scalar('train_acc', train_acc, epoch)
            logger.add_scalar('valid_acc', valid_acc, epoch)

            print(("Epoch {:2d}/{:2d} Train Loss {:07.4f} Vaild Loss {:07.4f} " +
                   "Train Acc {:07.4f} Vaild Acc {:07.4f}")
                  .format(epoch, args.epochs, train_loss, valid_loss,
                          train_acc*100, valid_acc*100))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-d', '--data-dir', type=str, default='data')
    parser.add_argument('-exp', '--exp-name', type=str, required=True)
    parser.add_argument('-eval', '--eval', action='store_true')
    parser.add_argument('-bin', '--bin', type=str, default='bin/oracle.pt')

    parser.add_argument('-film', '--use_film', action='store_true')
    parser.add_argument('-crops', '--crops-folder', type=str,
                        default='/Users/timbaumgartner/MSCOCO/guesswhat_crops')

    parser.add_argument('-global-features', '--global-features', type=str,
                        default='data/resnet152_block3.hdf5')
    parser.add_argument('-global-mapping', '--global-mapping', type=str,
                        default='data/resnet_block3_imagefile2id.json')

    parser.add_argument('-crop-features', '--crop-features', type=str,
                        # default='data/resnet152_block3.hdf5')
                        default='')
    parser.add_argument('-crop-mapping', '--crop-mapping', type=str,
                        # default='data/resnet_block3_imagefile2id.json')
                        default='')

    parser.add_argument('-f-hidden', '--film-mlp-hidden', type=int,
                        default=512)

    parser.add_argument('-ep', '--epochs', type=int, default=20)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001)
    parser.add_argument('-mo', '--min-occ', type=int, default=3)
    parser.add_argument('-gc', '--clip_value', type=float, default=3)

    parser.add_argument('-we', '--word-embedding-dim', type=int, default=300)
    parser.add_argument('-ce', '--category-embedding-dim', type=int,
                        default=512)
    parser.add_argument('-hs', '--hidden-size', type=int, default=512)
    parser.add_argument('-mh', '--mlp-hidden', type=int, default=512)

    args = parser.parse_args()

    main(args)
