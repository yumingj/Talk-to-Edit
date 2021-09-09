import argparse
import json
import sys
import time

import torch
import torch.nn as nn
import torch.utils.data

sys.path.append('.')
from accuracy import head_accuracy  # noqa
from dataset import EncoderDataset  # noqa
from lstm import Encoder  # noqa
from utils import AverageMeter, dict2str, save_checkpoint  # noqa
from utils.setup_logger import setup_logger  # noqa


def parse_args():
    """Parses arguments."""

    parser = argparse.ArgumentParser(description='Train the language encoder')

    # mode
    parser.add_argument('--debug', type=int, default=0)

    # training
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--val_batch', type=int, default=1024)

    # learning rate scheme
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)

    # LSTM hyperparameter
    parser.add_argument('--word_embedding_dim', default=300, type=int)
    parser.add_argument('--text_embed_size', default=1024, type=int)
    parser.add_argument('--linear_hidden_size', default=256, type=int)
    parser.add_argument('--linear_dropout_rate', default=0, type=float)

    # input directories
    parser.add_argument(
        '--vocab_file', required=True, type=str, help='path to vocab file.')
    parser.add_argument(
        '--metadata_file',
        default='./templates/metadata_fsm.json',
        type=str,
        help='path to metadata file.')
    parser.add_argument(
        '--train_set_dir', required=True, type=str, help='path to train data.')
    parser.add_argument(
        '--val_set_dir', required=True, type=str, help='path to val data.')
    # output directories
    parser.add_argument(
        '--work_dir',
        required=True,
        type=str,
        help='path to save checkpoint and log files.')

    # misc
    parser.add_argument(
        '--unlabeled_value',
        default=999,
        type=int,
        help='value to represent unlabeled value')
    parser.add_argument('--num_workers', default=8, type=int)

    return parser.parse_args()


best_val_acc, best_epoch, current_iters = 0, 0, 0


def main():
    """Main function."""

    # ################### Set Up #######################
    global args, best_val_acc, best_epoch

    args = parse_args()
    logger = setup_logger(
        args.work_dir, logger_name='train.txt', debug=args.debug)

    args.device = torch.device('cuda')

    logger.info('Saving arguments.')
    logger.info(dict2str(args.__dict__))

    # ################### Metadata #######################
    with open(args.metadata_file, 'r') as f:
        args.metadata = json.load(f)
        args.num_head = len(args.metadata.items())
    logger.info(f'args.num_head: {args.num_head}, ')
    logger.info(f'args.metadata: {args.metadata}.')

    # ################### Language Encoder #######################

    # load vocab file
    with open(args.vocab_file, 'r') as f:
        vocab = json.load(f)
    text_token_to_idx = vocab['text_token_to_idx']

    encoder = Encoder(
        token_to_idx=text_token_to_idx,
        word_embedding_dim=args.word_embedding_dim,
        text_embed_size=args.text_embed_size,
        metadata_file=args.metadata_file,
        linear_hidden_size=args.linear_hidden_size,
        linear_dropout_rate=args.linear_dropout_rate)
    encoder = encoder.to(args.device)

    # ################### DataLoader #######################

    logger.info('Preparing train_dataset')

    train_dataset = EncoderDataset(preprocessed_dir=args.train_set_dir)
    logger.info('Preparing train_loader')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=None)
    logger.info('Preparing val_dataset')
    val_dataset = EncoderDataset(preprocessed_dir=args.val_set_dir)
    logger.info('Preparing val_loader')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False)
    logger.info(f'Number of train text: {len(train_dataset)}, '
                f'Number of val text: {len(val_dataset)}.')
    data_loader = {
        'train': train_loader,
        'val': val_loader,
    }

    # ################### Optimizer #######################

    optimizer = torch.optim.Adam(
        encoder.parameters(), args.lr, weight_decay=args.weight_decay)

    # ################### Loss Function #######################
    criterion = nn.CrossEntropyLoss(
        reduction='mean', ignore_index=args.unlabeled_value)

    # ################### Epochs #######################

    for epoch in range(args.num_epochs):
        logger.info(
            '----------- Training: Epoch '
            f'({epoch + 1} / {args.num_epochs}),  LR: {args.lr:.4f}. ---------'
        )
        train_per_head_acc_avg, train_overall_acc = train(
            args,
            'train',
            encoder,
            data_loader['train'],
            criterion,
            optimizer,
            logger,
        )
        logger.info(
            'Train accuracy '
            f'({epoch + 1} / {args.num_epochs}), '
            f'{[str(round(i, 2))+"%" for i in train_per_head_acc_avg]}')
        val_per_head_acc_avg, val_overall_acc = train(
            args,
            'val',
            encoder,
            data_loader['val'],
            criterion,
            optimizer,
            logger,
        )
        logger.info('Validation accuracy '
                    f'({epoch + 1} / {args.num_epochs}), '
                    f'{[str(round(i, 2))+"%" for i in val_per_head_acc_avg]}')

        # whether this epoch has the highest val acc so far
        is_best = val_overall_acc > best_val_acc
        if is_best:
            best_epoch = epoch + 1
            best_val_acc = val_overall_acc
        logger.info(
            f'Best Epoch: {best_epoch}, best acc: {best_val_acc: .4f}.')
        save_checkpoint(
            args, {
                'epoch': epoch + 1,
                'best_epoch_so_far': best_epoch,
                'state_dict': encoder.state_dict(),
                'best_val_acc': best_val_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            checkpoint=args.work_dir)
    logger.info('successful')


def train(args, phase, encoder, data_loader, criterion, optimizer, logger):

    if phase == 'train':
        encoder.train()
    else:
        encoder.eval()

    # record time
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    # record accuracy
    per_head_acc_list = [AverageMeter() for _ in range(args.num_head)]

    for batch_idx, batch_data in enumerate(data_loader):
        data_time.update(time.time() - end)

        text, system_mode, labels = batch_data
        text = text.to(args.device)
        system_mode = system_mode.to(args.device)
        labels = labels.to(args.device)

        if phase == 'train':
            output = encoder(text)
        else:
            with torch.no_grad():
                output = encoder(text)
        loss_list = []

        # Labels: loss and acc
        for head_idx, (key, val) in enumerate(args.metadata.items()):
            loss = criterion(output[head_idx], labels[:, head_idx])
            loss_list.append(loss)
            acc_dict = head_accuracy(
                output=output[head_idx],
                target=labels[:, head_idx],
                unlabeled_value=args.unlabeled_value)
            acc = acc_dict['acc']
            labeled_count = int(acc_dict['labeled_count'])
            if labeled_count > 0:
                per_head_acc_list[head_idx].update(acc, labeled_count)

        loss_avg = sum(loss_list) / len(loss_list)

        if phase == 'train':
            optimizer.zero_grad()
            loss_avg.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.info(
            f'Batch: {batch_idx+1}, '
            f'Data time: {data_time.avg:.3f}s, Batch time: {batch_time.avg:.3f}s, '  # noqa
            f'loss: {loss_avg:.4f}.')

    overall_acc = 0
    per_head_acc_avg = []
    for head_idx in range(args.num_head):
        per_head_acc_avg.append(per_head_acc_list[head_idx].avg)
        overall_acc += per_head_acc_list[head_idx].avg
    overall_acc = overall_acc / args.num_head
    return per_head_acc_avg, overall_acc


if __name__ == '__main__':
    main()
