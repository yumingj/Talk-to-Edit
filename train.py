import argparse
import logging
import os
import os.path as osp
import random
import time

import numpy as np
import torch

from data.latent_code_dataset import LatentCodeDataset
from models import create_model
from utils.logger import MessageLogger, get_root_logger, init_tb_logger
from utils.numerical_metrics import compute_num_metrics
from utils.options import dict2str, dict_to_nonedict, parse
from utils.util import make_exp_dirs


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True)

    # mkdir and loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}.log")
    logger = get_root_logger(
        logger_name='base', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))
    # initialize tensorboard logger
    tb_logger = None
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir='./tb_logger/' + opt['name'])

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)

    # random seed
    seed = opt['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info(f'Random seed: {seed}')

    # set up data loader
    logger.info(f'Loading data from{opt["input_latent_dir"]}.')
    train_latent_dataset = LatentCodeDataset(
        input_dir=opt['dataset']['train_latent_dir'])
    train_latent_loader = torch.utils.data.DataLoader(
        dataset=train_latent_dataset,
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=opt['num_workers'],
        drop_last=True)
    logger.info(f'Number of train set: {len(train_latent_dataset)}.')
    opt['max_iters'] = opt['num_epochs'] * len(
        train_latent_dataset) // opt['batch_size']
    if opt['val_on_train_subset']:
        train_subset_latent_dataset = LatentCodeDataset(
            input_dir=opt['dataset']['train_subset_latent_dir'])
        train_subset_latent_loader = torch.utils.data.DataLoader(
            dataset=train_subset_latent_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=opt['num_workers'])
        logger.info(
            f'Number of train subset: {len(train_subset_latent_dataset)}.')
    if opt['val_on_valset']:
        val_latent_dataset = LatentCodeDataset(
            input_dir=opt['dataset']['val_latent_dir'])
        val_latent_loader = torch.utils.data.DataLoader(
            dataset=val_latent_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=opt['num_workers'])
        logger.info(f'Number of val set: {len(val_latent_dataset)}.')

    # load editing latent code
    editing_latent_codes = np.load(opt['editing_latent_code_path'])
    num_latent_codes = editing_latent_codes.shape[0]

    current_iter = 0
    best_metric = 10000
    best_epoch = None
    best_arcface = None
    best_predictor = None

    field_model = create_model(opt)

    data_time, iter_time = 0, 0
    current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    for epoch in range(opt['num_epochs']):
        lr = field_model.update_learning_rate(epoch)

        for _, batch_data in enumerate(train_latent_loader):
            data_time = time.time() - data_time

            current_iter += 1

            field_model.feed_data(batch_data)
            field_model.optimize_parameters()

            iter_time = time.time() - iter_time
            if current_iter % opt['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': [lr]})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(field_model.get_current_log())
                msg_logger(log_vars)

            data_time = time.time()
            iter_time = time.time()

        if epoch % opt['val_freq'] == 0:
            if opt['val_on_valset']:
                save_dir = f'{opt["path"]["visualization"]}/valset/epoch_{epoch:03d}'  # noqa
                os.makedirs(save_dir, exist_ok=opt['debug'])
                for batch_idx, batch_data in enumerate(val_latent_loader):
                    field_model.feed_data(batch_data)
                    field_model.inference(batch_idx, epoch, save_dir)
            if opt['val_on_train_subset']:
                save_dir = f'{opt["path"]["visualization"]}/trainsubset/epoch_{epoch:03d}'  # noqa
                os.makedirs(save_dir, exist_ok=opt['debug'])
                for batch_idx, batch_data in enumerate(
                        train_subset_latent_loader):
                    field_model.feed_data(batch_data)
                    field_model.inference(batch_idx, epoch, save_dir)

            save_path = f'{opt["path"]["visualization"]}/continuous_editing/epoch_{epoch:03d}'  # noqa
            os.makedirs(save_path, exist_ok=opt['debug'])
            editing_logger = get_root_logger(
                logger_name=f'editing_{epoch:03d}',
                log_level=logging.INFO,
                log_file=f'{save_path}/editing.log')

            field_model.continuous_editing(editing_latent_codes, save_path,
                                           editing_logger)

            arcface_sim, predictor_score = compute_num_metrics(
                save_path, num_latent_codes, opt['pretrained_arcface'],
                opt['attr_file'], opt['predictor_ckpt'],
                opt['attr_dict'][opt['attribute']], editing_logger)

            logger.info(f'Epoch: {epoch}, '
                        f'ArcFace: {arcface_sim: .4f}, '
                        f'Predictor: {predictor_score: .4f}.')

            metrics = 1 - arcface_sim + predictor_score

            if metrics < best_metric:
                best_epoch = epoch
                best_metric = metrics
                best_arcface = arcface_sim
                best_predictor = predictor_score

            logger.info(f'Best epoch: {best_epoch}, '
                        f'ArcFace: {best_arcface: .4f}, '
                        f'Predictor: {best_predictor: .4f}.')

            # save model
            field_model.save_network(
                field_model.field_function,
                f'{opt["path"]["models"]}/ckpt_epoch{epoch}.pth')


if __name__ == '__main__':
    main()
