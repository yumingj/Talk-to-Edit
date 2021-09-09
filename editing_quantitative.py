import argparse
import logging
import os

import numpy as np

from models import create_model
from utils.logger import get_root_logger
from utils.numerical_metrics import compute_num_metrics
from utils.options import dict2str, dict_to_nonedict, parse
from utils.util import make_exp_dirs


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    parser.add_argument(
        '--pretrained_path', type=str, help='Path to pretrained field model')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=False)

    # mkdir and loggers
    make_exp_dirs(opt)

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)

    # load editing latent code
    editing_latent_codes = np.load(opt['editing_latent_code_path'])
    num_latent_codes = editing_latent_codes.shape[0]

    save_path = f'{opt["path"]["visualization"]}'
    os.makedirs(save_path)
    editing_logger = get_root_logger(
        logger_name='editing',
        log_level=logging.INFO,
        log_file=f'{save_path}/editing.log')

    editing_logger.info(dict2str(opt))

    field_model = create_model(opt)

    field_model.load_network(args.pretrained_path)

    field_model.continuous_editing(editing_latent_codes, save_path,
                                   editing_logger)

    _, _ = compute_num_metrics(save_path, num_latent_codes,
                               opt['pretrained_arcface'], opt['attr_file'],
                               opt['predictor_ckpt'],
                               opt['attr_dict'][opt['attribute']],
                               editing_logger)


if __name__ == '__main__':
    main()
