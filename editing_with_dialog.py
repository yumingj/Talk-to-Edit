import argparse
import json
import logging
import os.path

import numpy as np
import torch

from models import create_model
from utils.dialog_edit_utils import dialog_with_real_user
from utils.inversion_utils import inversion
from utils.logger import get_root_logger
from utils.options import (dict2str, dict_to_nonedict, parse,
                           parse_args_from_opt, parse_opt_wrt_resolution)
from utils.util import make_exp_dirs


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--opt', default=None, type=str, help='Path to option YAML file.')
    return parser.parse_args()


def main():

    # ---------- Set up -----------
    args = parse_args()
    opt = parse(args.opt, is_train=False)
    opt = parse_opt_wrt_resolution(opt)
    args = parse_args_from_opt(args, opt)
    make_exp_dirs(opt)

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)

    # set up logger
    save_log_path = f'{opt["path"]["log"]}'
    dialog_logger = get_root_logger(
        logger_name='dialog',
        log_level=logging.INFO,
        log_file=f'{save_log_path}/dialog.log')
    dialog_logger.info(dict2str(opt))

    save_image_path = f'{opt["path"]["visualization"]}'
    os.makedirs(save_image_path)

    # ---------- Load files -----------
    dialog_logger.info('loading template files')
    with open(opt['feedback_templates_file'], 'r') as f:
        args.feedback_templates = json.load(f)
        args.feedback_replacement = args.feedback_templates['replacement']
    with open(opt['pool_file'], 'r') as f:
        pool = json.load(f)
        args.synonyms_dict = pool["synonyms"]

    # ---------- create model ----------
    field_model = create_model(opt)

    # ---------- load latent code ----------
    if opt['inversion']['is_real_image']:
        latent_code = inversion(opt, field_model)
    else:
        if opt['latent_code_path'] is None:
            latent_code = torch.randn(1, 512, device=torch.device('cuda'))
            with torch.no_grad():
                latent_code = field_model.stylegan_gen.get_latent(latent_code)
            latent_code = latent_code.cpu().numpy()
            np.save(f'{opt["path"]["visualization"]}/latent_code.npz.npy',
                    latent_code)
        else:
            i = opt['latent_code_index']
            latent_code = np.load(
                opt['latent_code_path'],
                allow_pickle=True).item()[f"{str(i).zfill(7)}.png"]
            latent_code = torch.from_numpy(latent_code).to(
                torch.device('cuda'))
            with torch.no_grad():
                latent_code = field_model.stylegan_gen.get_latent(latent_code)
            latent_code = latent_code.cpu().numpy()

    np.save(f'{opt["path"]["visualization"]}/latent_code.npz.npy', latent_code)

    # ---------- Perform dialog-based editing with user -----------
    dialog_overall_log = dialog_with_real_user(field_model, latent_code, opt,
                                               args, dialog_logger)

    # ---------- Log the dialog history -----------
    for (key, value) in dialog_overall_log.items():
        dialog_logger.info(f'{key}: {value}')
    dialog_logger.info('successfully end.')


if __name__ == '__main__':
    main()
