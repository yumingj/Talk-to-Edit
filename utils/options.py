import os
import os.path as osp
from collections import OrderedDict

import yaml


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    if opt.get('set_CUDA_VISIBLE_DEVICES', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        # print('export CUDA_VISIBLE_DEVICES=' + gpu_list, flush=True)
    else:
        pass
        # print('gpu_list: ', gpu_list, flush=True)

    opt['is_train'] = is_train

    # datasets
    if opt['is_train']:
        input_latent_dir = opt['input_latent_dir']
        opt['dataset'] = {}
        opt['dataset']['train_latent_dir'] = f'{input_latent_dir}/train'
        if opt['val_on_train_subset']:
            opt['dataset'][
                'train_subset_latent_dir'] = f'{input_latent_dir}/train_subset'
        if opt['val_on_valset']:
            opt['dataset']['val_latent_dir'] = f'{input_latent_dir}/val'

    # paths
    opt['path'] = {}
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['debug'] = True
            opt['val_freq'] = 1
            opt['print_freq'] = 1
            opt['save_checkpoint_freq'] = 1
            opt['dataset'][
                'train_latent_dir'] = f'{input_latent_dir}/train_subset'
            if opt['val_on_train_subset']:
                opt['dataset'][
                    'train_subset_latent_dir'] = f'{input_latent_dir}/train_subset'  # noqa
            if opt['val_on_valset']:
                opt['dataset'][
                    'val_latent_dir'] = f'{input_latent_dir}/train_subset'
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')
    # some basics for editing task
    opt['attr_list'] = ['Bangs', 'Eyeglasses', 'No_Beard', 'Smiling', 'Young']
    opt['attr_dict'] = {
        'Bangs': 0,
        'Eyeglasses': 1,
        'No_Beard': 2,
        'Smiling': 3,
        'Young': 4
    }

    if 'has_dialog' in opt.keys():
        opt['path']['dialog'] = osp.join(results_root, 'dialog')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':[\n'
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    """None dict. It will return none if key is not in the dict."""

    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    """Convert to NoneDict, which returns None for missing keys.

    Args:
        opt (dict): Option dict.

    Returns:
        (dict): NoneDict for options.
    """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def parse_args_from_opt(args, opt):
    '''
    Given the opt, parse it to args,
    since previous code for dialog and language
    uses args to pass arguments among different scripts
    '''
    for (key, value) in list(opt.items()):
        setattr(args, key, value)
    for (key, value) in list(opt['language_encoder'].items()):
        setattr(args, key, value)
    args.pretrained_checkpoint = opt['pretrained_language_encoder']
    return args


def parse_opt_wrt_resolution(opt):
    if opt['img_res'] == 1024:
        opt['channel_multiplier'] = opt['channel_multiplier_1024']
        opt['pretrained_field'] = opt['pretrained_field_1024']
        opt['predictor_ckpt'] = opt['predictor_ckpt_1024']
        opt['generator_ckpt'] = opt['generator_ckpt_1024']
        opt['replaced_layers'] = opt['replaced_layers_1024']

    elif opt['img_res'] == 128:
        opt['channel_multiplier'] = opt['channel_multiplier_128']
        opt['pretrained_field'] = opt['pretrained_field_128']
        opt['predictor_ckpt'] = opt['predictor_ckpt_128']
        opt['generator_ckpt'] = opt['generator_ckpt_128']
        opt['replaced_layers'] = opt['replaced_layers_128']

    return opt
