from __future__ import absolute_import

import datetime
import logging
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

# from mmcv.runner import get_dist_info, master_only

__all__ = [
    'Logger', 'LoggerMonitor', 'savefig', 'MessageLogger', 'init_tb_logger',
    'get_root_logger', 'dict2str'
]


def savefig(fname, dpi=None):
    dpi = 150 if dpi is None else dpi
    plt.savefig(fname, dpi=dpi)


def plot_overlap(logger, names=None):
    names = logger.names if names is None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]


class Logger(object):
    '''Save training process to log file with simple plot function.'''

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title is None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            if type(num) == int:
                self.file.write(str(num))
            elif type(num) == float:
                self.file.write("{0:.6f}".format(num))
            else:  # str
                self.file.write(str(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, out_file, names=None):
        names = self.names if names is None else names
        numbers = self.numbers
        fig, ax = plt.subplots(1, 1)
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            ax.plot(x, numbers[name])

            # whether add data labels to each point in the plot
            if False:
                for i in range(len(x)):
                    y = numbers[name][i]
                    # text = round(y, 2) # below 4 line are added by ziqi
                    if type(y) == int or type(y) == float:
                        text = round(y, 2)
                    else:
                        text = y
                    ax.text(x[i], y, text)

        ax.legend([self.title + '(' + name + ')' for name in names])
        loc = plticker.MultipleLocator(
            base=1.0
        )  # this locator puts ticks at regular intervals # ziqi added
        ax.xaxis.set_major_locator(loc)
        ax.grid(True)
        plt.savefig(out_file)
        plt.close()

    def close(self):
        if self.file is not None:
            self.file.close()

    def get_numbers(self):
        stats = {}
        for name in self.names:
            stats[name] = self.numbers[name]
        return stats


class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''

    def __init__(self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(
            legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'niter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['niter']
        self.use_tb_logger = opt['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    # @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, '
                   f'iter:{current_iter:8,d}, lr:(')
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '

        # time and estimated time
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time: {iter_time:.3f}, data_time: {data_time:.3f}] '

        # other items, especially losses
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '
            # tensorboard logger
            if self.use_tb_logger and 'debug' not in self.exp_name:
                self.tb_logger.add_scalar(k, v, current_iter)

        self.logger.info(message)


# @master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


def get_root_logger(logger_name='base', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: base.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


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
