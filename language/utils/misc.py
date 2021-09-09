'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os

import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'get_mean_and_std', 'init_params', 'mkdir_p', 'save_checkpoint',
    'AverageMeter'
]


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(  # noqa
        dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(args,
                    state,
                    is_best,
                    checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    epoch = str(state['epoch']).zfill(2)
    save_every_epoch = True
    if not os.path.exists(os.path.join(args.work_dir, 'checkpoints')):
        os.makedirs(os.path.join(args.work_dir, 'checkpoints'))
    if save_every_epoch:
        filename = 'checkpoint_' + epoch + '.pth.tar'
        filepath = os.path.join(checkpoint, 'checkpoints', filename)
        torch.save(state, filepath)
    if is_best:
        filename = 'model_best.pth.tar'
        filepath = os.path.join(checkpoint, 'checkpoints', filename)
        torch.save(state, filepath)
        # shutil.copyfile(filepath, os.path.join(checkpoint, \
        # 'model_best_'+epoch+'.pth.tar'))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0  # running average = running sum / running count
        self.sum = 0  # running sum
        self.count = 0  # running count

    def update(self, val, n=1):
        # n = batch_size

        # val = batch accuracy for an attribute
        # self.val = val

        # sum = 100 * accumulative correct predictions for this attribute
        self.sum += val * n

        # count = total samples so far
        self.count += n

        # avg = 100 * avg accuracy for this attribute
        # for all the batches so far
        self.avg = self.sum / self.count
