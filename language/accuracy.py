import torch


def head_accuracy(output, target, unlabeled_value=999):
    """
    Computes the precision@k for the specified values of k
    output: batch_size * num_cls (for a specific attribute)
    target: batch_size * 1 (for a specific attribute)
    return res: res = 100 * num_correct / batch_size, for a specific attribute
    for a batch
    """

    with torch.no_grad():
        batch_size = target.size(0)

        # _ = the largest score, pred = cls_idx with the largest score
        _, pred = output.topk(1, 1, True, True)
        pred = pred.reshape(-1)

        # acc = float(torch.sum(pred == target)) / float(batch_size) * 100
        return_dict = {}

        if unlabeled_value is not None:

            correct_count = torch.sum(
                (target != unlabeled_value) * (pred == target))
            labeled_count = torch.sum(target != unlabeled_value)

            if labeled_count:

                labeled_acc = float(correct_count) / float(labeled_count) * 100
            else:
                labeled_acc = 0

            return_dict['acc'] = labeled_acc
            return_dict['labeled_count'] = labeled_count
        else:

            return_dict['acc'] = acc  # noqa
            return_dict['labeled_count'] = batch_size

        return return_dict
