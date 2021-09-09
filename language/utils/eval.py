from __future__ import absolute_import, print_function

import torch

__all__ = ['classification_accuracy', 'regression_accuracy']


def classification_accuracy(output,
                            target,
                            class_wise=False,
                            num_cls=6,
                            excluded_cls_idx=None):
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

        acc = float(torch.sum(pred == target)) / float(batch_size) * 100
        return_dict = {'acc': acc}

        if excluded_cls_idx is not None:
            correct_count = torch.sum(
                (pred == target) * (target != excluded_cls_idx))
            labeled_count = torch.sum(target != excluded_cls_idx)
            if labeled_count:
                labeled_acc = float(correct_count) / float(labeled_count) * 100
            else:
                labeled_acc = 0

            return_dict['labeled_acc'] = labeled_acc
            return_dict['labeled_count'] = labeled_count
        else:
            return_dict['labeled_acc'] = acc
            return_dict['labeled_count'] = batch_size

        if class_wise:
            acc_class_wise = []
            per_class_count = []
            # actual number of classes <= num_cls=6
            for i in range(num_cls):
                total_sample_cls_i = torch.sum(target == i)
                if total_sample_cls_i:
                    correct_samples_cls_i = torch.sum(
                        (pred == i) * (target == i))
                    acc_class_wise.append(
                        float(correct_samples_cls_i) /
                        float(total_sample_cls_i) * 100)
                else:
                    acc_class_wise.append(0)
                per_class_count.append(total_sample_cls_i)

        return_dict['acc_class_wise'] = acc_class_wise
        return_dict['per_class_count'] = per_class_count

        return return_dict


def regression_accuracy(output,
                        target,
                        margin=0.2,
                        uni_neg=True,
                        class_wise=False,
                        num_cls=6,
                        excluded_cls_idx=None,
                        max_cls_value=5):
    """
    Computes the regression accuracy

    if predicted score is less than one margin from the ground-truth score, we
    consider it as correct otherwise it is incorrect， the acc is the
    percentage of correct regression

    class_wise: if True, then report overall accuracy and class-wise accuracy
                else, then only report overall accuracy
    """

    output = output.clone().reshape(-1)

    if uni_neg:
        output[(output <= 0 + margin) * (target == 0)] = 0
        output[(output >= max_cls_value - margin) *
               (target == max_cls_value)] = max_cls_value

    distance = torch.absolute(target - output)
    distance = distance - margin

    predicted_class = torch.zeros_like(target)
    # if distance <= 0, assign ground truth class
    predicted_class[distance <= 0] = target[distance <= 0]
    # if distance > 0, assign an invalid value
    predicted_class[distance > 0] = -1

    acc = float(torch.sum(predicted_class == target)) / float(
        target.size(0)) * 100

    return_dict = {'acc': acc}

    if excluded_cls_idx is not None:
        correct_count = torch.sum(
            (predicted_class == target) * (target != excluded_cls_idx))
        labeled_count = torch.sum(target != excluded_cls_idx)
        if labeled_count:
            labeled_acc = float(correct_count) / float(labeled_count) * 100
        else:
            labeled_acc = 0
        return_dict['labeled_acc'] = labeled_acc
        return_dict['labeled_count'] = labeled_count
    else:
        labeled_acc = acc
        return_dict['labeled_acc'] = acc
        return_dict['labeled_count'] = target.size(0)

    if class_wise:
        acc_class_wise = []
        per_class_count = []
        for i in range(num_cls):
            total_sample_cls_i = torch.sum(target == i)
            if total_sample_cls_i:
                correct_samples_cls_i = torch.sum(
                    (predicted_class == i) * (target == i))
                acc_class_wise.append(
                    float(correct_samples_cls_i) / float(total_sample_cls_i) *
                    100)
            else:
                acc_class_wise.append(0)
            per_class_count.append(total_sample_cls_i)

        return_dict['acc_class_wise'] = acc_class_wise
        return_dict['per_class_count'] = per_class_count

    return return_dict


def main():

    l1 = [
        0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 2, 2, 2, 1.7, 0, 3, 3, 2.79, 3.3, 0, 4,
        2, 5, 3, 0, 6, 6, 4.78, 6, 0
    ]
    l2 = [
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4,
        4, 5, 5, 5, 5, 5
    ]

    output = torch.FloatTensor(l1)
    target = torch.LongTensor(l2)
    acc = regression_accuracy(output, target, margin=0.2)
    print('acc:', acc)
    print()
    acc, acc_class_wise_list, per_class_count = regression_accuracy(
        output, target, margin=0.2, class_wise=True)
    print('acc:', acc)
    print('acc_class_wise_list:', acc_class_wise_list)
    print('per_class_count: ', per_class_count)


if __name__ == '__main__':
    main()
