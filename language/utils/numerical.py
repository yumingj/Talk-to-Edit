import json

import numpy as np

__all__ = ['get_weight', 'transpose_and_format']


def get_weight(args):
    """
    read the attribute class distribution file stats.txt and return the counts
    """

    # read counts from stats file
    stats_f = open(args.stats_file, "r")

    # each list [] in the count_list is for one attribute
    # each value in [] is the number of training samples
    # for that attribute value
    count_list = []
    for i in range(args.num_attr):
        count_list.append([])
    for row_idx, row in enumerate(stats_f):
        # row 0 is attr names, row 1 is unlabeled statistics
        if row_idx == 0 or row_idx == 1:
            continue
        # [:-1] because the last value is the new line character
        row = row.split(' ')[:-1]
        for new_idx_in_row, attr_val in enumerate(row):
            # print('num_idx:', num_idx, 'num:', num)
            if new_idx_in_row == 0:
                continue
            new_idx = new_idx_in_row - 1
            count_list[new_idx].append((int(attr_val)))  # **0.5)

    # weight for gt_remapping case
    count_list = np.array(count_list)
    num_attr = count_list.shape[0]
    num_cls = count_list.shape[1]

    if args.gt_remapping:
        remap_count_list = np.zeros((num_attr, num_cls))
        for attr_idx in range(num_attr):
            for cls_idx in range(num_cls):
                new_cls_idx = int(args.gt_remapping[attr_idx][cls_idx])
                remap_count_list[attr_idx][new_cls_idx] += count_list[
                    attr_idx][cls_idx]
        count_list = remap_count_list

    # For each attribute, among classes, weight Inversion and Normalization
    value_weights = []
    for attr_idx in range(num_attr):
        weight_l = np.zeros(num_cls)
        for cls_idx in range(num_cls):
            weight_l[cls_idx] = (1 / count_list[attr_idx][cls_idx]
                                 ) if count_list[attr_idx][cls_idx] else 0

        # normalize weight_l so that their average value is 1
        normalized_weight_l = np.zeros(num_cls)
        for cls_idx in range(num_cls):
            normalized_weight_l[cls_idx] = weight_l[cls_idx] / sum(weight_l)
        value_weights.append(normalized_weight_l)

    # Among attributes, weight Inversion and Normalization
    # count_sum_list = []
    # for a_list in count_list:
    #     count_sum_list.append(sum(a_list))
    # count_sum = sum(count_sum_list)
    # attribute_weights = []
    # for i in range(len(count_sum_list)):
    #     attribute_weight = count_sum / count_sum_list[i]
    #     attribute_weights.append(attribute_weight)
    # # normalize attribute_weights so that their average value is 1
    # normalized_attribute_weights = []
    # for i in range(len(attribute_weights)):
    #     normalized_attribute_weights.append(attribute_weights[i] /
    #                                         sum(attribute_weights) *
    #                                         len(attribute_weights))

    weights = {'value_weights': value_weights}

    return weights


def transpose_and_format(args, input):
    """
    input = [
        [#, #, #, #, #, #],
        [#, #, #, #, #, #],
        [#, #, #, #, #, #]
    ]
    where outer loop is attribute
    inner loop is class labels

    new_f:

    attr_val Bangs Smiling Young
    0 # # #
    1 # # #
    2 # # #
    3 # # #
    4 # # #
    5 # # #
    """

    with open(args.attr_file, 'r') as f:
        attr_f = json.load(f)
    attr_info = attr_f['attr_info']
    attr_list = ['attr_val']
    for key, val in attr_info.items():
        attr_list.append(val["name"])

    # new_f stores the output
    new_f = []

    # first line is the header
    new_f.append(attr_list)
    for i in range(len(input[0])):
        row = []
        row.append(i)
        for j in range(args.num_attr):
            row.append(round(input[j][i].item(), 2))
            # row.append(round(input[j][i], 2))
        new_f.append(row)
    return new_f
