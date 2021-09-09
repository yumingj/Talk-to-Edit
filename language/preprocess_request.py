import argparse
import json
import os
import sys

import numpy as np

sys.path.append('.')
from language_utils import *  # noqa
"""
Preprocess the text
"""


def parse_args():
    """Parses arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_vocab_path',
        required=True,
        type=str,
        help='path to the input vocabulary file')
    parser.add_argument(
        '--input_data_path',
        required=True,
        type=str,
        help='path to the input data file')
    parser.add_argument(
        '--metadata_file',
        type=str,
        default='./templates/metadata_fsm.json',
        help='directory to the metadata file')
    parser.add_argument(
        '--system_mode_file',
        type=str,
        default='./templates/system_mode.json',
        help='directory to the system_mode file')
    parser.add_argument(
        '--allow_unknown',
        default=0,
        type=int,
        help='whether allow unknown tokens (i.e. words)')
    parser.add_argument(
        '--expand_vocab',
        default=0,
        type=int,
        help='whether expand vocabularies')
    parser.add_argument(
        '--output_dir',
        required=True,
        type=str,
        help='folder to save the output vocabulary file')
    parser.add_argument(
        '--unlabeled_value',
        default=999,
        type=int,
        help='value to represent unlabeled value')

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=False)

    # load vocabulary
    print("Loading vocab")
    with open(args.input_vocab_path, 'r') as f:
        vocab = json.load(f)
    text_token_to_idx = vocab['text_token_to_idx']

    # load metadata file
    with open(args.metadata_file, 'r') as f:
        metadata = json.load(f)

    # load system_mode file
    with open(args.system_mode_file, 'r') as f:
        system_mode_file = json.load(f)

    # load input data
    with open(args.input_data_path, 'r') as f:
        input_data = json.load(f)

    # initialize lists to store encoded data
    text_encoded_list = []
    system_mode_encoded_list = []
    labels_encoded_list = []

    print('Encoding')

    for idx, data_sample in enumerate(input_data):

        # encode text
        text = data_sample['text']
        text_tokens = tokenize(text=text)  # noqa
        text_encoded = encode(  # noqa
            text_tokens=text_tokens,
            token_to_idx=text_token_to_idx,
            allow_unk=args.allow_unknown)
        text_encoded_list.append(text_encoded)

        # encode system_mode
        system_mode = data_sample['system_mode']
        system_mode_encoded = system_mode_file[system_mode]
        system_mode_encoded_list.append(system_mode_encoded)

        # encode labels
        labels_encoded = []
        for idx, (key, val) in enumerate(metadata.items()):
            label = data_sample[key]
            if label is None:
                # use args.unlabeled_value to represent missing labels
                label_encoded = args.unlabeled_value
            else:
                label_encoded = val[str(label)]
            labels_encoded.append(label_encoded)
        labels_encoded_list.append(labels_encoded)

    # Pad encoded text to equal length
    print('Padding tokens')
    text_encoded_padded_list = []
    max_text_length = max(len(text) for text in text_encoded_list)
    for text_encoded in text_encoded_list:
        while len(text_encoded) < max_text_length:
            text_encoded.append(text_token_to_idx['<NULL>'])
        text_encoded_padded_list.append(text_encoded)

    # save processed text
    np.save(
        os.path.join(args.output_dir, 'text.npy'), text_encoded_padded_list)
    np.savetxt(
        os.path.join(args.output_dir, 'text.txt'),
        text_encoded_padded_list,
        fmt='%.0f')

    # save processed system_mode
    np.save(
        os.path.join(args.output_dir, 'system_mode.npy'),
        system_mode_encoded_list)
    np.savetxt(
        os.path.join(args.output_dir, 'system_mode.txt'),
        system_mode_encoded_list,
        fmt='%.0f')

    # save processed labels
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels_encoded_list)
    np.savetxt(
        os.path.join(args.output_dir, 'labels.txt'),
        labels_encoded_list,
        fmt='%.0f')


if __name__ == '__main__':
    main()
