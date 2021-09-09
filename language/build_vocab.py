import argparse
import json
import os
import sys

sys.path.append('.')
from language_utils import *  # noqa
"""
Build vocabulary from all instantiated templates
"""


def parse_args():
    """Parses arguments."""

    parser = argparse.ArgumentParser(description='Build vocabulary')
    parser.add_argument(
        '--input_data_path',
        required=True,
        type=str,
        help='path to the input data file')
    parser.add_argument(
        '--output_dir',
        required=True,
        type=str,
        help='folder to save the output vocabulary file')

    return parser.parse_args()


def main():
    args = parse_args()

    # prepare output directory
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # load text data
    print("Loading text data from", args.input_data_path)
    with open(args.input_data_path, 'r') as f:
        input_data = json.load(f)

    # gather a list of text
    print("Building vocabulary from", len(input_data), "text data samples")
    text_list = []
    for idx, data_sample in enumerate(input_data):
        if idx % 10000 == 0:
            print('loaded', idx, '/', len(input_data))
        text = data_sample['text']
        text_list.append(text)

    # build vocabulary
    text_token_to_idx = build_vocab(text_list=text_list)  # noqa
    vocab = {
        'text_token_to_idx': text_token_to_idx,
    }

    # save vocabulary
    print("Saving vocabulary file to",
          os.path.join(args.output_dir, 'vocab.json'))
    with open(os.path.join(args.output_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=4)


if __name__ == '__main__':
    main()
