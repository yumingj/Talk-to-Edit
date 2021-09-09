import argparse
import json
import random

import torch

from .language_utils import *  # noqa
from .lstm import Encoder


def parse_args():
    """Parses arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_vocab_file',
        required=True,
        type=str,
        help='path to the input vocabulary file')
    parser.add_argument(
        '--allow_unknown',
        default=1,
        type=int,
        help='whether allow unknown tokens (i.e. words)')
    parser.add_argument(
        '--pretrained_checkpoint',
        default='',
        type=str,
        help='The pretrained network weights for testing')
    parser.add_argument(
        '--metadata_file',
        default='./templates/metadata_fsm.json',
        type=str,
        help='path to metadata file.')
    parser.add_argument(
        '--system_mode_file',
        default='./templates/system_mode.json',
        type=str,
        help='path to system_mode file.')
    parser.add_argument(
        '--device_name',
        default='gpu',
        type=str,
    )
    parser.add_argument(
        '--verbose',
        default=0,
        type=int,
    )

    # LSTM hyperparameter
    parser.add_argument('--word_embedding_dim', default=300, type=int)
    parser.add_argument('--text_embed_size', default=1024, type=int)
    parser.add_argument('--linear_hidden_size', default=256, type=int)
    parser.add_argument('--linear_dropout_rate', default=0, type=float)

    return parser.parse_args()


def main():

    args = parse_args()
    encode_request(args)


def encode_request(args, system_mode=None, dialog_logger=None):

    # set up
    if args.device_name == 'cpu':
        args.device = torch.device('cpu')
    elif args.device_name == 'gpu':
        args.device = torch.device('cuda')

    if dialog_logger is None:
        output_function = print
    else:
        # output_function = dialog_logger.info
        def output_function(input):
            # suppress output when called by other scripts
            pass
            return

        compulsory_output_function = dialog_logger.info

    # ---------------- STEP 1: Input the Request ----------------

    # choose system_mode
    with open(args.system_mode_file, 'r') as f:
        system_mode_dict = json.load(f)
    system_mode_list = []
    for (mode, mode_idx) in system_mode_dict.items():
        system_mode_list.append(mode)

    if __name__ == '__main__':
        assert system_mode is None
        system_mode = random.choice(system_mode_list)
        output_function('      PREDEFINED system_mode:', system_mode)
    else:
        assert system_mode is not None

    # input request
    if True:
        compulsory_output_function(
            'Enter your request (Press enter when you finish):')
        input_text = input()
    else:
        input_text = 'make the bangs slightly longer.'
    compulsory_output_function('USER INPUT >>> ' + input_text)

    # ---------------- STEP 2: Preprocess Request ----------------

    # output_function("      The system is trying to understand your request:")
    # output_function("      ########################################")

    # load vocabulary
    with open(args.input_vocab_file, 'r') as f:
        vocab = json.load(f)
    text_token_to_idx = vocab['text_token_to_idx']

    text_tokens = tokenize(text=input_text)  # noqa
    text_encoded = encode(  # noqa
        text_tokens=text_tokens,
        token_to_idx=text_token_to_idx,
        allow_unk=args.allow_unknown)
    text_encoded = to_long_tensor([text_encoded]).to(args.device)  # noqa

    # ---------------- STEP 3: Encode Request ----------------

    # prepare encoder
    encoder = Encoder(
        token_to_idx=text_token_to_idx,
        word_embedding_dim=args.word_embedding_dim,
        text_embed_size=args.text_embed_size,
        metadata_file=args.metadata_file,
        linear_hidden_size=args.linear_hidden_size,
        linear_dropout_rate=args.linear_dropout_rate)
    encoder = encoder.to(args.device)
    checkpoint = torch.load(args.pretrained_checkpoint)
    encoder.load_state_dict(checkpoint['state_dict'], True)
    encoder.eval()

    # forward pass
    output = encoder(text_encoded)

    # ---------------- STEP 4: Process Encoder Output ----------------

    output_labels = []
    for head_idx in range(len(output)):
        _, pred = torch.max(output[head_idx], 1)
        head_label = pred.cpu().numpy()[0]
        output_labels.append(head_label)

    # load metadata file
    with open(args.metadata_file, 'r') as f:
        metadata = json.load(f)

    # find mapping from value to label
    reversed_metadata = {}
    for idx, (key, val) in enumerate(metadata.items()):
        reversed_val = reverse_dict(val)  # noqa
        reversed_metadata[key] = reversed_val
    if args.verbose:
        output_function('reversed_metadata:', reversed_metadata)

    # convert predicted values to a dict of predicted labels
    output_semantic_labels = {}  # from LSTM output
    valid_semantic_labels = {}  # useful information among LSTM output
    for idx, (key, val) in enumerate(reversed_metadata.items()):
        output_semantic_labels[key] = val[output_labels[idx]]
        valid_semantic_labels[key] = None
    if args.verbose:
        output_function('output_semantic_labels:', output_semantic_labels)

    # extract predicted labels
    user_mode = output_semantic_labels[system_mode]
    valid_semantic_labels[system_mode] = user_mode

    request_mode = output_semantic_labels['request_mode']
    attribute = output_semantic_labels['attribute']
    score_change_direction = output_semantic_labels['score_change_direction']
    if output_semantic_labels['score_change_value'] is None:
        score_change_value = None
    else:
        score_change_value = int(output_semantic_labels['score_change_value'])
    if output_semantic_labels['target_score'] is None:
        target_score = None
    else:
        target_score = int(output_semantic_labels['target_score'])

    # print to screen
    output_function('      ENCODED user_mode:' + ' ' + user_mode)
    valid_semantic_labels['user_mode'] = user_mode
    if 'pureRequest' in user_mode:
        output_function('      ENCODED request_mode: ' + ' ' + request_mode)
        valid_semantic_labels['request_mode'] = request_mode
        output_function('      ENCODED attribute:' + ' ' + attribute)
        valid_semantic_labels['attribute'] = attribute
        # only output_function labels valid for this request_mode
        if request_mode == 'change_definite':
            output_function('      ENCODED score_change_direction:' + ' ' +
                            (score_change_direction))
            valid_semantic_labels[
                'score_change_direction'] = score_change_direction
            output_function('      ENCODED score_change_value:' + ' ' +
                            str(score_change_value))
            valid_semantic_labels['score_change_value'] = score_change_value
        elif request_mode == 'change_indefinite':
            output_function('      ENCODED score_change_direction:' + ' ' +
                            score_change_direction)
            valid_semantic_labels[
                'score_change_direction'] = score_change_direction
        elif request_mode == 'target':
            output_function('      ENCODED target_score:' + ' ' +
                            str(target_score))
            valid_semantic_labels['target_score'] = target_score

    valid_semantic_labels['text'] = input_text

    if args.verbose:
        output_function('valid_semantic_labels:' + ' ' +
                        str(valid_semantic_labels))
    # output_function("      ########################################")

    return valid_semantic_labels


if __name__ == '__main__':
    main()
