import argparse
import json
import os.path
import random
import sys

sys.path.append('.')
from language_utils import proper_capitalize  # noqa


def parse_args():
    """Parses arguments."""

    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--num_request',
        default=100,
        type=int,
        help='number of request data to generate')

    # template files
    parser.add_argument(
        '--user_templates_file',
        type=str,
        default='./templates/user_fsm.json',
        help='directory to the request templates file')
    parser.add_argument(
        '--pool_file',
        type=str,
        default='./templates/pool.json',
        help='directory to the word pool file')
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
    # output
    parser.add_argument(
        '--output_file_dir',
        required=True,
        type=str,
        help='folder to save the output request file')

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.output_file_dir):
        os.makedirs(args.output_file_dir, exist_ok=False)

    # load template files
    print('loading template files')
    with open(args.user_templates_file, 'r') as f:
        args.user_templates = json.load(f)
    with open(args.pool_file, 'r') as f:
        pool = json.load(f)
        args.synonyms_dict = pool["synonyms"]
        args.postfix_list = pool["postfix"]
    with open(args.metadata_file, 'r') as f:
        args.metadata = json.load(f)
    with open(args.system_mode_file, 'r') as f:
        args.system_mode_dict = json.load(f)
    args.system_mode_list = []
    for key, value in args.system_mode_dict.items():
        args.system_mode_list.append(key)

    attribute_list = ['Bangs', "Eyeglasses", "No_Beard", "Smiling", "Young"]
    target_score_list = [0, 1, 2, 3, 4, 5]
    score_change_direction_list = ['positive', 'negative']
    score_change_value_list = [1, 2, 3, 4, 5]

    request_list = []

    # instantiate requests
    for index in range(args.num_request):

        if index % 1000 == 0:
            print('generated', index, '/', args.num_request, 'requests')

        # randomly choose the semantic editing parameters
        system_mode = random.choice(args.system_mode_list)
        user_mode_list = list(args.metadata[system_mode].keys())
        user_mode = random.choice(user_mode_list)
        attribute = random.choice(attribute_list)
        score_change_value = random.choice(score_change_value_list)
        score_change_direction = random.choice(score_change_direction_list)
        target_score = random.choice(target_score_list)

        # instantiate a request according to the
        # chosen semantic editing parameters
        request = instantiate_training_request(
            args,
            attribute=attribute,
            user_mode=user_mode,
            score_change_direction=score_change_direction,
            score_change_value=score_change_value,
            target_score=target_score)

        request['system_mode'] = system_mode

        # assign each system_mode's user_mode
        for mode in args.system_mode_list:
            if system_mode == mode:
                request[mode] = request['user_mode']
            else:
                request[mode] = None

        request['index'] = index
        request_list.append(request)

    # save request dataset
    if not os.path.isdir(args.output_file_dir):
        os.makedirs(args.output_file_dir, exist_ok=True)
    with open(
            os.path.join(args.output_file_dir, 'training_request.json'),
            'w') as f:
        json.dump(request_list, f, indent=4)

    print('successfully saved.')


def instantiate_training_request(
    args,
    attribute=None,
    user_mode=None,
    score_change_direction=None,
    score_change_value=None,
    target_score=None,
):
    """
    Given semantic editing parameters, instantiate the request
    using the request templates.
    """

    request_mode = None

    instantiated_sentence = ''
    user_sub_mode_list = user_mode.split('_')

    for user_sub_mode_idx, user_sub_mode in enumerate(user_sub_mode_list):

        sub_mode_template = ''
        if user_sub_mode != 'pureRequest':
            sub_mode_templates = args.user_templates[user_sub_mode]
            for templates in sub_mode_templates:
                sub_mode_template += random.choice(templates)
        else:
            request_mode = random.choice(
                ['target', 'change_definite', 'change_indefinite'])

            request_templates = args.user_templates['pureRequest']
            attribute_templates = request_templates[attribute]

            # request is the score change direction and value
            if request_mode == 'change_definite':
                assert score_change_direction is not None
                assert score_change_value is not None
                target_score = None
                candidate_templates = attribute_templates['change'][
                    score_change_direction]['definite'][str(
                        score_change_value)]
            # request is the score change direction without value
            elif request_mode == 'change_indefinite':
                assert score_change_direction is not None
                score_change_value = None
                target_score = None
                candidate_templates = attribute_templates['change'][
                    score_change_direction]['indefinite']
            # request is the edit target
            elif request_mode == 'target':
                score_change_direction = None
                score_change_value = None
                assert target_score is not None
                candidate_templates = attribute_templates['target'][str(
                    target_score)]
            else:
                raise KeyError('Request mode "%s" not recognized' %
                               request_mode)

            # randomly choose one request template
            sub_mode_template = random.choice(candidate_templates)

        if user_sub_mode_idx >= 1:
            instantiated_sentence += ' '
        instantiated_sentence += sub_mode_template

    if 'pureRequest' not in user_sub_mode_list:
        score_change_direction = None
        score_change_value = None
        target_score = None
        attribute = None

    # to lower case
    instantiated_sentence = instantiated_sentence.lower()

    # randomly replace words with synonyms
    for word in args.synonyms_dict:
        new_word = random.choice(args.synonyms_dict[word])
        instantiated_sentence = instantiated_sentence.replace(word, new_word)

    # capitalize
    instantiated_sentence = proper_capitalize(instantiated_sentence)

    request = {
        "text": instantiated_sentence,
        "user_mode": user_mode,
        'request_mode': request_mode,
        "attribute": attribute,
        "score_change_direction": score_change_direction,
        "score_change_value": score_change_value,
        "target_score": target_score,
    }

    return request


if __name__ == '__main__':
    main()
