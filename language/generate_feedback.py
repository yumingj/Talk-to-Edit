import argparse
import json
import os.path
import random

import numpy as np

from .language_utils import proper_capitalize


def parse_args():
    """Parses arguments."""

    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--feedback_templates_file',
        default='./templates/feedback.json',
        type=str,
        help='directory to the request templates file')
    parser.add_argument(
        '--pool_file',
        default='./templates/pool.json',
        type=str,
        help='directory to the word pool file')
    parser.add_argument(
        '--num_feedback',
        default=100,
        type=int,
        help='number of feedback data to generate')
    parser.add_argument(
        '--output_file_dir',
        required=True,
        type=str,
        help='folder to save the output request file')
    parser.add_argument(
        '--output_file_name',
        required=True,
        type=str,
        help='name of the output request file')
    parser.add_argument(
        '--whether_enough_general_prob',
        default=0.2,
        type=float,
        help='probability of using general templates in whether_enough mode')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.output_file_dir):
        os.makedirs(args.output_file_dir, exist_ok=True)

    # load template files
    print('loading template files')
    with open(args.feedback_templates_file, 'r') as f:
        args.feedback_templates = json.load(f)
        args.feedback_replacement = args.feedback_templates['replacement']
    with open(args.pool_file, 'r') as f:
        pool = json.load(f)
        args.synonyms_dict = pool["synonyms"]

    system_mode_list = ['whats_next', 'whether_enough', 'suggestion']
    attribute_list = ['Bangs', "Eyeglasses", "No_Beard", "Smiling", "Young"]

    feedback_list = []
    output_txt = []

    # instantiate feedback
    for index in range(args.num_feedback):

        if index % 1000 == 0:
            print('generated', index, '/', args.num_feedback, 'feedback')

        # initialize feedback parameters
        attribute = None

        # randomly choose the feedback parameters
        system_mode = random.choice(system_mode_list)
        if system_mode == 'whether_enough' or system_mode == 'suggestion':
            attribute = random.choice(attribute_list)

        feedback = instantiate_feedback(
            args, system_mode=system_mode, attribute=attribute)

        feedback['index'] = index
        feedback_list.append(feedback)
        output_txt.append(feedback['text'])

    # save feedback dataset

    with open(os.path.join(args.output_file_dir, args.output_file_name),
              'w') as f:
        json.dump(feedback_list, f, indent=4)
    np.savetxt(
        os.path.join(args.output_file_dir, "feedback.txt"),
        output_txt,
        fmt='%s',
        delimiter='\t')

    print('successfully saved.')


def instantiate_feedback(args,
                         system_mode=None,
                         attribute=None,
                         exception_mode='normal'):
    """
    Given the feedback mode (i.e. system_mode) and the attribute (if any),
    return a feedback.
    """

    if exception_mode != 'normal':
        candidate_templates = args.feedback_templates[exception_mode]
        template = random.choice(candidate_templates)
        attribute = attribute
    else:
        # ---------- STEP 1: 1st part of feedback: 'ok' template ----------

        # instantiate the feedback prefix like "ok"
        ok_distribution_prob = random.uniform(0, 1)
        ok_template = ''

        if ok_distribution_prob < 0.7:
            ok_templates = args.feedback_templates['ok']
            for idx, templates in enumerate(ok_templates):
                if 0.3 < ok_distribution_prob < 0.7 and (idx == 0 or idx == 1):
                    continue
                ok_template += random.choice(templates)
            ok_template += ' '
            ok_template = ok_template[0].capitalize() + ok_template[1:]

        # ---------- STEP 2: 2nd part of feedback: content template ----------

        # feedback is trivial like "what's next?"
        if system_mode == 'whats_next':
            candidate_templates = args.feedback_templates['whats_next']
            template = random.choice(candidate_templates)
        # feedback asks whether the editing extent is enough
        elif system_mode == 'whether_enough':
            whether_enough_general_prob = random.uniform(0, 1)
            if whether_enough_general_prob < args.whether_enough_general_prob \
                or args.feedback_templates[
                    'whether_enough'][attribute] == []:
                candidate_templates = args.feedback_templates[
                    'whether_enough']['general']
            else:
                candidate_templates = args.feedback_templates[
                    'whether_enough'][attribute]
            template = random.choice(candidate_templates)
        # feedback provides suggestion on the next edit
        elif system_mode == 'suggestion':
            candidate_templates = args.feedback_templates['suggestion']
            template = random.choice(candidate_templates)
        else:
            raise KeyError('System mode "%s" not recognized' % system_mode)

        # ---------- STEP 3: Postprocess the instantiated template sentence ---------- # noqa

        # replace the <xxx> in the template with
        # proper attribute-specific words.
        # this is not applicable to 'whats_next' type of feedback
        if system_mode != 'whats_next':
            for word in args.feedback_replacement:
                new_word_dict = args.feedback_replacement[word]
                new_word = new_word_dict[attribute]
                template = template.replace(word, new_word)

    # to lower case
    template = template.lower()

    # randomly replace words with synonyms
    for word in args.synonyms_dict:
        replacing_word = random.choice(args.synonyms_dict[word])
        template = template.replace(word, replacing_word)

    # capitalize
    template = proper_capitalize(template)

    if exception_mode != 'normal':
        # after given feedback of cannot_edit
        # encode user request by pretending that
        # the system_mode is 'whats_next'
        system_mode = 'whats_next'
    else:
        template = ok_template + template

    # ---------- STEP 4: Return the feedback and its annotations ----------

    feedback = {
        "text": template,
        "system_mode": system_mode,
        "attribute": attribute
    }
    return feedback


if __name__ == '__main__':
    main()
