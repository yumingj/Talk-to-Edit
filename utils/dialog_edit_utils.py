import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from language.generate_feedback import instantiate_feedback
from language.run_encoder import encode_request
from models.utils import save_image

from utils.editing_utils import edit_target_attribute


def dialog_with_real_user(field_model,
                          latent_code,
                          opt,
                          args,
                          dialog_logger,
                          display_img=False):

    # initialize dialog recorder
    state_log = ['start']
    edit_log = []
    system_log = [{"text": None, "system_mode": 'start', "attribute": None}]
    user_log = []
    not_used_attribute = [
        'Bangs', "Eyeglasses", "No_Beard", "Smiling", "Young"
    ]
    text_log = []
    text_image_log = []

    # initialize first round's variables
    round_idx = 0

    edited_latent_code = None

    with torch.no_grad():
        start_image, start_label, start_score = \
            field_model.synthesize_and_predict(torch.from_numpy(latent_code).to(torch.device('cuda'))) # noqa

    save_image(start_image, f'{opt["path"]["visualization"]}/start_image.png')

    if display_img:
        plt.figure()
        plt.imshow(
            mpimg.imread(f'{opt["path"]["visualization"]}/start_image.png'))
        plt.axis('off')
        plt.show()

    # initialize attribtue_dict
    attribute_dict = {
        "Bangs": start_label[0],
        "Eyeglasses": start_label[1],
        "No_Beard": start_label[2],
        "Smiling": start_label[3],
        "Young": start_label[4],
    }
    dialog_logger.info('START IMAGE  >>> ' + str(attribute_dict))

    while True:

        dialog_logger.info('\n---------------------------------------- Edit ' +
                           str(round_idx) +
                           '----------------------------------------\n')

        # -------------------- TAKE USER INPUT --------------------
        # understand user input
        user_labels = encode_request(
            args,
            system_mode=system_log[-1]['system_mode'],
            dialog_logger=dialog_logger)
        text_image_log.append('USER:   ' + user_labels['text'])

        # update not_used_attribute
        if user_labels['attribute'] in not_used_attribute:
            not_used_attribute.remove(user_labels['attribute'])

        # #################### DECIDE STATE ####################
        state = decide_next_state(
            state=state_log[-1],
            system_mode=system_log[-1]['system_mode'],
            user_mode=user_labels['user_mode'])

        if state == 'end':
            user_log.append(user_labels)
            state_log.append(state)
            text_log.append('USER:   ' + user_labels['text'])
            break

        # #################### DECIDE EDIT ####################
        edit_labels = decide_next_edit(
            edit_log=edit_log,
            system_labels=system_log[-1],
            user_labels=user_labels,
            state=state,
            attribute_dict=attribute_dict,
            dialog_logger=dialog_logger)

        text_image_log.append(edit_labels)

        attribute_dict, exception_mode, latent_code, edited_latent_code = edit_target_attribute(  # noqa
            opt,
            attribute_dict,
            edit_labels,
            round_idx,
            latent_code,
            edited_latent_code,
            field_model,
            display_img=display_img)
        if state == 'no_edit':
            dialog_logger.info('NO EDIT  >>> ' + str(attribute_dict))
        else:
            dialog_logger.info('UPDATED IMAGE >>> ' + str(attribute_dict))
        text_image_log.append(attribute_dict.copy())

        # #################### DECIDE SYSTEM ####################
        # decide system feedback hard labels
        temp_system_labels = decide_next_feedback(
            system_labels=system_log[-1],
            user_labels=user_labels,
            state=state,
            edit_labels=edit_labels,
            not_used_attribute=not_used_attribute,
            round_idx=round_idx,
            exception_mode=exception_mode)

        # instantiate feedback
        system_labels = instantiate_feedback(
            args,
            system_mode=temp_system_labels['system_mode'],
            attribute=temp_system_labels['attribute'],
            exception_mode=exception_mode)

        dialog_logger.info('SYSTEM FEEDBACK >>> ' + system_labels['text'])

        # update not_used_attribute
        if system_labels['attribute'] in not_used_attribute:
            not_used_attribute.remove(system_labels['attribute'])

        # -------------------- UPDATE LOG --------------------
        state_log.append(state)
        edit_log.append(edit_labels)
        system_log.append(system_labels)
        user_log.append(user_labels)
        text_log.append('USER:   ' + user_labels['text'])
        text_log.append('SYSTEM: ' + system_labels['text'])
        text_log.append('')
        text_image_log.append('SYSTEM: ' + system_labels['text'])
        text_image_log.append('')

        round_idx += 1

    dialog_overall_log = {
        'state_log': state_log,
        'edit_log': edit_log,
        'system_log': system_log,
        'user_log': user_log,
        'text_log': text_log,
        'text_image_log': text_image_log
    }
    dialog_logger.info('Dialog successfully ended.')

    return dialog_overall_log


def decide_next_state(state, system_mode, user_mode):
    """
    Input: state, system, user
    Output: next state
    """

    if state == 'start':
        assert system_mode == 'start'
        assert user_mode == 'start_pureRequest'
        next_state = 'edit'

    elif state == 'edit':
        if system_mode == 'suggestion':
            if user_mode == 'yes':
                next_state = 'edit'
            elif user_mode == 'yes_pureRequest':
                next_state = 'edit'
            elif user_mode == 'no_pureRequest':
                next_state = 'edit'
            elif user_mode == 'no':
                next_state = 'no_edit'
            elif user_mode == 'no_end':
                next_state = 'end'
            else:
                raise ValueError("invalid user_mode")
        elif system_mode == 'whether_enough':
            if user_mode == 'yes':
                next_state = 'no_edit'
            elif user_mode == 'yes_pureRequest':
                next_state = 'edit'
            elif user_mode == 'yes_end':
                next_state = 'end'
            elif user_mode == 'no':
                next_state = 'edit'
            elif user_mode == 'no_pureRequest':
                next_state = 'edit'
            else:
                raise ValueError("invalid user_mode")
        elif system_mode == 'whats_next':
            if user_mode == 'pureRequest':
                next_state = 'edit'
            elif user_mode == 'end':
                next_state = 'end'
        else:
            raise ValueError("invalid system_mode")

    elif state == 'no_edit':
        if system_mode == 'suggestion':
            if user_mode == 'yes':
                next_state = 'edit'
            elif user_mode == 'yes_pureRequest':
                next_state = 'edit'
            elif user_mode == 'no_pureRequest':
                next_state = 'edit'
            elif user_mode == 'no':
                next_state = 'no_edit'
            elif user_mode == 'no_end':
                next_state = 'end'
            else:
                raise ValueError("invalid user_mode")
        elif system_mode == 'whether_enough':
            raise ValueError("invalid system_mode")
        elif system_mode == 'whats_next':
            if user_mode == 'pureRequest':
                next_state = 'edit'
            elif user_mode == 'end':
                next_state = 'end'
        else:
            raise ValueError("invalid system_mode")
    elif state == 'end':
        raise ValueError("invalid state")

    else:
        raise ValueError("invalid state")

    return next_state


def decide_next_edit(edit_log, system_labels, user_labels, state,
                     attribute_dict, dialog_logger):
    """
    Input: previous edit, system, user, resulting state, attribute labels
    Output: current edit
    """

    attribute = None
    score_change_direction = None
    score_change_value = None
    target_score = None

    if len(edit_log) > 0:
        edit_labels = edit_log[-1]

    # ---------- decide edit_labels ----------
    if len(edit_log) == 0:
        # now is the first round, so edit according to user request
        assert 'pureRequest' in user_labels['user_mode']
        assert state == 'edit'
        attribute = user_labels['attribute']
        score_change_direction = user_labels['score_change_direction']
        score_change_value = user_labels['score_change_value']
        target_score = user_labels['target_score']
        if user_labels['request_mode'] == 'change_indefinite':
            assert score_change_value is None
            score_change_value = 1

    elif 'pureRequest' in user_labels['user_mode']:
        # edit according to user request
        assert state == 'edit'
        attribute = user_labels['attribute']
        score_change_direction = user_labels['score_change_direction']
        score_change_value = user_labels['score_change_value']
        target_score = user_labels['target_score']
        if user_labels['request_mode'] == 'change_indefinite':
            assert score_change_value is None
            score_change_value = 1

    elif system_labels['system_mode'] == 'whether_enough' and user_labels[
            'user_mode'] == 'no':
        # continue the previous edit
        assert state == 'edit'
        attribute = edit_labels['attribute']
        score_change_direction = edit_labels['score_change_direction']
        score_change_value = 1
        target_score = None

    elif system_labels['system_mode'] == 'suggestion' and user_labels[
            'user_mode'] == 'yes':
        # play with the suggested attribute, random direction
        # (small degree --> positive direction)
        assert state == 'edit'
        attribute = system_labels['attribute']

        if attribute_dict[attribute] <= 2:
            score_change_direction = 'positive'
        else:
            score_change_direction = 'negative'

        score_change_value = 1
        target_score = None

    else:
        # no edit
        assert (state == 'no_edit' or state == 'end')
        attribute = None
        score_change_direction = None
        score_change_value = None
        target_score = None

    # --- The code below is moderation mechanism for language encoder ---
    if system_labels['system_mode'] == 'suggestion' and user_labels[
            'user_mode'] == 'yes':
        attribute = system_labels['attribute']

    # ---------- Fill in all the values in edit_labels ----------
    if attribute is None:
        assert score_change_direction is None
        assert score_change_value is None
        assert target_score is None
    elif target_score is not None:
        assert score_change_direction is None
        assert score_change_value is None
        if target_score > attribute_dict[attribute]:
            score_change_direction = 'positive'
        elif target_score < attribute_dict[attribute]:
            score_change_direction = 'negative'
        else:
            pass
        score_change_value = abs(target_score - attribute_dict[attribute])
    elif score_change_direction is not None:
        assert score_change_value is not None
        if score_change_direction == 'positive':
            target_score = attribute_dict[attribute] + score_change_value
        elif score_change_direction == 'negative':
            target_score = attribute_dict[attribute] - score_change_value
        else:
            raise ValueError('invalid direction')
        # boundary value checking
        if target_score > 5:
            target_score = 5
            score_change_value = abs(target_score - attribute_dict[attribute])
        elif target_score < 0:
            target_score = 0
            score_change_value = abs(target_score - attribute_dict[attribute])

    next_edit_labels = {
        'attribute': attribute,
        'score_change_direction': score_change_direction,
        "score_change_value": score_change_value,
        'target_score': target_score
    }

    return next_edit_labels


def decide_next_feedback(system_labels, user_labels, state, edit_labels,
                         not_used_attribute, round_idx, exception_mode):
    """
    Input: system, user, state, edit + others
    Output: system
    """

    assert (state == 'edit' or state == 'no_edit')

    while True:

        if exception_mode != 'normal':
            system_mode = 'whats_next'
            feedback_attribute = None
            break

        system_mode = None
        feedback_attribute = None

        if system_labels['system_mode'] == 'suggestion' and user_labels[
                'user_mode'] == 'yes':
            assert state == 'edit'
            system_mode = 'whether_enough'
            feedback_attribute = system_labels['attribute']
            break

        # ---------- whether_enough ----------
        # first round has higher chance for whether_enough
        whether_enough_random_num = random.uniform(0, 1)
        if round_idx == 0:
            whether_enough_prob = 0.8
            if whether_enough_random_num < whether_enough_prob:
                system_mode = 'whether_enough'
                if state == 'no_edit':
                    continue
                else:
                    feedback_attribute = user_labels['attribute']
                    assert feedback_attribute is not None

        # ---------- whats_next ----------
        # higher chance at earlier rounds
        if system_mode is None:
            whats_next_random_num = random.uniform(0, 1)
            whats_next_prob_list = [0.5, 0.4, 0.3, 0.3]
            if round_idx <= 3:
                whats_next_prob = whats_next_prob_list[round_idx]
            else:
                whats_next_prob = 0.2
            if whats_next_random_num < whats_next_prob:
                system_mode = 'whats_next'
                feedback_attribute = None

        # ---------- suggestion ----------
        # if a lot of attribute has been edited, don't be suggestion
        if system_mode is None:
            suggestion_random_num = random.uniform(0, 1)
            suggestion_prob = len(not_used_attribute) * 0.2
            if suggestion_random_num < suggestion_prob:
                system_mode = 'suggestion'
                if len(not_used_attribute) > 0:
                    feedback_attribute = random.choice(not_used_attribute)
                else:
                    system_mode = None

        # ---------- whether_enough ----------
        # if not chosen to be 'whats_next' or 'suggestion',
        # then use 'whether_enough'
        if system_mode is None:
            system_mode = 'whether_enough'
            if state == 'no_edit':
                continue
            else:
                feedback_attribute = edit_labels['attribute']
                assert feedback_attribute is not None

        # if state is no_edit, system_mode cannot be whether_enough
        if not (state == 'no_edit' and system_mode == 'whether_enough'):
            break

    next_system_labels = {
        'exception_mode': exception_mode,
        'system_mode': system_mode,
        'attribute': feedback_attribute
    }

    return next_system_labels
