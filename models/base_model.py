import logging
import math
from collections import OrderedDict

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models.archs.attribute_predictor_arch import resnet50
from models.archs.field_function_arch import FieldFunction
from models.archs.stylegan2.model import Generator
from models.losses.arcface_loss import ArcFaceLoss
from models.losses.discriminator_loss import DiscriminatorLoss
from models.utils import (postprocess, predictor_to_label, save_image,
                          transform_image)

logger = logging.getLogger('base')


class BaseModel():
    """Base model.
    """

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
        self.is_train = opt['is_train']
        self.target_attr_idx = opt['attr_dict'][opt['attribute']]

        # define stylegan generator
        self.stylegan_gen = Generator(
            size=opt['img_res'],
            style_dim=opt['latent_dim'],
            n_mlp=opt['n_mlp'],
            channel_multiplier=opt['channel_multiplier']).to(self.device)

        self.truncation = 1.0
        self.truncation_latent = None
        self.randomize_noise = False
        if opt['latent_space'] == 'z':
            self.input_is_latent = False
            self.latent_code_is_w_space = False
        else:
            self.input_is_latent = True
            self.latent_code_is_w_space = True

        self.transform_z_to_w = opt.get('transform_z_to_w', True)

        if opt['img_res'] == 128:
            self.w_space_channel_num = 12
            logger.info(
                f'Loading stylegan model from: {opt["generator_ckpt"]}')
            checkpoint = torch.load(opt['generator_ckpt'])
            self.stylegan_gen.load_state_dict(checkpoint["g_ema"], strict=True)
            self.img_resize = False
        elif opt['img_res'] == 1024:
            self.w_space_channel_num = 18
            logger.info(
                f'Loading stylegan model from: {opt["generator_ckpt"]}')
            checkpoint = torch.load(opt['generator_ckpt'])
            self.stylegan_gen.load_state_dict(checkpoint, strict=True)
            self.img_resize = True

        # define attribute predictor
        self.predictor = resnet50(attr_file=opt['attr_file'])
        self.predictor = self.predictor.to(self.device)

        logger.info(f'Loading model from: {opt["predictor_ckpt"]}')
        checkpoint = torch.load(opt['predictor_ckpt'])
        self.predictor.load_state_dict(checkpoint['state_dict'], strict=True)
        self.predictor.eval()

        # define field function
        self.field_function = FieldFunction(
            num_layer=opt['num_layer'],
            latent_dim=512,
            hidden_dim=opt['hidden_dim'],
            leaky_relu_neg_slope=opt['leaky_relu_neg_slope'])

        self.field_function = self.field_function.to(self.device)

        self.fix_layers = False
        if self.is_train:
            self.init_training_settings()
            self.log_dict = OrderedDict()

    def init_training_settings(self):
        # set up optimizers
        self.optimizer = torch.optim.Adam(
            self.field_function.parameters(),
            self.opt['lr'],
            weight_decay=self.opt['weight_decay'])

        # define loss functions
        # predictor loss
        self.criterion_predictor = nn.CrossEntropyLoss(reduction='mean')

        # arcface loss
        if self.opt['arcface_weight'] > 0:
            self.criterion_arcface = ArcFaceLoss(
                self.opt['pretrained_arcface'], self.opt['arcface_loss_type'])
        else:
            self.criterion_arcface = None

        # discriminator loss
        if self.opt['arcface_weight'] > 0:
            self.criterion_disc = DiscriminatorLoss(
                self.opt['discriminator_ckpt'], self.opt['img_res'])
        else:
            self.criterion_disc = None

    def feed_data(self, data):
        self.original_latent_code = data[0].to(self.device)
        self.original_label = data[1].to(self.device)
        self.gt_label = self.original_label.clone()
        self.gt_label[:, self.target_attr_idx] = \
            self.gt_label[:, self.target_attr_idx] + 1

    def optimize_parameters(self):
        self.field_function.train()

        if self.latent_code_is_w_space and self.transform_z_to_w:
            # translate original z space latent code to w space
            with torch.no_grad():
                original_latent_code = self.stylegan_gen.get_latent(
                    self.original_latent_code)
        else:
            original_latent_code = self.original_latent_code

        # modify latent code via field function
        edited_dict = self.modify_latent_code(original_latent_code)
        edited_image = self.synthesize_image(edited_dict['edited_latent_code'])
        predictor_output = self.predictor(
            transform_image(edited_image, self.img_resize))

        # compute loss function
        loss_total = 0

        assert self.opt['num_attr'] == len(predictor_output)
        loss_list = []

        # iterate over each attribute
        for attr_idx in range(self.opt['num_attr']):
            loss_attr = self.criterion_predictor(predictor_output[attr_idx],
                                                 self.gt_label[:, attr_idx])
            if attr_idx == self.target_attr_idx:
                loss_attr = loss_attr * self.opt['edited_attribute_weight']
            loss_list.append(loss_attr)
        predictor_loss = sum(loss_list) / len(loss_list)
        self.log_dict['predictor_loss'] = predictor_loss

        loss_total += predictor_loss

        if self.criterion_arcface is not None:
            original_image = self.synthesize_image(original_latent_code)
            arcface_loss = self.criterion_arcface(original_image, edited_image,
                                                  self.img_resize)
            loss_total += self.opt['arcface_weight'] * arcface_loss
            self.log_dict['arcface_loss'] = arcface_loss

        if self.opt['disc_weight'] > 0:
            disc_loss = self.criterion_disc(edited_image)
            loss_total += disc_loss * self.opt['disc_weight']
            self.log_dict['disc_loss'] = disc_loss

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        self.log_dict['loss_total'] = loss_total

    def get_current_log(self):
        return self.log_dict

    def update_learning_rate(self, epoch):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        lr = self.optimizer.param_groups[0]['lr']

        if self.opt['lr_decay'] == 'step':
            lr = self.opt['lr'] * (
                self.opt['gamma']**(epoch // self.opt['step']))
        elif self.opt['lr_decay'] == 'cos':
            lr = self.opt['lr'] * (
                1 + math.cos(math.pi * epoch / self.opt['num_epochs'])) / 2
        elif self.opt['lr_decay'] == 'linear':
            lr = self.opt['lr'] * (1 - epoch / self.opt['num_epochs'])
        elif self.opt['lr_decay'] == 'linear2exp':
            if epoch < self.opt['turning_point'] + 1:
                # learning rate decay as 95%
                # at the turning point (1 / 95% = 1.0526)
                lr = self.opt['lr'] * (
                    1 - epoch / int(self.opt['turning_point'] * 1.0526))
            else:
                lr *= self.opt['gamma']
        elif self.opt['lr_decay'] == 'schedule':
            if epoch in self.opt['schedule']:
                lr *= self.opt['gamma']
        else:
            raise ValueError('Unknown lr mode {}'.format(self.opt['lr_decay']))
        # set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def save_network(self, net, save_path):
        """Save networks.

        Args:
            net (nn.Module): Network to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
        """
        state_dict = net.state_dict()
        torch.save(state_dict, save_path)

    def load_network(self, pretrained_field):
        checkpoint = torch.load(pretrained_field)

        self.field_function.load_state_dict(checkpoint, strict=True)
        self.field_function.eval()

    def synthesize_image(self, sample_latent_code):
        synthesized_img, _ = self.stylegan_gen(
            [sample_latent_code],
            truncation=self.truncation,
            input_is_latent=self.input_is_latent,
            truncation_latent=self.truncation_latent,
            randomize_noise=self.randomize_noise)

        return synthesized_img

    def synthesize_and_predict(self, sample_latent_code):
        synthesized_img = self.synthesize_image(sample_latent_code)

        current_predictor_output = self.predictor(
            transform_image(synthesized_img, self.img_resize))
        predicted_label, predicted_score = predictor_to_label(
            current_predictor_output)

        return synthesized_img, predicted_label, predicted_score

    def inference(self, batch_idx, epoch, save_dir):
        self.field_function.eval()

        assert self.original_latent_code.size()[0] == 1

        if self.latent_code_is_w_space and self.transform_z_to_w:
            # translate original z space latent code to w space
            with torch.no_grad():
                original_latent_code = self.stylegan_gen.get_latent(
                    self.original_latent_code)
        else:
            original_latent_code = self.original_latent_code

        with torch.no_grad():
            original_image = self.synthesize_image(original_latent_code)
        original_image = postprocess(original_image.cpu().detach().numpy())

        # field function mapping
        with torch.no_grad():
            return_dict = self.modify_latent_code(original_latent_code)

        with torch.no_grad():
            edited_image, edited_label, _ = self.synthesize_and_predict(
                return_dict['edited_latent_code'])

        edited_image = postprocess(edited_image.cpu().detach().numpy())
        concat_images = cv2.hconcat([original_image[0], edited_image[0]])
        save_image(
            concat_images,
            f'{save_dir}/{batch_idx:03d}_epoch_{epoch:03d}_{self.opt["exp_name"]}_original_{self.original_label[0][self.target_attr_idx]}_edited_{edited_label[self.target_attr_idx]}.png',  # noqa
            need_post_process=False)

        self.field_function.train()

    def continuous_editing(self, latent_codes, save_dir, editing_logger):
        total_num = latent_codes.shape[0]

        for sample_id in range(total_num):
            sample_latent_code = torch.from_numpy(
                latent_codes[sample_id:sample_id + 1]).to(
                    torch.device('cuda'))

            if self.latent_code_is_w_space and self.transform_z_to_w:
                # translate original z space latent code to w space
                with torch.no_grad():
                    sample_latent_code = self.stylegan_gen.get_latent(
                        sample_latent_code)

            # synthesize
            with torch.no_grad():
                original_image, start_label, start_score = \
                    self.synthesize_and_predict(sample_latent_code)

            target_attr_label = int(start_label[self.target_attr_idx])
            target_score = start_score[self.target_attr_idx]

            save_name = f'{sample_id:03d}_num_edits_0_class_{target_attr_label}.png'  # noqa
            save_image(original_image, f'{save_dir}/{save_name}')

            editing_logger.info(f'{save_name}: {start_label}, {start_score}')
            # skip images with low confidence
            if target_score < self.opt['confidence_thresh']:
                editing_logger.info(
                    f'Sample {sample_id:03d} is not confident, skip.')
                continue

            # skip images that are already the max_cls_num
            if target_attr_label == self.opt['max_cls_num']:
                editing_logger.info(
                    f'Sample {sample_id:03d} is already the max_cls_num, skip.'
                )
                continue

            num_trials = 0
            num_edits = 0

            current_stage_scores_list = []
            current_stage_labels_list = []
            current_stage_images_list = []
            current_stage_target_scores_list = []

            previous_target_attr_label = target_attr_label

            if self.fix_layers:
                edited_latent_code = sample_latent_code.unsqueeze(1).repeat(
                    1, self.w_space_channel_num, 1)

            while target_attr_label < self.opt['max_cls_num']:
                num_trials += 1
                with torch.no_grad():
                    # modify sampled latent code
                    if self.fix_layers:
                        # for fix layers, the input to the field_function is w
                        # space, but the input to the stylegan is w plus space
                        edited_dict = self.modify_latent_code(
                            sample_latent_code, edited_latent_code)
                        sample_latent_code = sample_latent_code + edited_dict[
                            'field']
                        edited_latent_code = edited_dict['edited_latent_code']
                    else:
                        # for other modes, the input to the field function and
                        # stylegan are same (both w space or z space)
                        edited_dict = self.modify_latent_code(
                            sample_latent_code)
                        sample_latent_code = edited_dict['edited_latent_code']

                    edited_image, edited_label, edited_score = \
                        self.synthesize_and_predict(edited_dict['edited_latent_code']) # noqa

                target_attr_label = edited_label[self.target_attr_idx]
                target_attr_score = edited_score[self.target_attr_idx]
                if target_attr_label != previous_target_attr_label:
                    num_edits += 1

                if num_edits > 0:
                    if target_attr_label == previous_target_attr_label:
                        current_stage_images_list.append(edited_image)
                        current_stage_labels_list.append(edited_label)
                        current_stage_scores_list.append(edited_score)
                        current_stage_target_scores_list.append(
                            target_attr_score)
                    else:
                        if num_edits > 1:
                            # save images for previous stage
                            max_value = max(current_stage_target_scores_list)
                            max_index = current_stage_target_scores_list.index(
                                max_value)
                            saved_image = current_stage_images_list[max_index]
                            saved_label = current_stage_labels_list[max_index]
                            saved_score = current_stage_scores_list[max_index]
                            save_name = f'{sample_id:03d}_num_edits_{num_edits-1}_class_{previous_target_attr_label}.png'  # noqa
                            save_image(saved_image, f'{save_dir}/{save_name}')
                            editing_logger.info(
                                f'{save_name}: {saved_label}, {saved_score}')

                        current_stage_images_list = []
                        current_stage_labels_list = []
                        current_stage_scores_list = []
                        current_stage_target_scores_list = []
                        num_trials = 0

                        current_stage_images_list.append(edited_image)
                        current_stage_labels_list.append(edited_label)
                        current_stage_scores_list.append(edited_score)
                        current_stage_target_scores_list.append(
                            target_attr_score)

                previous_target_attr_label = target_attr_label
                if self.opt['print_every']:
                    save_name = f'{sample_id:03d}_num_edits_{num_edits}_num_trials_{num_trials}_class_{target_attr_label}.png'  # noqa\
                    saved_image(edited_image, f'{save_dir}/{save_name}')
                    editing_logger.info(
                        f'{save_name}: {edited_label}, {edited_score}')

                if num_trials > self.opt['max_trials_num']:
                    editing_logger.info('Maximum edits num reached.')
                    break

            if num_edits > 0:
                # save images for previous stage
                max_value = max(current_stage_target_scores_list)
                max_index = current_stage_target_scores_list.index(max_value)
                saved_image = current_stage_images_list[max_index]
                saved_label = current_stage_labels_list[max_index]
                saved_score = current_stage_scores_list[max_index]
                save_name = f'{sample_id:03d}_num_edits_{num_edits}_class_{previous_target_attr_label}.png'  # noqa
                save_image(saved_image, f'{save_dir}/{save_name}')
                editing_logger.info(
                    f'{save_name}: {saved_label}, {saved_score}')

            editing_logger.info(f'{sample_id:03d}: Finish editing.')

    def continuous_editing_with_target(self,
                                       latent_codes,
                                       target_cls,
                                       save_dir,
                                       editing_logger,
                                       edited_latent_code,
                                       prefix,
                                       print_intermediate_result=False,
                                       display_img=False):
        total_num = latent_codes.shape[0]

        for sample_id in range(total_num):

            sample_latent_code = torch.from_numpy(
                latent_codes[sample_id:sample_id + 1]).to(
                    torch.device('cuda'))
            start_latent_codes = sample_latent_code
            start_edited_latent_code = edited_latent_code

            exception_mode = 'normal'

            # synthesize
            if edited_latent_code is None:
                if self.latent_code_is_w_space and self.transform_z_to_w:
                    # translate original z space latent code to w space
                    with torch.no_grad():
                        sample_latent_code = self.stylegan_gen.get_latent(
                            sample_latent_code)

                with torch.no_grad():
                    original_image, start_label, start_score = \
                        self.synthesize_and_predict(sample_latent_code)
            else:
                with torch.no_grad():
                    original_image, start_label, start_score = \
                        self.synthesize_and_predict(edited_latent_code)

            target_attr_label = int(start_label[self.target_attr_idx])
            target_score = start_score[self.target_attr_idx]

            # save_name = f'{prefix}_{sample_id:03d}_num_edits_0_class_{target_attr_label}_attr_idx_{self.target_attr_idx}.png'  # noqa
            ### save_image(original_image, f'{save_dir}/{save_name}')

            # editing_logger.info(f'{save_name}: {start_label}, {start_score}')
            # skip images with low confidence
            if target_score < self.opt['confidence_thresh']:
                if editing_logger:
                    editing_logger.info(
                        f'Sample {sample_id:03d} is not confident, skip.')
                continue

            # skip images that are already the target class num
            if target_attr_label == target_cls:
                if editing_logger:
                    editing_logger.info(
                        f'Sample {sample_id:03d} is already at the target class, skip.'
                    )
                # return the exactly the input image and input latent codes
                saved_label = start_label
                saved_latent_code = start_latent_codes
                saved_editing_latent_code = start_edited_latent_code
                saved_score = start_score
                # save_name = f'{prefix}_{sample_id:03d}_num_edits_1_class_{target_attr_label}_attr_idx_{self.target_attr_idx}.png'  # noqa
                ### save_image(original_image, f'{save_dir}/{save_name}')
                # editing_logger.info(
                #     f'{save_name}: {saved_label}, {saved_score}')
                exception_mode = 'already_at_target_class'
                continue
            elif target_attr_label < target_cls:
                direction = 'positive'
                alpha = 1
            elif target_attr_label > target_cls:
                direction = 'negative'
                alpha = -1

            num_trials = 0
            num_edits = 0

            current_stage_scores_list = []
            current_stage_labels_list = []
            current_stage_images_list = []
            current_stage_target_scores_list = []
            current_stage_latent_code_list = []
            current_stage_editing_latent_code_list = []

            previous_target_attr_label = target_attr_label

            if self.fix_layers:
                if edited_latent_code is None:
                    edited_latent_code = sample_latent_code.unsqueeze(
                        1).repeat(1, self.w_space_channel_num, 1)

            while ((direction == 'positive') and
                   (target_attr_label <= target_cls) and
                   (target_attr_label < self.opt['max_cls_num'])) or (
                       (direction == 'negative') and
                       (target_attr_label >= target_cls) and
                       (target_attr_label > self.opt['min_cls_num'])):
                num_trials += 1
                with torch.no_grad():
                    # modify sampled latent code
                    if self.fix_layers:
                        # for fix layers, the input to the field_function is w
                        # space, but the input to the stylegan is w plus space
                        edited_dict = self.modify_latent_code_bidirection(
                            sample_latent_code, edited_latent_code, alpha)
                        sample_latent_code = sample_latent_code + alpha * edited_dict[
                            'field']
                        edited_latent_code = edited_dict['edited_latent_code']
                    else:
                        # for other modes, the input to the field function and
                        # stylegan are same (both w space or z space)
                        edited_dict = self.modify_latent_code_bidirection(
                            latent_code_w=sample_latent_code, alpha=1)
                        sample_latent_code = edited_dict['edited_latent_code']

                    edited_image, edited_label, edited_score = \
                        self.synthesize_and_predict(edited_dict['edited_latent_code']) # noqa

                target_attr_label = edited_label[self.target_attr_idx]
                target_attr_score = edited_score[self.target_attr_idx]

                if ((direction == 'positive') and
                    (target_attr_label > target_cls)) or (
                        (direction == 'negative') and
                        (target_attr_label < target_cls)):
                    if num_edits == 0:
                        saved_label = edited_label
                        saved_latent_code = sample_latent_code
                        saved_editing_latent_code = edited_latent_code
                        save_name = f'{prefix}_{sample_id:03d}_num_edits_{num_edits+1}_class_{target_attr_label}_attr_idx_{self.target_attr_idx}.png'  # noqa
                        saved_image = edited_image
                        saved_score = edited_score
                        save_image(saved_image, f'{save_dir}/{save_name}')
                        if display_img:
                            plt.figure()
                            plt.imshow(mpimg.imread(f'{save_dir}/{save_name}'))
                            plt.axis('off')
                            plt.show()
                        if editing_logger:
                            editing_logger.info(
                                f'{save_name}: {saved_label}, {saved_score}')

                    break

                if target_attr_label != previous_target_attr_label:
                    num_edits += 1

                if num_edits > 0:
                    if target_attr_label == previous_target_attr_label:
                        current_stage_images_list.append(edited_image)
                        current_stage_labels_list.append(edited_label)
                        current_stage_scores_list.append(edited_score)
                        current_stage_target_scores_list.append(
                            target_attr_score)
                        current_stage_latent_code_list.append(
                            sample_latent_code)
                        current_stage_editing_latent_code_list.append(
                            edited_latent_code)
                    else:
                        if num_edits > 1:
                            # save images for previous stage
                            max_value = max(current_stage_target_scores_list)
                            max_index = current_stage_target_scores_list.index(
                                max_value)
                            saved_image = current_stage_images_list[max_index]
                            saved_label = current_stage_labels_list[max_index]
                            saved_score = current_stage_scores_list[max_index]
                            saved_latent_code = current_stage_latent_code_list[
                                max_index]
                            saved_editing_latent_code = current_stage_editing_latent_code_list[
                                max_index]
                            save_name = f'{prefix}_{sample_id:03d}_num_edits_{num_edits-1}_class_{previous_target_attr_label}_attr_idx_{self.target_attr_idx}.png'  # noqa
                            if print_intermediate_result:
                                save_image(saved_image,
                                           f'{save_dir}/{save_name}')
                            if editing_logger:
                                editing_logger.info(
                                    f'{save_name}: {saved_label}, {saved_score}'
                                )

                        current_stage_images_list = []
                        current_stage_labels_list = []
                        current_stage_scores_list = []
                        current_stage_target_scores_list = []
                        current_stage_latent_code_list = []
                        current_stage_editing_latent_code_list = []
                        num_trials = 0

                        current_stage_images_list.append(edited_image)
                        current_stage_labels_list.append(edited_label)
                        current_stage_scores_list.append(edited_score)
                        current_stage_target_scores_list.append(
                            target_attr_score)
                        current_stage_latent_code_list.append(
                            sample_latent_code)
                        current_stage_editing_latent_code_list.append(
                            edited_latent_code)

                previous_target_attr_label = target_attr_label

                if num_trials > self.opt['max_trials_num']:
                    if num_edits == 0:
                        saved_label = start_label
                        saved_latent_code = start_latent_codes
                        saved_editing_latent_code = start_edited_latent_code
                        saved_score = start_score
                        # save_name = f'{prefix}_{sample_id:03d}_num_edits_1_class_{target_attr_label}_attr_idx_{self.target_attr_idx}.png'  # noqa
                        ### save_image(original_image, f'{save_dir}/{save_name}')
                        # if editing_logger:
                        #     editing_logger.info(
                        #         f'{save_name}: {saved_label}, {saved_score}')
                        exception_mode = 'max_edit_num_reached'
                    break

            if num_edits > 0:
                # save images for previous stage
                max_value = max(current_stage_target_scores_list)
                max_index = current_stage_target_scores_list.index(max_value)
                saved_image = current_stage_images_list[max_index]
                saved_label = current_stage_labels_list[max_index]
                saved_score = current_stage_scores_list[max_index]
                saved_latent_code = current_stage_latent_code_list[max_index]
                saved_editing_latent_code = current_stage_editing_latent_code_list[
                    max_index]
                save_name = f'{prefix}_{sample_id:03d}_num_edits_{num_edits}_class_{previous_target_attr_label}_attr_idx_{self.target_attr_idx}.png'  # noqa
                save_image(saved_image, f'{save_dir}/{save_name}')
                if display_img:
                    plt.figure()
                    plt.imshow(mpimg.imread(f'{save_dir}/{save_name}'))
                    plt.axis('off')
                    plt.show()
                if editing_logger:
                    editing_logger.info(
                        f'{save_name}: {saved_label}, {saved_score}')

        return saved_latent_code, saved_editing_latent_code, saved_label, exception_mode
