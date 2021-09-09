import argparse
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.archs.attribute_predictor_arch import resnet50
from models.losses.arcface_loss import resnet_face18
from models.utils import output_to_label
from PIL import Image


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Continuous image editing via field function')

    # inference
    parser.add_argument(
        '--attribute',
        type=str,
        required=True,
        help='[Bangs, Eyeglasses, No_Beard, Smiling, Young]')

    parser.add_argument('--confidence_thresh', type=float, default=0.8)

    # input and output directories
    parser.add_argument(
        '--work_dir',
        required=True,
        type=str,
        metavar='PATH',
        help='path to save checkpoint and log files.')
    parser.add_argument(
        '--image_dir',
        required=True,
        type=str,
        metavar='PATH',
        help='path to save checkpoint and log files.')
    parser.add_argument('--image_num', type=int, required=True)
    parser.add_argument('--debug', default=0, type=int)

    # predictor args
    parser.add_argument(
        '--attr_file',
        required=True,
        type=str,
        help='directory to attribute metadata')
    parser.add_argument(
        '--predictor_ckpt',
        required=True,
        type=str,
        help='The pretrained network weights for testing')
    parser.add_argument('--num_attr', type=int, default=5)

    # arcface loss args
    parser.add_argument(
        '--pretrained_arcface',
        default=  # noqa
        '../share_work_dirs/pretrained_arcface/arcface_resnet18_110.pth',
        type=str)

    return parser.parse_args()


def get_edited_images_list(img_dir, img_idx):
    return_img_list = []
    img_path_list = glob.glob(f'{img_dir}/{img_idx:03d}_*.png')
    start_img_path = glob.glob(f'{img_dir}/{img_idx:03d}_num_edits_0_*.png')
    assert len(start_img_path) == 1
    return_img_list.append(start_img_path[0])

    num_edits = len(img_path_list) - 1
    if num_edits > 0:
        for edit_idx in range(1, num_edits + 1):
            img_path_edit_list = glob.glob(
                f'{img_dir}/{img_idx:03d}_num_edits_{edit_idx}_*.png')
            assert len(img_path_edit_list) == 1
            return_img_list.append(img_path_edit_list[0])

    return return_img_list


def load_image_predictor(img_path,
                         transform=transforms.Compose([transforms.ToTensor()
                                                       ])):
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = image.to(torch.device('cuda')).unsqueeze(0)

    if image.size()[-1] > 128:
        image = F.interpolate(image, (128, 128), mode='area')

    img_mean = torch.Tensor([0.485, 0.456,
                             0.406]).view(1, 3, 1, 1).to(torch.device('cuda'))
    img_std = torch.Tensor([0.229, 0.224,
                            0.225]).view(1, 3, 1, 1).to(torch.device('cuda'))
    image = (image - img_mean) / img_std

    return image


def load_image_arcface(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = image[:, :, np.newaxis]
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5

    image = torch.from_numpy(image).to(torch.device('cuda'))

    if image.size()[-1] > 128:
        image = F.interpolate(image, (128, 128), mode='area')

    return image


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def predictor_score(predictor_output, gt_label, target_attr_idx,
                    criterion_predictor):
    num_attr = len(predictor_output)
    loss_avg = 0
    count = 0
    for attr_idx in range(num_attr):
        if attr_idx == target_attr_idx:
            continue
        loss_attr = criterion_predictor(
            predictor_output[attr_idx],
            gt_label[attr_idx].unsqueeze(0).to(torch.device('cuda')))

        loss_avg += loss_attr
        count += 1
    loss_avg = loss_avg / count

    return loss_avg


def compute_num_metrics(image_dir, image_num, pretrained_arcface, attr_file,
                        pretrained_predictor, target_attr_idx, logger):

    # define arcface model
    arcface_model = resnet_face18(use_se=False)
    arcface_model = nn.DataParallel(arcface_model)
    arcface_model.load_state_dict(torch.load(pretrained_arcface), strict=True)
    arcface_model.to(torch.device('cuda'))
    arcface_model.eval()

    # define predictor model
    predictor = predictor = resnet50(attr_file=attr_file)
    predictor = predictor.to(torch.device('cuda'))

    checkpoint = torch.load(pretrained_predictor)
    predictor.load_state_dict(checkpoint['state_dict'], strict=True)
    predictor.eval()

    criterion_predictor = nn.CrossEntropyLoss(reduction='mean')

    arcface_sim_dataset = 0
    predictor_score_dataset = 0
    count = 0
    for img_idx in range(image_num):
        edit_image_list = get_edited_images_list(image_dir, img_idx)
        num_edits = len(edit_image_list) - 1
        arcface_sim_img = 0
        predictor_score_img = 0
        if num_edits > 0:
            # read image for arcface
            source_img_arcface = load_image_arcface(edit_image_list[0])
            with torch.no_grad():
                source_feature = arcface_model(
                    source_img_arcface).cpu().numpy()
            # read image for predictor
            source_img_predictor = load_image_predictor(edit_image_list[0])
            with torch.no_grad():
                source_predictor_output = predictor(source_img_predictor)
            source_label, score = output_to_label(source_predictor_output)
            for edit_idx in range(1, num_edits + 1):
                # arcface cosine similarity
                edited_img_arcface = load_image_arcface(
                    edit_image_list[edit_idx])
                with torch.no_grad():
                    edited_feature = arcface_model(
                        edited_img_arcface).cpu().numpy()
                temp_arcface_sim = cosin_metric(source_feature,
                                                edited_feature.transpose(
                                                    1, 0))[0][0]
                arcface_sim_img += temp_arcface_sim
                # predictor score
                edited_img_predictor = load_image_predictor(
                    edit_image_list[edit_idx])
                with torch.no_grad():
                    edited_predictor_output = predictor(edited_img_predictor)
                temp_predictor_score_img = predictor_score(
                    edited_predictor_output, source_label, target_attr_idx,
                    criterion_predictor)
                predictor_score_img += temp_predictor_score_img

            arcface_sim_img = arcface_sim_img / num_edits
            predictor_score_img = predictor_score_img / num_edits
            arcface_sim_dataset += arcface_sim_img
            predictor_score_dataset += predictor_score_img
            count += 1
            logger.info(
                f'{img_idx:03d}: Arcface: {arcface_sim_img: .4f}, Predictor: {predictor_score_img: .4f}.'  # noqa
            )
        else:
            logger.info(f'{img_idx:03d}: no available edits.')

    arcface_sim_dataset = arcface_sim_dataset / count
    predictor_score_dataset = predictor_score_dataset / count
    logger.info(
        f'Avg: {arcface_sim_dataset: .4f}, {predictor_score_dataset: .4f}.')

    return arcface_sim_dataset, predictor_score_dataset
