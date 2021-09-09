import argparse
import glob
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image

from models.archs.attribute_predictor_arch import resnet50
from models.utils import output_to_label
from utils.logger import get_root_logger
from utils.options import dict2str

attr_predictor_eval_ckpt = './download/pretrained_models/eval_predictor.pth.tar'


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


def load_face_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 128.0

    image = torch.from_numpy(image).to(torch.device('cuda'))

    return image


def load_image_predictor(img_path,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
                         ])):
    image = Image.open(img_path).convert('RGB')
    image = transform(image)

    image = image.to(torch.device('cuda')).unsqueeze(0)

    return image


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


def compute_num_metrics(image_dir, image_num, target_attr_idx, logger):
    # use different face model and predictor model from training phase
    # define face recognition model
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(
        torch.device('cuda'))

    # define attribute predictor model
    predictor = resnet50(attr_file='./configs/attributes_5.json', )
    predictor = predictor.to(torch.device('cuda'))

    checkpoint = torch.load(attr_predictor_eval_ckpt)
    predictor.load_state_dict(checkpoint['state_dict'], strict=True)
    predictor.eval()

    criterion_predictor = nn.CrossEntropyLoss(reduction='mean')

    face_distance_dataset = 0
    predictor_score_dataset = 0
    count = 0
    for img_idx in range(image_num):
        edit_image_list = get_edited_images_list(image_dir, img_idx)
        num_edits = len(edit_image_list) - 1
        face_distance_img = 0
        predictor_score_img = 0
        if num_edits > 0:
            # face recognition feature
            source_img = load_face_image(edit_image_list[0])
            source_img_feat = resnet(source_img)
            # attribute label for predictor
            source_img_predictor = load_image_predictor(edit_image_list[0])
            with torch.no_grad():
                source_predictor_output = predictor(source_img_predictor)
            source_label, score = output_to_label(source_predictor_output)
            for edit_idx in range(1, num_edits + 1):
                edited_img = load_face_image(edit_image_list[edit_idx])
                edited_img_feat = resnet(edited_img)
                temp_face_dist = torch.norm(source_img_feat -
                                            edited_img_feat).item()
                face_distance_img += temp_face_dist
                # attribute predictor score
                edited_img_predictor = load_image_predictor(
                    edit_image_list[edit_idx])
                with torch.no_grad():
                    edited_predictor_output = predictor(edited_img_predictor)
                temp_predictor_score_img = predictor_score(
                    edited_predictor_output, source_label, target_attr_idx,
                    criterion_predictor)
                predictor_score_img += temp_predictor_score_img

            face_distance_img = face_distance_img / num_edits
            face_distance_dataset += face_distance_img
            predictor_score_img = predictor_score_img / num_edits
            predictor_score_dataset += predictor_score_img
            count += 1
            logger.info(
                f'{img_idx:03d}: Identity Preservation: {face_distance_img: .4f}, Attribute Preservation: {predictor_score_img: .4f}.'
            )
        else:
            logger.info(f'{img_idx:03d}: no available edits.')

    face_distance_dataset = face_distance_dataset / count
    predictor_score_dataset = predictor_score_dataset / count
    logger.info(
        f'Avg: {face_distance_dataset: .4f}, {predictor_score_dataset: .4f}.')

    return face_distance_dataset, predictor_score_dataset


def main():
    """Main function."""
    args = parse_args()
    args.attr_dict = {
        'Bangs': 0,
        'Eyeglasses': 1,
        'No_Beard': 2,
        'Smiling': 3,
        'Young': 4
    }

    logger = get_root_logger(
        logger_name='base',
        log_level=logging.INFO,
        log_file=f'{args.work_dir}/quantitative_results.txt')
    logger.info(dict2str(args.__dict__))

    compute_num_metrics(args.image_dir, args.image_num,
                        args.attr_dict[args.attribute], logger)


if __name__ == '__main__':
    main()
