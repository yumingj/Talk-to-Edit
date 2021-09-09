"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html
requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""

import cv2
import dlib
import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage
from facenet_pytorch import MTCNN
from PIL import Image


def crop_img(img_size, input_img_path, cropped_output_path, device='cuda'):
    if img_size == 128:
        return crop_img_128(input_img_path, cropped_output_path, device)
    elif img_size == 1024:
        return crop_img_1024(input_img_path, cropped_output_path)
    else:
        raise NotImplementedError


def crop_img_128(input_img_path, cropped_output_path, device='cuda'):
    mtcnn = MTCNN(select_largest=True, device=device)

    img = Image.open(input_img_path).convert('RGB')
    img = np.uint8(img)
    bboxes, _ = mtcnn.detect(img)
    w0, h0, w1, h1 = bboxes[0]
    hc, wc = (h0 + h1) / 2, (w0 + w1) / 2
    crop = int(((h1 - h0) + (w1 - w0)) / 2 / 2 * 1.1)
    h0 = int(hc - crop + crop + crop * 0.15)
    w0 = int(wc - crop + crop)

    x0, y0, w, h = w0 - crop, h0 - crop, crop * 2, crop * 2
    im = cv2.imread(input_img_path)
    im_pad = cv2.copyMakeBorder(
        im, h, h, w, w,
        cv2.BORDER_REPLICATE)  # allow cropping outside by replicating borders
    im_crop = im_pad[y0 + h:y0 + h * 2, x0 + w:x0 + w * 2]

    im_crop = cv2.resize(im_crop, (128, 128), interpolation=cv2.INTER_AREA)

    cv2.imwrite(cropped_output_path, im_crop)

    return True


# download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 # noqa
predictor = dlib.shape_predictor(
    './download/pretrained_models/shape_predictor_68_face_landmarks.dat'  # noqa
)


def get_landmark(filepath):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    if len(dets) < 1:
        return False, None

    # print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        # k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        # print("Part 0: {}, Part 1: {} ...".format(
        # shape.part(0), shape.part(1)))

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    # lm is a shape=(68,2) np.array
    return True, lm


def crop_img_1024(input_img_path, cropped_output_path):
    """
    :param filepath: str
    :return: PIL Image
    """

    success, lm = get_landmark(input_img_path)
    if success is False:
        return False

    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(input_img_path)

    output_size = 1024
    transform_size = 4096
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border,
                img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border,
               0), max(-pad[1] + border,
                       0), max(pad[2] - img.size[0] + border,
                               0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
            'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 -
            np.minimum(np.float32(x) / pad[0],
                       np.float32(w - 1 - x) / pad[2]), 1.0 -
            np.minimum(np.float32(y) / pad[1],
                       np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) -
                img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(
            np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    img.save(cropped_output_path)

    return True
