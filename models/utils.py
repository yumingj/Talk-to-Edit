import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def postprocess(images, channel_order='BGR', min_val=-1.0, max_val=1.0):
    """Postprocesses the output images if needed.

        This function assumes the input numpy array is with shape [batch_size,
        channel, height, width]. Here, `channel = 3` for color image and
        `channel = 1` for grayscale image. The return images are with shape
        [batch_size, height, width, channel]. NOTE: The channel order of output
        image will always be `RGB`.

        Args:
          images: The raw output from the generator.

        Returns:
          The postprocessed images with dtype `numpy.uint8` with range
            [0, 255].

        Raises:
          ValueError: If the input `images` are not with type `numpy.ndarray`
            or not with shape [batch_size, channel, height, width].
        """
    if not isinstance(images, np.ndarray):
        raise ValueError('Images should be with type `numpy.ndarray`!')

    images_shape = images.shape
    if len(images_shape) != 4 or images_shape[1] not in [1, 3]:
        raise ValueError(f'Input should be with shape [batch_size, channel, '
                         f'height, width], where channel equals to 1 or 3. '
                         f'But {images_shape} is received!')
    images = (images - min_val) * 255 / (max_val - min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    if channel_order == 'BGR':
        images = images[:, :, :, ::-1]

    return images


def transform_image(image, resize=False):
    # transform image range to [0, 1]
    image = (image + 1) * 255 / 2
    # TODO: int()? quantization?
    image = torch.clamp(image + 0.5, 0, 255)
    image = image / 255.
    if resize:
        image = F.interpolate(image, (128, 128), mode='area')

    # normalize image to imagenet range
    img_mean = torch.Tensor([0.485, 0.456,
                             0.406]).view(1, 3, 1, 1).to(torch.device('cuda'))
    img_std = torch.Tensor([0.229, 0.224,
                            0.225]).view(1, 3, 1, 1).to(torch.device('cuda'))
    image = (image - img_mean) / img_std

    return image


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def output_to_label(output):
    """
    INPUT
    - output: [num_attr, batch_size, num_classes]
    OUTPUT
    - scores: [num_attr, batch_size, num_classes] (softmaxed)
    - label: [num_attr, batch_size]
    """
    scores = []
    labels = []
    for attr_idx in range(len(output)):
        _, label = torch.max(input=output[attr_idx], dim=1)
        label = label.cpu().numpy()[0]
        labels.append(label)

        score_per_attr = output[attr_idx].cpu().numpy()[0]
        # softmax
        score_per_attr = (np.exp(score_per_attr) /
                          np.sum(np.exp(score_per_attr)))[label]
        scores.append(score_per_attr)

    scores = torch.FloatTensor(scores)
    labels = torch.LongTensor(labels)

    return labels, scores


def predictor_to_label(predictor_output):

    scores = []
    labels = []
    for attr_idx in range(len(predictor_output)):
        _, label = torch.max(input=predictor_output[attr_idx], dim=1)
        label = label.cpu().numpy()[0]
        labels.append(label)

        score_per_attr = predictor_output[attr_idx].cpu().numpy()[0]
        # softmax
        score_per_attr = (np.exp(score_per_attr) /
                          np.sum(np.exp(score_per_attr)))[label]
        scores.append(score_per_attr)

    return labels, scores


def save_image(img, save_path, need_post_process=True):
    if need_post_process:
        cv2.imwrite(save_path, postprocess(img.cpu().detach().numpy())[0])
    else:
        cv2.imwrite(save_path, img)
