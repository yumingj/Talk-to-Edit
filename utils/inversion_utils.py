import math

import models.archs.stylegan2.lpips as lpips
import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from utils.crop_img import crop_img


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss +
                (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2) +
                (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2))

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (tensor.detach().clamp_(min=-1, max=1).add(1).div_(2).mul(255).type(
        torch.uint8).permute(0, 2, 3, 1).to("cpu").numpy())


def inversion(opt, field_model):

    inv_opt = opt['inversion']
    device = inv_opt['device']

    img_size = opt['img_res']

    # inversion
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    if inv_opt['crop_img']:
        cropped_output_path = f'{opt["path"]["visualization"]}/cropped.png'
        crop_img(img_size, inv_opt['img_path'], cropped_output_path, device)
        img = transform(Image.open(cropped_output_path).convert("RGB"))
    else:
        img = transform(Image.open(inv_opt['img_path']).convert("RGB"))

    img = img.unsqueeze(0).to(torch.device('cuda'))

    batch, channel, height, width = img.shape

    if height > 256:
        factor = height // 256

        img = img.reshape(batch, channel, height // factor, factor,
                          width // factor, factor)
        img = img.mean([3, 5])

    n_mean_latent = 10000
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = field_model.stylegan_gen.style_forward(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() /
                      n_mean_latent)**0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(
        img.shape[0], 1)

    latent_in.requires_grad = True

    optim_params = []

    for v in field_model.stylegan_gen.parameters():
        if v.requires_grad:
            optim_params.append(v)
    optimizer = optim.Adam([{
        'params': [latent_in]
    }, {
        'params': optim_params,
        'lr': inv_opt['lr_gen']
    }],
                           lr=inv_opt['lr'])

    pbar = tqdm(range(inv_opt['step']))

    latent_path = []
    for i in pbar:
        t = i / inv_opt['step']
        lr = get_lr(t, inv_opt['lr'])
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * inv_opt['noise'] * max(
            0, 1 - t / inv_opt['noise_ramp'])**2

        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen, _ = field_model.stylegan_gen([latent_n],
                                              input_is_latent=True,
                                              randomize_noise=False)

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(batch, channel, height // factor, factor,
                                      width // factor, factor)
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, img).sum()
        mse_loss = F.mse_loss(img_gen, img)

        loss = p_loss + inv_opt['img_mse_weight'] * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description((f"total: {loss:.4f}; perceptual: {p_loss:.4f};"
                              f" mse: {mse_loss:.4f}; lr: {lr:.4f}"))

    latent_code = latent_in[0].cpu().detach().numpy()
    latent_code = np.expand_dims(latent_code, axis=0)

    return latent_code
