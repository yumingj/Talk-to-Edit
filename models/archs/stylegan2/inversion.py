import argparse
import math
import os

import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

import lpips
from model import Generator


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


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="output image sizes of the generator")
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.05,
        help="strength of the noise level")
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument(
        "--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--randomise_noise", type=int, default=1)
    parser.add_argument(
        "--img_mse_weight",
        type=float,
        default=0,
        help="weight of the mse loss")
    parser.add_argument(
        "files",
        metavar="FILES",
        nargs="+",
        help="path to image files to be projected")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "--postfix", default='', type=str, help='postfix for filenames')
    parser.add_argument(
        "--latent_type",
        required=True,
        type=str,
        help='z or w, not case sensitive')
    parser.add_argument(
        "--w_path", default='', type=str, help='path to w latent code')
    parser.add_argument('--w_mse_weight', default=0, type=float)
    parser.add_argument('--w_loss_type', default='mse', type=str)

    args = parser.parse_args()

    # latent space type
    args.latent_type = args.latent_type.lower()
    if args.latent_type == 'z':
        args.input_is_latent = False
    elif args.latent_type == 'w':
        args.input_is_latent = True
    else:
        assert False, "Unrecognized args.latent_type"

    n_mean_latent = 10000

    resize = min(args.size, 256)

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    imgs = []

    for imgfile in args.files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    if args.w_mse_weight:
        assert args.latent_type == 'z'
        w_latent_code = np.load(args.w_path)
        w_latent_code = torch.tensor(w_latent_code).to(device)

    # g_ema = Generator(args.size, 512, 8) # ziqi modified
    g_ema = Generator(args.size, 512, 8, 1)

    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() /
                      n_mean_latent)**0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))

    if args.latent_type == 'w':
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(
            imgs.shape[0], 1)
    elif args.latent_type == 'z':
        latent_in = noise_sample.mean(0).detach().clone().unsqueeze(0).repeat(
            imgs.shape[0], 1)

    if args.w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True

    if args.randomise_noise:
        print('Noise term will be optimized together.')
        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
        for noise in noises:
            noise.requires_grad = True
        optimizer = optim.Adam(
            [latent_in] + noises + [g_ema.parameters()], lr=args.lr)
    else:
        optim_params = []
        for v in g_ema.parameters():
            if v.requires_grad:
                optim_params.append(v)
        optimizer = optim.Adam([{
            'params': [latent_in]
        }, {
            'params': optim_params,
            'lr': 1e-4
        }],
                               lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(
            0, 1 - t / args.noise_ramp)**2
        if args.latent_type == 'z':
            latent_w = g_ema.style(latent_in)
            latent_n = latent_noise(latent_w, noise_strength.item())
        else:
            latent_n = latent_noise(latent_in, noise_strength.item())

        if args.randomise_noise:
            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)
        else:
            img_gen, _ = g_ema([latent_n],
                               input_is_latent=True,
                               randomize_noise=False)

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(batch, channel, height // factor, factor,
                                      width // factor, factor)
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).sum()
        mse_loss = F.mse_loss(img_gen, imgs)
        if args.randomise_noise:
            n_loss = noise_regularize(noises)
        else:
            n_loss = 0

        loss = p_loss + args.noise_regularize * n_loss + args.img_mse_weight * mse_loss

        if args.w_mse_weight > 0:
            # this loss is only applicable to z space
            assert args.latent_type == 'z'
            if args.w_loss_type == 'mse':
                w_mse_loss = F.mse_loss(latent_w, w_latent_code)
            elif args.w_loss_type == 'l1':
                w_mse_loss = F.l1_loss(latent_w, w_latent_code)
            loss += args.w_mse_weight * w_mse_loss
        else:
            w_mse_loss = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.randomise_noise:
            noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description((
            f"total: {loss:.4f}; perceptual: {p_loss:.4f}; noise regularize: {n_loss:.4f};"
            f" mse: {mse_loss:.4f}; w_mse_loss: {w_mse_loss:.4f}; lr: {lr:.4f}"
        ))

    if args.randomise_noise:
        img_gen, _ = g_ema([latent_path[-1]],
                           input_is_latent=args.input_is_latent,
                           noise=noises)
    else:
        img_gen, _ = g_ema([latent_path[-1]],
                           input_is_latent=args.input_is_latent,
                           randomize_noise=False)

    filename = os.path.splitext(os.path.basename(args.files[0]))[0] + ".pt"

    img_ar = make_image(img_gen)

    result_file = {}
    for i, input_name in enumerate(args.files):
        result_file[input_name] = {"img": img_gen[i], "latent": latent_in[i]}
        if args.randomise_noise:
            noise_single = []
            for noise in noises:
                noise_single.append(noise[i:i + 1])
            result_file[input_name]["noise"] = noise_single

        img_name = os.path.splitext(
            os.path.basename(input_name)
        )[0] + '_' + args.postfix + '-' + args.latent_type + "-project.png"
        pil_img = Image.fromarray(img_ar[i])

        # save image
        if not os.path.isdir(os.path.join(args.output_dir, 'recovered_image')):
            os.makedirs(
                os.path.join(args.output_dir, 'recovered_image'),
                exist_ok=False)
        pil_img.save(
            os.path.join(args.output_dir, 'recovered_image', img_name))

        latent_code = latent_in[i].cpu()
        latent_code = latent_code.detach().numpy()
        latent_code = np.expand_dims(latent_code, axis=0)
        print('latent_code:', len(latent_code), len(latent_code[0]))
        # save latent code
        if not os.path.isdir(os.path.join(args.output_dir, 'latent_codes')):
            os.makedirs(
                os.path.join(args.output_dir, 'latent_codes'), exist_ok=False)
        np.save(
            f'{args.output_dir}/latent_codes/{img_name}_{args.latent_type}.npz.npy',
            latent_code)

        if not os.path.isdir(os.path.join(args.output_dir, 'checkpoint')):
            os.makedirs(
                os.path.join(args.output_dir, 'checkpoint'), exist_ok=False)
        torch.save(
            {
                "g_ema": g_ema.state_dict(),
            },
            f"{os.path.join(args.output_dir, 'checkpoint')}/{img_name}_{args.latent_type}.pt",
        )

    # save info
    if not os.path.isdir(os.path.join(args.output_dir, 'pt')):
        os.makedirs(os.path.join(args.output_dir, 'pt'), exist_ok=False)
    torch.save(
        result_file,
        os.path.join(
            args.output_dir,
            os.path.join(args.output_dir, 'pt',
                         filename + '_' + args.latent_type)))
