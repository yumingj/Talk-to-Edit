import argparse
import os
import sys

import numpy as np
import torch
from torchvision import utils
from tqdm import tqdm

sys.path.append('..')
from stylegan2_pytorch.model import Generator


def generate(args, g_ema, device, mean_latent):

    if not os.path.exists(args.synthetic_image_dir):
        os.makedirs(args.synthetic_image_dir)

    latent_code = {}
    w_space_code = {}
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, w_space = g_ema([sample_z],
                                    truncation=args.truncation,
                                    truncation_latent=mean_latent,
                                    return_latents=True,
                                    randomize_noise=False)

            utils.save_image(
                sample,
                os.path.join(args.synthetic_image_dir,
                             f"{str(i).zfill(7)}.png"),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            latent_code[f"{str(i).zfill(7)}.png"] = sample_z.cpu().numpy()
            w_space_code[f"{str(i).zfill(7)}.png"] = w_space.cpu().numpy()

    # save latent code
    np.save(f'{args.synthetic_image_dir}/latent_code.npz', latent_code)
    np.save(f'{args.synthetic_image_dir}/w_space_code.npz', w_space_code)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Generate samples from the generator")

    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="output image size of the generator")
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics",
        type=int,
        default=20,
        help="number of images to be generated")
    parser.add_argument(
        "--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--synthetic_image_dir",
        default='',
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size,
        args.latent,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
