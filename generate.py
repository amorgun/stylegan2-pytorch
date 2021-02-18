import argparse

import pathlib
import torch
from torchvision import utils
from lightning import StyleGAN2
from tqdm import tqdm


def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        root = pathlib.Path(args.folder)
        root.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                root/f"{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size-h", type=int, default=1024, help="image height for the model"
    )
    parser.add_argument(
        "--size-w", type=int, default=1024, help="image width for the model"
    )
    parser.add_argument(
        "--log-size", type=int, default=7, help="depth of the model"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
#         default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default='sample',
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = StyleGAN2.load_from_checkpoint(checkpoint_path=args.ckpt).g_ema.to(args.device)
#     checkpoint = torch.load(args.ckpt)
#     g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, args.device, mean_latent)
