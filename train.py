"""Train the blind grayscale denoising model."""

import argparse
import os

import jittor as jt
import numpy as np
from jittor import nn, optim
from jittor.dataset import DataLoader
from tensorboardX import SummaryWriter

from dataset import Dataset, prepare_data
from models import UNet
from utils import batch_PSNR, batch_SSIM, weights_init_kaiming


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DnCNN-style blind denoiser")
    parser.add_argument("--preprocess", type=str2bool, default=False)
    parser.add_argument("--batch-size", "--batchSize", dest="batch_size", type=int, default=128)
    parser.add_argument("--num_of_layers", type=int, default=17, help="Kept for old commands.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--milestone", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outf", type=str, default="logs/DnCNN-B")
    parser.add_argument("--mode", choices=("S", "B"), default="B")
    parser.add_argument("--noiseL", type=float, default=25, help="Noise level for mode S.")
    parser.add_argument("--val_noiseL", type=float, default=25)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-cuda", type=str2bool, default=True)
    return parser.parse_args()


def make_noise(clean, mode, noise_level, blind_range=(0, 55)):
    """Create Gaussian noise and the matching noise-level map."""

    if mode == "S":
        sigma = noise_level / 255.0
        return jt.randn(clean.shape) * sigma, jt.full_like(clean, sigma)

    noise = jt.zeros(clean.shape)
    noise_map = jt.zeros(clean.shape)
    sigmas = np.random.uniform(blind_range[0], blind_range[1], size=clean.shape[0])
    for batch_idx, sigma in enumerate(sigmas):
        sigma = sigma / 255.0
        sample_shape = noise[batch_idx, :, :, :].shape
        noise[batch_idx, :, :, :] = jt.randn(sample_shape) * sigma
        noise_map[batch_idx, :, :, :] = sigma
    return noise, noise_map


def build_loaders(args):
    train_set = Dataset(train=True)
    val_set = Dataset(train=False)
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset=val_set,
        num_workers=args.num_workers,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    return train_set, val_set, train_loader, val_loader


def train_one_epoch(model, criterion, optimizer, loader, args, epoch, step, writer, train_size):
    num_batches = (train_size + args.batch_size - 1) // args.batch_size
    current_lr = args.lr if epoch < args.milestone else args.lr / 10.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    print(f"learning rate {current_lr:.6f}")

    model.train()
    for batch_idx, clean in enumerate(loader, start=1):
        optimizer.zero_grad()
        noise, noise_map = make_noise(clean, args.mode, args.noiseL)
        noisy = clean + noise
        model_input = jt.concat([noisy, noise_map], dim=1)

        predicted_noise = model(model_input)
        loss = criterion(predicted_noise, noise) / (noisy.shape[0] * 2)
        optimizer.step(loss)

        restored = jt.clamp(noisy - predicted_noise, 0.0, 1.0)
        psnr = batch_PSNR(restored, clean, 1.0)
        ssim = batch_SSIM(restored, clean, 1.0)
        print(
            "[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f"
            % (epoch + 1, batch_idx, num_batches, loss.item(), psnr, ssim)
        )

        if step % 10 == 0:
            writer.add_scalar("loss", loss.item(), step)
            writer.add_scalar("PSNR/train", psnr, step)
            writer.add_scalar("SSIM/train", ssim, step)
        step += 1
    return step


def validate(model, loader, args, val_size):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0

    with jt.no_grad():
        for clean in loader:
            sigma = args.val_noiseL / 255.0
            noise = jt.randn(clean.shape) * sigma
            noisy = clean + noise
            noise_map = jt.full_like(clean, sigma)
            model_input = jt.concat([noisy, noise_map], dim=1)

            predicted_noise = model(model_input)
            restored = jt.clamp(noisy - predicted_noise, 0.0, 1.0)
            total_psnr += batch_PSNR(restored, clean, 1.0)
            total_ssim += batch_SSIM(restored, clean, 1.0)

    return total_psnr / val_size, total_ssim / val_size


def main():
    args = parse_args()
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    os.makedirs(args.outf, exist_ok=True)

    if args.preprocess:
        aug_times = 1 if args.mode == "S" else 2
        prepare_data(
            data_path=args.data_dir,
            patch_size=64,
            stride=10,
            aug_times=aug_times,
        )

    print("Loading dataset ...")
    train_set, val_set, train_loader, val_loader = build_loaders(args)
    print(f"# of training samples: {len(train_set)}")

    model = UNet(channels=1)
    model.apply(weights_init_kaiming)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.outf)

    step = 0
    for epoch in range(args.epochs):
        step = train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            args,
            epoch,
            step,
            writer,
            len(train_set),
        )
        psnr, ssim = validate(model, val_loader, args, len(val_set))
        print(f"\n[epoch {epoch + 1}] PSNR_val: {psnr:.4f} SSIM_val: {ssim:.4f}")
        writer.add_scalar("PSNR/val", psnr, epoch)
        writer.add_scalar("SSIM/val", ssim, epoch)
        jt.save(model.state_dict(), os.path.join(args.outf, "net.pkl"))

    writer.close()


if __name__ == "__main__":
    main()
