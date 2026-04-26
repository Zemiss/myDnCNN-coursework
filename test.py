"""Evaluate a trained denoising model on Set12 or Set68."""

import argparse
import glob
import os

import cv2
import jittor as jt
import numpy as np

from models import UNet
from utils import batch_PSNR, batch_SSIM, normalize_uint8


PAD_FACTOR = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DnCNN-style denoiser")
    parser.add_argument("--num_of_layers", type=int, default=17, help="Kept for old commands.")
    parser.add_argument("--logdir", type=str, default="logs/DnCNN-B")
    parser.add_argument("--test_data", type=str, default="Set12", choices=("Set12", "Set68"))
    parser.add_argument("--test_noiseL", type=float, default=25)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--use-cuda", type=lambda value: value.lower() in ("1", "true", "yes"), default=True)
    return parser.parse_args()


def pad_to_multiple(tensor, factor=PAD_FACTOR):
    """Pad BCHW tensor on the bottom/right edges to match the U-Net stride."""

    _, _, height, width = tensor.shape
    padded_height = (height + factor - 1) // factor * factor
    padded_width = (width + factor - 1) // factor * factor
    pad_h = padded_height - height
    pad_w = padded_width - width
    if pad_h == 0 and pad_w == 0:
        return tensor, height, width

    padded = np.pad(tensor.numpy(), ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return jt.array(padded), height, width


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = normalize_uint8(np.float32(image))
    image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
    return jt.array(image)


def evaluate_image(model, clean, noise_level):
    padded_clean, original_height, original_width = pad_to_multiple(clean)
    sigma = noise_level / 255.0
    noise = jt.randn(padded_clean.shape) * sigma
    noisy = padded_clean + noise
    noise_map = jt.full_like(padded_clean, sigma)
    model_input = jt.concat([noisy, noise_map], dim=1)

    with jt.no_grad():
        predicted_noise = model(model_input)
        restored = jt.clamp(noisy - predicted_noise, 0.0, 1.0)

    restored = restored[..., :original_height, :original_width]
    return batch_PSNR(restored, clean, 1.0), batch_SSIM(restored, clean, 1.0)


def main():
    args = parse_args()
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    print(f"Using Jittor with CUDA: {jt.flags.use_cuda}")

    model_path = os.path.join(args.logdir, "net.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("Loading model ...")
    model = UNet(channels=1)
    model.load_state_dict(jt.load(model_path))
    model.eval()

    pattern = os.path.join(args.data_dir, args.test_data, "*.png")
    image_files = sorted(glob.glob(pattern))
    if not image_files:
        raise FileNotFoundError(f"No images found: {pattern}")

    total_psnr = 0.0
    total_ssim = 0.0
    for image_path in image_files:
        clean = load_image(image_path)
        psnr, ssim = evaluate_image(model, clean, args.test_noiseL)
        total_psnr += psnr
        total_ssim += ssim
        print(f"{os.path.basename(image_path)} PSNR {psnr:.4f} SSIM {ssim:.4f}")

    count = len(image_files)
    print(f"\nAverage PSNR on {args.test_data} (Noise={args.test_noiseL}): {total_psnr / count:.4f}")
    print(f"Average SSIM on {args.test_data} (Noise={args.test_noiseL}): {total_ssim / count:.4f}")


if __name__ == "__main__":
    main()
