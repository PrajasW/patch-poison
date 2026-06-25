# PatchPoison
# Calculates PSNR, LPIPS and SSIM between
# 1. Clean Dataset and Poisoned Dataset (imperceptibility)
# 2. Poisoned Dataset and The Renders (reconstruction degradation)


import torch
import numpy as np
import os
import sys
import random
from random import randint
import uuid
import time
import re
import cv2
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import multiprocessing
from gpuinfo import GPUInfo
from datetime import datetime
import matplotlib.pyplot as plt

def load_image_as_tensor(image_path):
    """Load an image and convert to tensor format (C, H, W) normalized to [0, 1]."""
    # Load with alpha channel if PNG
    if image_path.endswith('.png'):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is not None and len(image.shape) == 3 and image.shape[2] == 4:
            # BGRA to RGB, ignore alpha for comparison
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is None:
        return None
    
    # Convert to tensor (H, W, C) -> (C, H, W), normalize to [0, 1]
    tensor = torch.from_numpy(image).float() / 255.0
    tensor = tensor.permute(2, 0, 1)  # (C, H, W)
    return tensor

def find_images_in_dataset(dataset_path):
    """Find all images in a dataset directory (checking train folder first, then root)."""
    image_extensions = ('.png', '.jpg', '.jpeg')
    images = {}
    
    # Check common subdirectories
    search_dirs = [
        os.path.join(dataset_path, 'train'),
        os.path.join(dataset_path, 'images'),
        dataset_path
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for filename in os.listdir(search_dir):
                if filename.lower().endswith(image_extensions):
                    images[filename] = os.path.join(search_dir, filename)
    
    return images

def benchmark_dataset_comparison(args):
    """Compare images between source dataset (-s) and base dataset (-b)."""
    if not args.base_path:
        print("No base path provided (-b), skipping dataset comparison.")
        return
    
    print(f"\n=== Comparing datasets ===")
    print(f"source: {args.source_path}")
    print(f"Base:   {args.base_path}")
    
    # Find images in both datasets
    source_images = find_images_in_dataset(args.source_path)
    base_images = find_images_in_dataset(args.base_path)
    
    print(f"Found {len(source_images)} images in source dataset")
    print(f"Found {len(base_images)} images in base dataset")
    
    # Find common images
    common_images = set(source_images.keys()) & set(base_images.keys())
    print(f"Found {len(common_images)} common images")
    
    if len(common_images) == 0:
        print("No common images found between datasets!")
        return
    
    os.makedirs(args.model_path, exist_ok=True)
    comparison_dir = os.path.join(args.model_path, args.dataset_comparison_dir)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Initialize LPIPS model
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    
    SSIM_views = []
    PSNR_views = []
    LPIPS_views = []
    
    for idx, filename in enumerate(sorted(common_images)):
        source_path = source_images[filename]
        base_path = base_images[filename]
        
        source_tensor = load_image_as_tensor(source_path)
        base_tensor = load_image_as_tensor(base_path)
        
        if source_tensor is None or base_tensor is None:
            print(f"Warning: Could not load {filename}, skipping...")
            continue
        
        # Move to GPU
        source_tensor = source_tensor.cuda()
        base_tensor = base_tensor.cuda()
        
        # Ensure same size (resize if needed)
        if source_tensor.shape != base_tensor.shape:
            print(f"Warning: Size mismatch for {filename}: {source_tensor.shape} vs {base_tensor.shape}, skipping...")
            continue
        
        # Compute metrics
        ssim_val = ssim(source_tensor, base_tensor).item()
        psnr_val = psnr(source_tensor, base_tensor).mean().item()
        # LPIPS expects input in range [-1, 1]
        lpips_val = lpips_fn(source_tensor.unsqueeze(0) * 2 - 1, base_tensor.unsqueeze(0) * 2 - 1).item()
        
        SSIM_views.append(ssim_val)
        PSNR_views.append(psnr_val)
        LPIPS_views.append(lpips_val)
        
        # Save side-by-side comparison
        source_np = torch.clamp(source_tensor, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
        base_np = torch.clamp(base_tensor, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(source_np)
        axes[0].set_title("Poisoned DS")
        axes[0].axis("off")
        axes[1].imshow(base_np)
        axes[1].set_title("Clean DS")
        axes[1].axis("off")
        fig.suptitle(f"SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f}, LPIPS: {lpips_val:.4f}")
        fig.tight_layout()
        save_path = os.path.join(comparison_dir, f"compare_{idx:04d}_{filename}")
        # Ensure valid extension
        if not save_path.lower().endswith('.png'):
            save_path = save_path.rsplit('.', 1)[0] + '.png'
        fig.savefig(save_path)
        plt.close(fig)
    
    if len(SSIM_views) == 0:
        print("No valid image pairs found for comparison!")
        return
    
    mean_SSIM = round(sum(SSIM_views) / len(SSIM_views), 4)
    mean_PSNR = round(sum(PSNR_views) / len(PSNR_views), 4)
    mean_LPIPS = round(sum(LPIPS_views) / len(LPIPS_views), 4)
    
    print(f"\n=== Dataset Comparison Results ===")
    print(f"Compared {len(SSIM_views)} image pairs")
    print(f"Mean SSIM: {mean_SSIM}")
    print(f"Mean PSNR: {mean_PSNR}")
    print(f"Mean LPIPS: {mean_LPIPS}")
    
    # Save log
    log_path = os.path.join(args.model_path, "benchmark_dataset_comparison.log")
    timestamp = datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(
            f"{timestamp}\npoisoned={args.source_path}\nbase={args.base_path}\n"
            f"num_images={len(SSIM_views)}\nmean_ssim={mean_SSIM}\nmean_psnr={mean_PSNR}\nmean_lpips={mean_LPIPS}\n"
            f"images_dir={args.dataset_comparison_dir}\n"
        )
    print(f"Logged metrics to {log_path}")

def benchmark_recon_quality(args):
    gaussians = GaussianModel(args.sh_degree)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.load_ply(args.model_path + '/victim_model.ply')
    os.makedirs(args.model_path, exist_ok=True)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    num_views = len(viewpoint_stack)
    os.makedirs(os.path.join(args.model_path, args.save_images_dir), exist_ok=True)
    
    # Initialize LPIPS model
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    
    SSIM_views = []
    PSNR_views = []
    LPIPS_views = []
    for camid, cam in enumerate(viewpoint_stack):
        gt_image = cam.original_image.cuda()
        render_image = render(cam, gaussians, pipe, background)["render"]
        SSIM_views.append(ssim(render_image, gt_image).item())
        PSNR_views.append(psnr(render_image, gt_image).mean().item())
        # LPIPS expects input in range [-1, 1]
        LPIPS_views.append(lpips_fn(render_image.unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1).item())
        # Save side-by-side comparison for every view
        gt_np = torch.clamp(gt_image, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
        render_np = torch.clamp(render_image, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gt_np)
        axes[0].set_title("GT (Poisoned DS)")
        axes[0].axis("off")
        axes[1].imshow(render_np)
        axes[1].set_title("Render")
        axes[1].axis("off")
        fig.tight_layout()
        save_path = os.path.join(args.model_path, args.save_images_dir, f"view_{camid:04d}.png")
        fig.savefig(save_path)
        plt.close(fig)
    mean_SSIM = round(sum(SSIM_views)/len(SSIM_views), 4)
    mean_PSNR = round(sum(PSNR_views)/len(PSNR_views), 4)
    mean_LPIPS = round(sum(LPIPS_views)/len(LPIPS_views), 4)
    print(f"Mean SSIM: {mean_SSIM}")
    print(f"Mean PSNR: {mean_PSNR}")
    print(f"Mean LPIPS: {mean_LPIPS}")
    log_path = os.path.join(args.model_path, "benchmark_recon_quality.log")
    timestamp = datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(
            f"{timestamp}\nviews={num_views}\nmean_ssim={mean_SSIM}\nmean_psnr={mean_PSNR}\nmean_lpips={mean_LPIPS}\nimages_dir={args.save_images_dir}\n"
        )
    print(f"Logged metrics to {log_path}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="3DGS Victim Benchmark")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--exp_runs", type=int, default=3)
    parser.add_argument("--save_images_dir", type=str, default="image_comparison")
    parser.add_argument("-b", "--base_path", type=str, default=None, 
                        help="Base dataset path for comparing source images against")
    parser.add_argument("--dataset_comparison_dir", type=str, default="dataset_comparison",
                        help="Directory to save dataset comparison images")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    benchmark_recon_quality(args)
    
    # Run dataset comparison if base path is provided
    benchmark_dataset_comparison(args)
    

    ## usage:
    # python benchmark.py -s [data path] -m [output path] --gpu [x]

# python custom_benchmark.py -s [source_data_path] -m [output_path] -b [base_data_path] --gpu [x]

# python ./victim/gaussian-splatting/custom_benchmark.py -s dataset/Nerf_Synthetic/da3_chair/ -m log/da3/Nerf_Synthetic/chair/exp_run_1 -b dataset/Nerf_Synthetic/chair