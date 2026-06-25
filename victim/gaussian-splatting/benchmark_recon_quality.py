#
# Copyright (C) 2024, Jiahao Lu @ Skywork AI
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use.
#
# For inquiries contact jiahao.lu@u.nus.edu

import torch
import numpy as np
import os
import sys
import random
from random import randint
import uuid
import time
import re
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import multiprocessing
from gpuinfo import GPUInfo
from datetime import datetime
import matplotlib.pyplot as plt

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
    SSIM_views = []
    PSNR_views = []
    for camid, cam in enumerate(viewpoint_stack):
        gt_image = cam.original_image.cuda()
        render_image = render(cam, gaussians, pipe, background)["render"]
        SSIM_views.append(ssim(render_image, gt_image).item())
        PSNR_views.append(psnr(render_image, gt_image).mean().item())
        # Save side-by-side comparison for every view
        gt_np = torch.clamp(gt_image, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
        render_np = torch.clamp(render_image, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gt_np)
        axes[0].set_title("GT")
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
    print(f"Mean SSIM: {mean_SSIM}")
    print(f"Mean PSNR: {mean_PSNR}")
    log_path = os.path.join(args.model_path, "benchmark_recon_quality.log")
    timestamp = datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(
            f"{timestamp}\tviews={num_views}\tmean_ssim={mean_SSIM}\tmean_psnr={mean_PSNR}\timages_dir={args.save_images_dir}\n"
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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    benchmark_recon_quality(args)
    

    ## usage:
    # python benchmark.py -s [data path] -m [output path] --gpu [x]