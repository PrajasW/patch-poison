import torch
import numpy as np
import os
import sys
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import torchvision
from tqdm import tqdm
from datetime import datetime


def render_views(viewpoint_stack, gaussians, pipe, background, output_dir, prefix=""):
    """Render all views from the viewpoint stack and save images."""
    os.makedirs(output_dir, exist_ok=True)

    for idx, cam in enumerate(tqdm(viewpoint_stack, desc=f"Rendering {prefix} views")):
        rendering = render(cam, gaussians, pipe, background)["render"]
        rendering = torch.clamp(rendering, 0.0, 1.0)

        save_path = os.path.join(output_dir, f"{prefix}_{idx:05d}.png")
        torchvision.utils.save_image(rendering, save_path)

    print(f"Saved {len(viewpoint_stack)} {prefix} renders to {output_dir}")


def render_model(args):
    """Load a single model and render from all camera angles."""
    # Extract params
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # Load model
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    # Load the PLY file
    ply_path = args.ply_path
    if not ply_path:
        # Default: look for victim_model.ply in model_path
        ply_path = os.path.join(args.model_path, "victim_model.ply")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    print(f"Loading model from: {ply_path}")
    gaussians.load_ply(ply_path)

    # Setup background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Setup output directory
    output_dir = args.output_path if args.output_path else "render_output"
    os.makedirs(output_dir, exist_ok=True)

    total_rendered = 0

    with torch.no_grad():
        # Render train cameras
        if not args.skip_train:
            train_cameras = scene.getTrainCameras().copy()
            train_dir = os.path.join(output_dir, "train")
            render_views(train_cameras, gaussians, pipe, background, train_dir, prefix="train")
            total_rendered += len(train_cameras)

        # Render test cameras
        if not args.skip_test:
            test_cameras = scene.getTestCameras().copy()
            test_dir = os.path.join(output_dir, "test")
            render_views(test_cameras, gaussians, pipe, background, test_dir, prefix="test")
            total_rendered += len(test_cameras)

    # Log results
    log_path = os.path.join(output_dir, "render_log.txt")
    timestamp = datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(
            f"{timestamp}\tply={ply_path}\tsource={dataset.source_path}\t"
            f"total_rendered={total_rendered}\tskip_train={args.skip_train}\t"
            f"skip_test={args.skip_test}\n"
        )

    print(f"\n{'='*60}")
    print(f"Rendering Complete")
    print(f"{'='*60}")
    print(f"PLY file:        {ply_path}")
    print(f"Source data:     {dataset.source_path}")
    print(f"Output dir:      {output_dir}")
    print(f"Total rendered:  {total_rendered} views")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render a 3DGS model from all camera angles")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ply_path", type=str, default=None,
                        help="Path to the .ply model file. If not provided, defaults to <model_path>/victim_model.ply")
    parser.add_argument("--output_path", type=str, default="render_output",
                        help="Output directory for rendered images")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip rendering train camera views")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip rendering test camera views")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    safe_state(args.quiet)

    print(f"Render Model Script")
    print(f"  PLY path:    {args.ply_path or '<model_path>/victim_model.ply'}")
    print(f"  Source data: {args.source_path}")
    print(f"  Output:      {args.output_path}")
    print(f"  Skip train:  {args.skip_train}")
    print(f"  Skip test:   {args.skip_test}")
    print(f"  GPU:         {args.gpu}")
    print()

    render_model(args)


    ## Usage:
    # python render_model.py -s [data path] -m [model path] --output_path [output dir] --gpu [x]
    #
    # Examples:
    #
    # Render all (train + test) camera views:
    #   python victim/gaussian-splatting/render_model.py \
    #       -s dataset/Nerf_Synthetic/chair_colmap/ \
    #       -m log/colmap/Nerf_Synthetic/chair/exp_run_1 \
    #       --output_path render_results/chair \
    #       --gpu 0
    #
    # Render only test camera views with a custom PLY file:
    #   python victim/gaussian-splatting/render_model.py \
    #       -s dataset/Nerf_Synthetic/chair_colmap/ \
    #       -m log/colmap/Nerf_Synthetic/chair/exp_run_1 \
    #       --ply_path /path/to/custom_model.ply \
    #       --output_path render_results/chair_test \
    #       --skip_train \
    #       --gpu 0
    #
    # Render only train camera views:
    #   python victim/gaussian-splatting/render_model.py \
    #       -s dataset/Nerf_Synthetic/chair_colmap/ \
    #       -m log/colmap/Nerf_Synthetic/chair/exp_run_1 \
    #       --output_path render_results/chair_train \
    #       --skip_test \
    #       --gpu 0
