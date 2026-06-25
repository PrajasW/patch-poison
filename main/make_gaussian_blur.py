import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# NeRF synthetic scenes in this repository.
SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]

# Gaussian blur kernel sizes (must be odd).
KERNEL_SIZES = [3, 7, 11, 21]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_NUM_WORKERS = 100


def apply_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur with the given kernel size."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def process_scene_train(
    src_scene: Path,
    dst_scene: Path,
    kernel_size: int,
    num_workers: int,
    progress_desc: str,
) -> int:
    """Apply Gaussian blur to all train images of one scene."""
    processed = 0
    dst_scene.mkdir(parents=True, exist_ok=True)
    src_train = src_scene / "train"
    dst_train = dst_scene / "train"

    if not src_train.exists():
        print(f"Warning: train directory not found for scene at {src_scene}")
        return 0

    dst_train.mkdir(parents=True, exist_ok=True)

    def _process_image(image_path: Path, output_path: Path) -> bool:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: failed to read image {image_path}")
            return False

        blurred = apply_gaussian_blur(image, kernel_size)
        return cv2.imwrite(str(output_path), blurred)

    image_files = [
        p for p in sorted(src_train.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(
            lambda p: _process_image(p, dst_train / p.name),
            image_files,
        )
        for ok in tqdm(results, total=len(image_files), desc=progress_desc, leave=False):
            if ok:
                processed += 1

    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gaussian-blur baseline datasets for all scenes."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel image workers per scene/variant (default: {DEFAULT_NUM_WORKERS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_workers = max(1, args.num_workers)

    input_root = Path("~/patch-poison/dataset/nerf_synthetic").expanduser()
    output_root = Path("/ssd_scratch/prajas/dataset/ns_gaussian_blur")

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Generating Gaussian-blur baseline datasets")
    print("=" * 72)
    print(f"Input root    : {input_root}")
    print(f"Output root   : {output_root}")
    print(f"Scenes        : {SCENES}")
    print(f"Kernel sizes  : {KERNEL_SIZES}")
    print(f"Workers       : {num_workers}")
    print("=" * 72)

    total_images = 0
    total_variants = 0
    valid_scenes = [scene for scene in SCENES if (input_root / scene).exists()]
    variant_total = len(valid_scenes) * len(KERNEL_SIZES)
    variant_pbar = tqdm(total=variant_total, desc="Variants", position=0)

    for scene in SCENES:
        src_scene = input_root / scene
        if not src_scene.exists():
            print(f"Skipping scene '{scene}' (not found at {src_scene})")
            continue

        for kernel_size in KERNEL_SIZES:
            dst_scene = output_root / scene / f"kernel_{kernel_size}"
            dst_scene.mkdir(parents=True, exist_ok=True)

            processed = process_scene_train(
                src_scene,
                dst_scene,
                kernel_size,
                num_workers=num_workers,
                progress_desc=f"{scene} kernel={kernel_size}",
            )
            total_images += processed
            total_variants += 1
            variant_pbar.update(1)

            print(
                f"[{scene}] kernel_{kernel_size}: processed {processed} train images -> {dst_scene}"
            )

    variant_pbar.close()

    print("=" * 72)
    print(f"Completed. Variants created: {total_variants}")
    print(f"Total processed train images: {total_images}")
    print(f"Output saved under: {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
