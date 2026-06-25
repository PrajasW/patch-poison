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

# Fixed patch size and block size.
PATCH_SIZE = 100
BLOCK_SIZE = 4

# White intensity values to test (black stays at 0).
WHITE_VALUES = [10, 25, 50, 75, 100, 150, 200, 255]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_NUM_WORKERS = 100


def create_checkerboard(size: int, block_size: int, white_value: int) -> np.ndarray:
    """Create an RGB checkerboard patch with a custom white intensity.

    The 'black' squares are always (0, 0, 0).
    The 'white' squares are (white_value, white_value, white_value).
    """
    patch = np.zeros((size, size, 3), dtype=np.uint8)

    for y in range(0, size, block_size):
        for x in range(0, size, block_size):
            if ((y // block_size) + (x // block_size)) % 2 == 0:
                patch[y:y + block_size, x:x + block_size] = (
                    white_value, white_value, white_value
                )

    return patch


def overlay_patch_bottom_left(image: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """Overlay patch on the bottom-left corner with safe clipping."""
    output = image.copy()
    h, w = output.shape[:2]
    ph, pw = patch.shape[:2]

    start_y = h - ph
    start_x = 0

    y1 = max(0, start_y)
    x1 = max(0, start_x)
    y2 = min(h, start_y + ph)
    x2 = min(w, start_x + pw)

    if y1 >= y2 or x1 >= x2:
        return output

    py1 = y1 - start_y
    px1 = x1 - start_x
    py2 = py1 + (y2 - y1)
    px2 = px1 + (x2 - x1)

    output[y1:y2, x1:x2] = patch[py1:py2, px1:px2]
    return output


def copy_scene_with_patched_train(
    src_scene: Path,
    dst_scene: Path,
    patch: np.ndarray,
    num_workers: int,
    progress_desc: str,
) -> int:
    """Write poisoned train images for one scene into dst_scene/train/."""
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

        patched = overlay_patch_bottom_left(image, patch)
        return cv2.imwrite(str(output_path), patched)

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
        description="Generate checkerboard PatchPoison datasets with varying colour contrast."
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
    output_root = Path("/ssd_scratch/prajas/dataset/ns_contrast")

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Generating colour-contrast checkerboard PatchPoison datasets")
    print("=" * 72)
    print(f"Input root    : {input_root}")
    print(f"Output root   : {output_root}")
    print(f"Scenes        : {SCENES}")
    print(f"Patch size    : {PATCH_SIZE}")
    print(f"Block size    : {BLOCK_SIZE}")
    print(f"White values  : {WHITE_VALUES}")
    print(f"Workers       : {num_workers}")
    print("=" * 72)

    total_images = 0
    total_variants = 0
    valid_scenes = [scene for scene in SCENES if (input_root / scene).exists()]
    variant_total = len(valid_scenes) * len(WHITE_VALUES)
    variant_pbar = tqdm(total=variant_total, desc="Variants", position=0)

    for scene in SCENES:
        src_scene = input_root / scene
        if not src_scene.exists():
            print(f"Skipping scene '{scene}' (not found at {src_scene})")
            continue

        for white_val in WHITE_VALUES:
            patch = create_checkerboard(
                size=PATCH_SIZE, block_size=BLOCK_SIZE, white_value=white_val
            )
            dst_scene = output_root / scene / f"white_{white_val}"
            dst_scene.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(dst_scene / "checkerboard_patch.png"), patch)

            processed = copy_scene_with_patched_train(
                src_scene,
                dst_scene,
                patch,
                num_workers=num_workers,
                progress_desc=f"{scene} white={white_val}",
            )
            total_images += processed
            total_variants += 1
            variant_pbar.update(1)

            print(
                f"[{scene}] white_{white_val}: patched {processed} train images -> {dst_scene}"
            )

    variant_pbar.close()

    print("=" * 72)
    print(f"Completed. Variants created: {total_variants}")
    print(f"Total patched train images: {total_images}")
    print(f"Output saved under: {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
