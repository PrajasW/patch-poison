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

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_NUM_WORKERS = 100


# ─────────────────────────────────────────────────────────────
#  Pattern creation helpers  (matching patch_attack_variants.py)
# ─────────────────────────────────────────────────────────────

def create_checkerboard(patch, size, block_size=4):
    """Create a fine checkerboard pattern (high frequency)."""
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                patch[i:i+block_size, j:j+block_size] = [255, 255, 255]
            else:
                patch[i:i+block_size, j:j+block_size] = [0, 0, 0]
    return patch


def create_diagonal_lines(patch, size):
    """Add diagonal lines for more texture."""
    for i in range(0, size, 8):
        cv2.line(patch, (0, i), (i, 0), (128, 128, 128), 1)
        cv2.line(patch, (size-1, i), (size-1-i, size-1), (128, 128, 128), 1)
    return patch


def create_diagonal_lines_top_left(patch, size):
    """Add diagonal lines from top-left corner only."""
    for i in range(0, size, 8):
        cv2.line(patch, (0, i), (i, 0), (128, 128, 128), 1)
    return patch


def create_diagonal_lines_bottom_right(patch, size):
    """Add diagonal lines from bottom-right corner only."""
    for i in range(0, size, 8):
        cv2.line(patch, (size-1, i), (size-1-i, size-1), (128, 128, 128), 1)
    return patch


def create_circles(patch, size):
    """Add circles for blob detection."""
    for i in range(10, size-10, 20):
        for j in range(10, size-10, 20):
            cv2.circle(patch, (i, j), 3, (200, 200, 200), -1)
    return patch


def create_patch_variant(
    size,
    use_checkerboard=False,
    use_lines=False,
    use_circles=False,
    use_lines_top_left=False,
    use_lines_bottom_right=False,
):
    """Create a patch with specified pattern combinations."""
    patch = np.zeros((size, size, 3), dtype=np.uint8)

    if use_checkerboard:
        patch = create_checkerboard(patch, size, BLOCK_SIZE)
    if use_lines:
        patch = create_diagonal_lines(patch, size)
    if use_lines_top_left:
        patch = create_diagonal_lines_top_left(patch, size)
    if use_lines_bottom_right:
        patch = create_diagonal_lines_bottom_right(patch, size)
    if use_circles:
        patch = create_circles(patch, size)

    return patch


# The 9 pattern variants (same as patch_attack_variants.py)
PATCH_VARIANTS = {
    'checkerboard_only': {
        'use_checkerboard': True,
        'use_lines': False,
        'use_circles': False,
        'description': 'Checkerboard pattern only'
    },
    'lines_only': {
        'use_checkerboard': False,
        'use_lines': True,
        'use_circles': False,
        'description': 'Diagonal lines only'
    },
    'lines_only_top_left': {
        'use_checkerboard': False,
        'use_lines': False,
        'use_circles': False,
        'use_lines_top_left': True,
        'use_lines_bottom_right': False,
        'description': 'Diagonal lines from top-left corner only'
    },
    'lines_only_bottom_right': {
        'use_checkerboard': False,
        'use_lines': False,
        'use_circles': False,
        'use_lines_top_left': False,
        'use_lines_bottom_right': True,
        'description': 'Diagonal lines from bottom-right corner only'
    },
    'circles_only': {
        'use_checkerboard': False,
        'use_lines': False,
        'use_circles': True,
        'description': 'Circles only'
    },
    'checkerboard_lines': {
        'use_checkerboard': True,
        'use_lines': True,
        'use_circles': False,
        'description': 'Checkerboard + Diagonal lines'
    },
    'lines_circles': {
        'use_checkerboard': False,
        'use_lines': True,
        'use_circles': True,
        'description': 'Diagonal lines + Circles'
    },
    'checkerboard_circles': {
        'use_checkerboard': True,
        'use_lines': False,
        'use_circles': True,
        'description': 'Checkerboard + Circles'
    },
    'all_patterns': {
        'use_checkerboard': True,
        'use_lines': True,
        'use_circles': True,
        'description': 'Checkerboard + Diagonal lines + Circles'
    },
}


# ─────────────────────────────────────────────────
#  Image overlay & dataset helpers
# ─────────────────────────────────────────────────

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
        description="Generate PatchPoison datasets for all pattern variants and all scenes."
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
    output_root = Path("/ssd_scratch/prajas/dataset/ns_variants")

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    variant_names = list(PATCH_VARIANTS.keys())

    print("=" * 72)
    print("Generating pattern-variant PatchPoison datasets (all scenes)")
    print("=" * 72)
    print(f"Input root      : {input_root}")
    print(f"Output root     : {output_root}")
    print(f"Scenes          : {SCENES}")
    print(f"Patch size      : {PATCH_SIZE}")
    print(f"Block size      : {BLOCK_SIZE}")
    print(f"Pattern variants: {variant_names}")
    print(f"Workers         : {num_workers}")
    print("=" * 72)

    total_images = 0
    total_variants = 0
    valid_scenes = [scene for scene in SCENES if (input_root / scene).exists()]
    variant_total = len(valid_scenes) * len(PATCH_VARIANTS)
    variant_pbar = tqdm(total=variant_total, desc="Variants", position=0)

    for scene in SCENES:
        src_scene = input_root / scene
        if not src_scene.exists():
            print(f"Skipping scene '{scene}' (not found at {src_scene})")
            continue

        for variant_name, variant_config in PATCH_VARIANTS.items():
            patch = create_patch_variant(
                size=PATCH_SIZE,
                use_checkerboard=variant_config.get('use_checkerboard', False),
                use_lines=variant_config.get('use_lines', False),
                use_circles=variant_config.get('use_circles', False),
                use_lines_top_left=variant_config.get('use_lines_top_left', False),
                use_lines_bottom_right=variant_config.get('use_lines_bottom_right', False),
            )

            dst_scene = output_root / scene / variant_name
            dst_scene.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(dst_scene / f"{variant_name}_patch.png"), patch)

            processed = copy_scene_with_patched_train(
                src_scene,
                dst_scene,
                patch,
                num_workers=num_workers,
                progress_desc=f"{scene} {variant_name}",
            )
            total_images += processed
            total_variants += 1
            variant_pbar.update(1)

            print(
                f"[{scene}] {variant_name}: {variant_config['description']} "
                f"- patched {processed} train images -> {dst_scene}"
            )

    variant_pbar.close()

    print("=" * 72)
    print(f"Completed. Variants created: {total_variants}")
    print(f"Total patched train images: {total_images}")
    print(f"Output saved under: {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
