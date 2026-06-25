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

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_NUM_WORKERS = 100

# ---------------------------------------------------------------------------
# Shear configurations: (shear_x, shear_y)  — None = random per-image
#   - x-only, y-only, both, random
# ---------------------------------------------------------------------------
SHEAR_CONFIGS = {
    "shear_x_0.2":    (0.2, 0.0),
    "shear_y_0.2":    (0.0, 0.2),
    "shear_xy_0.15":  (0.15, 0.15),
    "shear_random":   (None, None),  # uniform random in [-0.3, 0.3) per-image
}

# ---------------------------------------------------------------------------
# Rotation configurations: angle in degrees (None = random per-image)
# ---------------------------------------------------------------------------
ROTATION_CONFIGS = {
    "rotate_15":      15.0,
    "rotate_30":      30.0,
    "rotate_45":      45.0,
    "rotate_random":  None,   # uniform random in [-45, 45]
}


# ===== Transform functions ====================================================

def apply_shear(image: np.ndarray, shear_x: float, shear_y: float) -> np.ndarray:
    """Apply an affine shear transform, keeping the original image size.

    shear_x and shear_y are shear factors (e.g. 0.2).
    The output is filled with black (0) wherever pixels are missing.
    """
    h, w = image.shape[:2]
    # Affine matrix: [[1, shear_x, 0], [shear_y, 1, 0]]
    M = np.float32([[1, shear_x, 0],
                     [shear_y, 1, 0]])
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the image around its centre by *angle* degrees.

    The output canvas is the same size as the input; regions outside the
    rotated image are filled with black (0).
    """
    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


# ===== Scene processing ======================================================

def process_scene_shear(
    src_scene: Path,
    dst_scene: Path,
    shear_x: float | None,
    shear_y: float | None,
    num_workers: int,
    progress_desc: str,
) -> int:
    """Apply shear to all train images of one scene.

    If *shear_x* or *shear_y* is None, a uniform random value in
    [-0.3, 0.3) is sampled independently for each image.
    """
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
        sx = shear_x if shear_x is not None else np.random.uniform(-0.3, 0.3)
        sy = shear_y if shear_y is not None else np.random.uniform(-0.3, 0.3)
        transformed = apply_shear(image, sx, sy)
        return cv2.imwrite(str(output_path), transformed)

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


def process_scene_rotation(
    src_scene: Path,
    dst_scene: Path,
    angle: float | None,
    num_workers: int,
    progress_desc: str,
) -> int:
    """Apply rotation to all train images of one scene.

    If *angle* is None a uniform random angle in [0, 360) is sampled
    independently for each image.
    """
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
        a = angle if angle is not None else np.random.uniform(-45, 45)
        transformed = apply_rotation(image, a)
        return cv2.imwrite(str(output_path), transformed)

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


# ===== CLI ====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate geometric-transform (shear + rotation) baseline datasets."
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
    output_root = Path("/ssd_scratch/prajas/dataset/ns_geometric")

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    # Total variants = shear configs + rotation configs
    total_shear = len(SHEAR_CONFIGS)
    total_rotation = len(ROTATION_CONFIGS)
    valid_scenes = [s for s in SCENES if (input_root / s).exists()]
    variant_total = len(valid_scenes) * (total_shear + total_rotation)

    print("=" * 72)
    print("Generating geometric-transform baseline datasets")
    print("=" * 72)
    print(f"Input root        : {input_root}")
    print(f"Output root       : {output_root}")
    print(f"Scenes            : {SCENES}")
    print(f"Shear configs     : {list(SHEAR_CONFIGS.keys())}")
    print(f"Rotation configs  : {list(ROTATION_CONFIGS.keys())}")
    print(f"Workers           : {num_workers}")
    print("=" * 72)

    total_images = 0
    total_variants = 0
    variant_pbar = tqdm(
        total=len(valid_scenes) * (total_shear + total_rotation),
        desc="Variants",
        position=0,
    )

    for scene in SCENES:
        src_scene = input_root / scene
        if not src_scene.exists():
            print(f"Skipping scene '{scene}' (not found at {src_scene})")
            continue

        # -- Shear variants --
        for name, (sx, sy) in SHEAR_CONFIGS.items():
            dst_scene = output_root / scene / name
            dst_scene.mkdir(parents=True, exist_ok=True)

            processed = process_scene_shear(
                src_scene,
                dst_scene,
                sx, sy,
                num_workers=num_workers,
                progress_desc=f"{scene} {name}",
            )
            total_images += processed
            total_variants += 1
            variant_pbar.update(1)

            print(
                f"[{scene}] {name}: processed {processed} train images -> {dst_scene}"
            )

        # -- Rotation variants --
        for name, angle in ROTATION_CONFIGS.items():
            dst_scene = output_root / scene / name
            dst_scene.mkdir(parents=True, exist_ok=True)

            processed = process_scene_rotation(
                src_scene,
                dst_scene,
                angle,
                num_workers=num_workers,
                progress_desc=f"{scene} {name}",
            )
            total_images += processed
            total_variants += 1
            variant_pbar.update(1)

            print(
                f"[{scene}] {name}: processed {processed} train images -> {dst_scene}"
            )

    variant_pbar.close()

    print("=" * 72)
    print(f"Completed. Variants created: {total_variants}")
    print(f"Total processed train images: {total_images}")
    print(f"Output saved under: {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
