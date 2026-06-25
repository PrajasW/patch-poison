import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


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

PATCH_SIZE = 100
BLOCK_SIZE = 4

POISON_RATIOS = [5, 10, 25, 50, 75, 100]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_NUM_WORKERS = 100


def create_checkerboard(size: int, block_size: int = 4) -> np.ndarray:
    """Create an RGB checkerboard patch."""
    patch = np.zeros((size, size, 3), dtype=np.uint8)

    for y in range(0, size, block_size):
        for x in range(0, size, block_size):
            if ((y // block_size) + (x // block_size)) % 2 == 0:
                patch[y:y + block_size, x:x + block_size] = (255, 255, 255)

    return patch


def overlay_patch_bottom_left(image: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """Overlay patch in bottom-left corner."""
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


def copy_scene_with_partial_poison(
    src_scene: Path,
    dst_scene: Path,
    patch: np.ndarray,
    ratio_pct: int,
    num_workers: int,
    progress_desc: str,
) -> tuple[int, int]:

    dst_scene.mkdir(parents=True, exist_ok=True)

    src_train = src_scene / "train"
    dst_train = dst_scene / "train"

    if not src_train.exists():
        print(f"Warning: train directory not found for scene at {src_scene}")
        return 0, 0

    dst_train.mkdir(parents=True, exist_ok=True)

    image_files = [
        p for p in sorted(src_train.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    total = len(image_files)

    if total == 0:
        return 0, 0

    num_to_poison = int(total * ratio_pct / 100)

    if ratio_pct == 100:
        num_to_poison = total

    poisoned_names = [p.name for p in image_files[:num_to_poison]]
    clean_names = [p.name for p in image_files[num_to_poison:]]
    print(f"  Poisoning {num_to_poison}/{total} images:")
    print(f"    Poisoned: {poisoned_names}")
    print(f"    Clean   : {clean_names}")

    def _process_image(args: tuple[int, Path, Path]) -> tuple[bool, bool]:

        idx, image_path, output_path = args

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: failed to read image {image_path}")
            return False, False

        if idx < num_to_poison:
            output = overlay_patch_bottom_left(image, patch)
            ok = cv2.imwrite(str(output_path), output)
            return ok, True
        else:
            ok = cv2.imwrite(str(output_path), image)
            return ok, False

    tasks = [
        (idx, p, dst_train / p.name)
        for idx, p in enumerate(image_files)
    ]

    num_poisoned = 0
    num_clean = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_process_image, tasks)

        for ok, was_poisoned in tqdm(
            results,
            total=len(tasks),
            desc=progress_desc,
            leave=False
        ):
            if ok:
                if was_poisoned:
                    num_poisoned += 1
                else:
                    num_clean += 1

    return num_poisoned, num_clean


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Generate partially poisoned checkerboard datasets"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of parallel workers"
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    num_workers = max(1, args.num_workers)

    input_root = Path("~/patch-poison/dataset/nerf_synthetic").expanduser()
    output_root = Path("/ssd_scratch/prajas/dataset/ns_ratio")

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Generating partially-poisoned checkerboard PatchPoison datasets")
    print("=" * 72)

    print(f"Input root     : {input_root}")
    print(f"Output root    : {output_root}")
    print(f"Scenes         : {SCENES}")
    print(f"Patch size     : {PATCH_SIZE}")
    print(f"Block size     : {BLOCK_SIZE}")
    print(f"Poison ratios  : {POISON_RATIOS}%")
    print(f"Workers        : {num_workers}")

    print("=" * 72)

    patch = create_checkerboard(size=PATCH_SIZE, block_size=BLOCK_SIZE)

    total_poisoned = 0
    total_clean = 0
    total_variants = 0

    valid_scenes = [scene for scene in SCENES if (input_root / scene).exists()]
    variant_total = len(valid_scenes) * len(POISON_RATIOS)

    variant_pbar = tqdm(total=variant_total, desc="Variants", position=0)

    for scene in SCENES:

        src_scene = input_root / scene

        if not src_scene.exists():
            print(f"Skipping scene '{scene}' (not found)")
            continue

        for ratio_pct in POISON_RATIOS:

            dst_scene = output_root / scene / f"poison_{ratio_pct}pct"
            dst_scene.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(dst_scene / "checkerboard_patch.png"), patch)

            num_poisoned, num_clean = copy_scene_with_partial_poison(
                src_scene,
                dst_scene,
                patch,
                ratio_pct=ratio_pct,
                num_workers=num_workers,
                progress_desc=f"{scene} ratio={ratio_pct}%"
            )

            total_poisoned += num_poisoned
            total_clean += num_clean
            total_variants += 1

            variant_pbar.update(1)

            print(
                f"[{scene}] poison_{ratio_pct}pct: "
                f"{num_poisoned} poisoned + {num_clean} clean "
                f"= {num_poisoned + num_clean} total -> {dst_scene}"
            )

    variant_pbar.close()

    print("=" * 72)
    print(f"Completed. Variants created: {total_variants}")
    print(f"Total poisoned train images : {total_poisoned}")
    print(f"Total clean train images    : {total_clean}")
    print(f"Output saved under          : {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()