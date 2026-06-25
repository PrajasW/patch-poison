#!/bin/bash
#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH --partition u22
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --job-name=variants_poison
#SBATCH -w gnode073
#SBATCH --output=variants.log

# set -euo pipefail

export QT_QPA_PLATFORM=offscreen

echo "gnode: $(hostname)"
echo "==================================="

nvidia-smi

echo "loading modules"
module load u22/cuda/11.8


echo "loading conda environment"
source ~/.bashrc
conda activate poison_splat

# Change to workspace root
cd ~/patch-poison/

VARIANTS_DIR="/ssd_scratch/prajas/dataset/ns_variants"
CLEAN_ROOT="dataset/nerf_synthetic"
LOG_ROOT="/ssd_scratch/prajas/logs/ns_variants"
NUM_GPUS=4

python main/make_variants.py

mkdir -p "${LOG_ROOT}"

if [ ! -d "${VARIANTS_DIR}" ]; then
    echo "ERROR: Variants directory not found: ${VARIANTS_DIR}"
    exit 1
fi

if [ ! -d "${CLEAN_ROOT}" ]; then
    echo "ERROR: Clean dataset directory not found: ${CLEAN_ROOT}"
    exit 1
fi

VARIANTS=(
    chair/checkerboard_only
    chair/lines_only
    chair/lines_only_top_left
    chair/lines_only_bottom_right
    chair/circles_only
    chair/checkerboard_lines
    chair/lines_circles
    chair/checkerboard_circles
    chair/all_patterns

    drums/checkerboard_only
    drums/lines_only
    drums/lines_only_top_left
    drums/lines_only_bottom_right
    drums/circles_only
    drums/checkerboard_lines
    drums/lines_circles
    drums/checkerboard_circles
    drums/all_patterns

    ficus/checkerboard_only
    ficus/lines_only
    ficus/lines_only_top_left
    ficus/lines_only_bottom_right
    ficus/circles_only
    ficus/checkerboard_lines
    ficus/lines_circles
    ficus/checkerboard_circles
    ficus/all_patterns

    hotdog/checkerboard_only
    hotdog/lines_only
    hotdog/lines_only_top_left
    hotdog/lines_only_bottom_right
    hotdog/circles_only
    hotdog/checkerboard_lines
    hotdog/lines_circles
    hotdog/checkerboard_circles
    hotdog/all_patterns

    lego/checkerboard_only
    lego/lines_only
    lego/lines_only_top_left
    lego/lines_only_bottom_right
    lego/circles_only
    lego/checkerboard_lines
    lego/lines_circles
    lego/checkerboard_circles
    lego/all_patterns

    materials/checkerboard_only
    materials/lines_only
    materials/lines_only_top_left
    materials/lines_only_bottom_right
    materials/circles_only
    materials/checkerboard_lines
    materials/lines_circles
    materials/checkerboard_circles
    materials/all_patterns

    mic/checkerboard_only
    mic/lines_only
    mic/lines_only_top_left
    mic/lines_only_bottom_right
    mic/circles_only
    mic/checkerboard_lines
    mic/lines_circles
    mic/checkerboard_circles
    mic/all_patterns

    ship/checkerboard_only
    ship/lines_only
    ship/lines_only_top_left
    ship/lines_only_bottom_right
    ship/circles_only
    ship/checkerboard_lines
    ship/lines_circles
    ship/checkerboard_circles
    ship/all_patterns
)

echo "========================================"
echo "  Benchmark Pipeline for Pattern Variants"
echo "========================================"
echo "Variants directory: ${VARIANTS_DIR}"
echo "Number of variants: ${#VARIANTS[@]}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Variants: ${VARIANTS[*]}"
echo ""



# ============================
#   Function Definitions
# ============================

run_benchmark() {
    local gpu=$1
    local variant=$2
    local dataset_path="${VARIANTS_DIR}/${variant}"
    local log_path="${LOG_ROOT}/${variant}"

    echo ">>> [GPU ${gpu}] Benchmarking ${variant} ..."
    mkdir -p "${log_path}"
    python victim/gaussian-splatting/benchmark.py --gpu ${gpu} --exp_runs 3 \
        -s ${dataset_path}/ \
        -m ${log_path}/
    echo ">>> [GPU ${gpu}] Benchmarking completed for ${variant}"
}

run_eval_clean() {
    local gpu=$1
    local variant=$2
    local dataset_path="${VARIANTS_DIR}/${variant}"
    local log_path="${LOG_ROOT}/${variant}"
    local dataset_name="${variant%%/*}"
    local base_dataset="${CLEAN_ROOT}/${dataset_name}"

    echo ">>> [GPU ${gpu}] Evaluating ${variant} on clean data ..."
    python victim/gaussian-splatting/custom_benchmark.py --gpu ${gpu} \
        -s ${dataset_path}/ \
        -m ${log_path}/exp_run_1 \
        -b ${base_dataset}/
    python victim/gaussian-splatting/custom_benchmark.py --gpu ${gpu} \
        -s ${dataset_path}/ \
        -m ${log_path}/exp_run_2 \
        -b ${base_dataset}/
    python victim/gaussian-splatting/custom_benchmark.py --gpu ${gpu} \
        -s ${dataset_path}/ \
        -m ${log_path}/exp_run_3 \
        -b ${base_dataset}/

    echo ">>> [GPU ${gpu}] Evaluation completed for ${variant}"
}

# ============================
#   STEP 1: COLMAP Processing (Sequential)
# ============================

echo "########################################"
echo "  STEP 1: COLMAP Processing (Sequential)"
echo "########################################"


COLMAP_DONE=0
for VARIANT in "${VARIANTS[@]}"; do
    DATASET_PATH="${VARIANTS_DIR}/${VARIANT}"

    echo ""
    echo "=== Running COLMAP for ${VARIANT} ==="

    if [ ! -d "${DATASET_PATH}" ]; then
        echo "WARNING: Dataset not found at ${DATASET_PATH}. Skipping..."
        continue
    fi

    cd "${DATASET_PATH}"

    if [ -d "train" ] && [ ! -e "images" ]; then
        echo "Creating images -> train symlink for COLMAP processing..."
        ln -s train images
    fi

    if [ -d "sparse/0" ]; then
        echo "COLMAP output already exists (sparse/0 found). Skipping COLMAP..."
    else
        echo "Running COLMAP feature extraction..."
        rm -f database.db

        colmap feature_extractor \
            --database_path database.db \
            --image_path images \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model SIMPLE_PINHOLE \
            --SiftExtraction.use_gpu 0

        echo "Running COLMAP feature matching..."
        colmap exhaustive_matcher \
            --database_path database.db \
            --SiftMatching.use_gpu 0

        echo "Running COLMAP sparse reconstruction..."
        mkdir -p sparse
        colmap mapper \
            --database_path database.db \
            --image_path images \
            --output_path sparse

        echo "Converting COLMAP model to TXT format..."
        colmap model_converter \
            --input_path sparse/0 \
            --output_path sparse/0 \
            --output_type TXT

        echo "COLMAP processing completed for ${VARIANT}!"
    fi

    cd ~/patch-poison/

    COLMAP_DONE=$((COLMAP_DONE + 1))
done

echo ""
echo "=== All COLMAP processing completed ==="

# ============================
#   STEP 2: Benchmarking (Multi-GPU Parallel)
# ============================

echo ""
echo "########################################"
echo "  STEP 2: Benchmarking (Multi-GPU Parallel)"
echo "########################################"

TOTAL=${#VARIANTS[@]}
BATCHES=$(((TOTAL + NUM_GPUS - 1) / NUM_GPUS))

for ((i=0; i<$TOTAL; i+=NUM_GPUS)); do
    batch=$((i/NUM_GPUS + 1))
    echo "---- Starting benchmark batch ${batch} ----"
    for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
        idx=$((i+gpu))
        if [ $idx -lt $TOTAL ]; then
            run_benchmark $gpu "${VARIANTS[$idx]}" &
        fi
    done
    wait
done

echo "=== All benchmarking runs completed ==="

# ============================
#   STEP 3: Evaluation on Clean Data (Multi-GPU Parallel)
# ============================

echo ""
echo "########################################"
echo "  STEP 3: Evaluation on Clean Data (Multi-GPU Parallel)"
echo "########################################"


for ((i=0; i<$TOTAL; i+=NUM_GPUS)); do
    batch=$((i/NUM_GPUS + 1))
    echo "---- Starting evaluation batch ${batch} ----"
    for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
        idx=$((i+gpu))
        if [ $idx -lt $TOTAL ]; then
            run_eval_clean $gpu "${VARIANTS[$idx]}" &
        fi
    done
    wait
done

echo "=== All evaluations completed ==="

# ============================
#   Summary
# ============================

echo ""
echo "########################################"
echo "  All Variants Completed Successfully!"
echo "########################################"
echo ""
echo "Results saved to: ${LOG_ROOT}/"
echo ""
echo "Processed variants:"
for VARIANT in "${VARIANTS[@]}"; do
    echo "  - ${VARIANT} -> ${LOG_ROOT}/${VARIANT}/"
done
echo ""

