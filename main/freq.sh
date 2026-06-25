#!/bin/bash
#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH --partition u22
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --job-name=crazy_research_work
#SBATCH -w gnode068
#SBATCH --output=freq.log

set -euo pipefail

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

VARIANTS_DIR="/ssd_scratch/prajas/dataset/ns_freq"
CLEAN_ROOT="dataset/nerf_synthetic"
LOG_ROOT="/ssd_scratch/prajas/logs/ns_freq"
NUM_GPUS=4

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
    chair/block_size_1
    chair/block_size_2
    chair/block_size_4
    chair/block_size_6
    chair/block_size_8
    chair/block_size_10
    chair/block_size_16
    chair/block_size_25
    chair/block_size_50

    drums/block_size_1
    drums/block_size_2
    drums/block_size_4
    drums/block_size_6
    drums/block_size_8
    drums/block_size_10
    drums/block_size_16
    drums/block_size_25
    drums/block_size_50

    ficus/block_size_1
    ficus/block_size_2
    ficus/block_size_4
    ficus/block_size_6
    ficus/block_size_8
    ficus/block_size_10
    ficus/block_size_16
    ficus/block_size_25
    ficus/block_size_50

    hotdog/block_size_1
    hotdog/block_size_2
    hotdog/block_size_4
    hotdog/block_size_6
    hotdog/block_size_8
    hotdog/block_size_10
    hotdog/block_size_16
    hotdog/block_size_25
    hotdog/block_size_50

    lego/block_size_1
    lego/block_size_2
    lego/block_size_4
    lego/block_size_6
    lego/block_size_8
    lego/block_size_10
    lego/block_size_16
    lego/block_size_25
    lego/block_size_50

    materials/block_size_1
    materials/block_size_2
    materials/block_size_4
    materials/block_size_6
    materials/block_size_8
    materials/block_size_10
    materials/block_size_16
    materials/block_size_25
    materials/block_size_50

    mic/block_size_1
    mic/block_size_2
    mic/block_size_4
    mic/block_size_6
    mic/block_size_8
    mic/block_size_10
    mic/block_size_16
    mic/block_size_25
    mic/block_size_50

    ship/block_size_1
    ship/block_size_2
    ship/block_size_4
    ship/block_size_6
    ship/block_size_8
    ship/block_size_10
    ship/block_size_16
    ship/block_size_25
    ship/block_size_50
)

echo "========================================"
echo "  Benchmark Pipeline for PatchPoison Block Sizes"
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

