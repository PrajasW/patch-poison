#!/bin/bash
#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH --partition u22
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --job-name=contrast_poison
#SBATCH -w gnode068
#SBATCH --output=contrast.log

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

VARIANTS_DIR="/ssd_scratch/prajas/dataset/ns_contrast"
CLEAN_ROOT="dataset/nerf_synthetic"
LOG_ROOT="/ssd_scratch/prajas/logs/ns_contrast"
NUM_GPUS=4

python main/make_contrast.py

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
    chair/white_10
    chair/white_25
    chair/white_50
    chair/white_75
    chair/white_100
    chair/white_150
    chair/white_200
    chair/white_255

    drums/white_10
    drums/white_25
    drums/white_50
    drums/white_75
    drums/white_100
    drums/white_150
    drums/white_200
    drums/white_255

    ficus/white_10
    ficus/white_25
    ficus/white_50
    ficus/white_75
    ficus/white_100
    ficus/white_150
    ficus/white_200
    ficus/white_255

    hotdog/white_10
    hotdog/white_25
    hotdog/white_50
    hotdog/white_75
    hotdog/white_100
    hotdog/white_150
    hotdog/white_200
    hotdog/white_255

    lego/white_10
    lego/white_25
    lego/white_50
    lego/white_75
    lego/white_100
    lego/white_150
    lego/white_200
    lego/white_255

    materials/white_10
    materials/white_25
    materials/white_50
    materials/white_75
    materials/white_100
    materials/white_150
    materials/white_200
    materials/white_255

    mic/white_10
    mic/white_25
    mic/white_50
    mic/white_75
    mic/white_100
    mic/white_150
    mic/white_200
    mic/white_255

    ship/white_10
    ship/white_25
    ship/white_50
    ship/white_75
    ship/white_100
    ship/white_150
    ship/white_200
    ship/white_255
)

echo "========================================"
echo "  Benchmark Pipeline for Colour Contrast"
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

