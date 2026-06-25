#!/bin/bash
#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH --partition u22
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --job-name=clean_baseline
#SBATCH -w gnode076
#SBATCH --output=clean.log

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

# ---- Paths ----
# We copy the original clean dataset to ssd_scratch so the pipeline
# (COLMAP, symlinks, etc.) can write into it without touching the
# canonical source under ~/patch-poison/dataset.
CLEAN_SRC="dataset/nerf_synthetic"
CLEAN_ROOT="/ssd_scratch/prajas/dataset/ns_clean"
LOG_ROOT="/ssd_scratch/prajas/logs/ns_clean"
NUM_GPUS=4

mkdir -p "${LOG_ROOT}"

if [ ! -d "${CLEAN_SRC}" ]; then
    echo "ERROR: Clean source dataset directory not found: ${CLEAN_SRC}"
    exit 1
fi

# ---- Scenes ----
SCENES=(
    chair
    drums
    ficus
    hotdog
    lego
    materials
    mic
    ship
)

echo "========================================"
echo "  Clean Baseline Pipeline (no patch)"
echo "========================================"
echo "Clean source  : ${CLEAN_SRC}"
echo "Working copy  : ${CLEAN_ROOT}"
echo "Log root      : ${LOG_ROOT}"
echo "Number of scenes : ${#SCENES[@]}"
echo "Number of GPUs   : ${NUM_GPUS}"
echo "Scenes: ${SCENES[*]}"
echo ""

# ============================
#   STEP 0: Copy clean data to scratch
# ============================

echo "########################################"
echo "  STEP 0: Preparing clean data on scratch"
echo "########################################"

for SCENE in "${SCENES[@]}"; do
    DST="${CLEAN_ROOT}/${SCENE}"
    SRC="${CLEAN_SRC}/${SCENE}"

    if [ ! -d "${SRC}" ]; then
        echo "WARNING: Scene '${SCENE}' not found at ${SRC}. Skipping copy..."
        continue
    fi

    if [ -d "${DST}/train" ]; then
        echo "Scene '${SCENE}' already exists at ${DST}. Skipping copy..."
        continue
    fi

    echo "Copying ${SRC}/train -> ${DST}/train ..."
    mkdir -p "${DST}"
    cp -r "${SRC}/train" "${DST}/train"
done

echo "=== Clean data preparation completed ==="

# ============================
#   Function Definitions
# ============================

run_benchmark() {
    local gpu=$1
    local scene=$2
    local dataset_path="${CLEAN_ROOT}/${scene}"
    local log_path="${LOG_ROOT}/${scene}"

    echo ">>> [GPU ${gpu}] Benchmarking ${scene} ..."
    mkdir -p "${log_path}"
    python victim/gaussian-splatting/benchmark.py --gpu ${gpu} --exp_runs 3 \
        -s ${dataset_path}/ \
        -m ${log_path}/
    echo ">>> [GPU ${gpu}] Benchmarking completed for ${scene}"
}

run_eval_clean() {
    local gpu=$1
    local scene=$2
    local dataset_path="${CLEAN_ROOT}/${scene}"
    local log_path="${LOG_ROOT}/${scene}"
    local base_dataset="${CLEAN_SRC}/${scene}"

    echo ">>> [GPU ${gpu}] Evaluating ${scene} on clean data ..."
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

    echo ">>> [GPU ${gpu}] Evaluation completed for ${scene}"
}

# ============================
#   STEP 1: COLMAP Processing (Sequential)
# ============================

echo ""
echo "########################################"
echo "  STEP 1: COLMAP Processing (Sequential)"
echo "########################################"

COLMAP_DONE=0
for SCENE in "${SCENES[@]}"; do
    DATASET_PATH="${CLEAN_ROOT}/${SCENE}"

    echo ""
    echo "=== Running COLMAP for ${SCENE} ==="

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

        echo "COLMAP processing completed for ${SCENE}!"
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

TOTAL=${#SCENES[@]}
BATCHES=$(((TOTAL + NUM_GPUS - 1) / NUM_GPUS))

for ((i=0; i<$TOTAL; i+=NUM_GPUS)); do
    batch=$((i/NUM_GPUS + 1))
    echo "---- Starting benchmark batch ${batch} ----"
    for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
        idx=$((i+gpu))
        if [ $idx -lt $TOTAL ]; then
            run_benchmark $gpu "${SCENES[$idx]}" &
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
            run_eval_clean $gpu "${SCENES[$idx]}" &
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
echo "  All Clean Baseline Scenes Completed!"
echo "########################################"
echo ""
echo "Results saved to: ${LOG_ROOT}/"
echo ""
echo "Processed scenes:"
for SCENE in "${SCENES[@]}"; do
    echo "  - ${SCENE} -> ${LOG_ROOT}/${SCENE}/"
done
echo ""

