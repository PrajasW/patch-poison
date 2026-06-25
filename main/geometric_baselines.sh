#!/bin/bash
#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH --partition u22
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --job-name=geo_baselines
#SBATCH -w gnode080
#SBATCH --output=geometric_baselines.log


set -euo pipefail

export QT_QPA_PLATFORM=offscreen
# export XDG_RUNTIME_DIR=/tmp/$USER-runtime
# mkdir -p $XDG_RUNTIME_DIR

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

CLEAN_ROOT="dataset/nerf_synthetic"
NUM_GPUS=4

# ================================================================
#  Dataset directories and log directories
# ================================================================

BLUR_DIR="/ssd_scratch/prajas/dataset/ns_gaussian_blur"
BLUR_LOG="/ssd_scratch/prajas/logs/ns_gaussian_blur"

JPEG_DIR="/ssd_scratch/prajas/dataset/ns_jpeg_compression"
JPEG_LOG="/ssd_scratch/prajas/logs/ns_jpeg_compression"

NOISE_DIR="/ssd_scratch/prajas/dataset/ns_gaussian_noise"
NOISE_LOG="/ssd_scratch/prajas/logs/ns_gaussian_noise"

GEO_DIR="/ssd_scratch/prajas/dataset/ns_geometric"
GEO_LOG="/ssd_scratch/prajas/logs/ns_geometric"

mkdir -p "${BLUR_LOG}" "${JPEG_LOG}" "${NOISE_LOG}" "${GEO_LOG}"

# ================================================================
#  All variants, explicitly listed.
#  Format: DATASET_DIR|LOG_DIR|scene/variant
# ================================================================

VARIANTS=(
    # =============================================
    #   Gaussian Blur  (kernel_3, 7, 11, 21)
    # =============================================

    "${BLUR_DIR}|${BLUR_LOG}|chair/kernel_3"
    "${BLUR_DIR}|${BLUR_LOG}|chair/kernel_7"
    "${BLUR_DIR}|${BLUR_LOG}|chair/kernel_11"
    "${BLUR_DIR}|${BLUR_LOG}|chair/kernel_21"

    "${BLUR_DIR}|${BLUR_LOG}|drums/kernel_3"
    "${BLUR_DIR}|${BLUR_LOG}|drums/kernel_7"
    "${BLUR_DIR}|${BLUR_LOG}|drums/kernel_11"
    "${BLUR_DIR}|${BLUR_LOG}|drums/kernel_21"

    "${BLUR_DIR}|${BLUR_LOG}|ficus/kernel_3"
    "${BLUR_DIR}|${BLUR_LOG}|ficus/kernel_7"
    "${BLUR_DIR}|${BLUR_LOG}|ficus/kernel_11"
    "${BLUR_DIR}|${BLUR_LOG}|ficus/kernel_21"

    "${BLUR_DIR}|${BLUR_LOG}|hotdog/kernel_3"
    "${BLUR_DIR}|${BLUR_LOG}|hotdog/kernel_7"
    "${BLUR_DIR}|${BLUR_LOG}|hotdog/kernel_11"
    "${BLUR_DIR}|${BLUR_LOG}|hotdog/kernel_21"

    "${BLUR_DIR}|${BLUR_LOG}|lego/kernel_3"
    "${BLUR_DIR}|${BLUR_LOG}|lego/kernel_7"
    "${BLUR_DIR}|${BLUR_LOG}|lego/kernel_11"
    "${BLUR_DIR}|${BLUR_LOG}|lego/kernel_21"

    "${BLUR_DIR}|${BLUR_LOG}|materials/kernel_3"
    "${BLUR_DIR}|${BLUR_LOG}|materials/kernel_7"
    "${BLUR_DIR}|${BLUR_LOG}|materials/kernel_11"
    "${BLUR_DIR}|${BLUR_LOG}|materials/kernel_21"

    "${BLUR_DIR}|${BLUR_LOG}|mic/kernel_3"
    "${BLUR_DIR}|${BLUR_LOG}|mic/kernel_7"
    "${BLUR_DIR}|${BLUR_LOG}|mic/kernel_11"
    "${BLUR_DIR}|${BLUR_LOG}|mic/kernel_21"

    "${BLUR_DIR}|${BLUR_LOG}|ship/kernel_3"
    "${BLUR_DIR}|${BLUR_LOG}|ship/kernel_7"
    "${BLUR_DIR}|${BLUR_LOG}|ship/kernel_11"
    "${BLUR_DIR}|${BLUR_LOG}|ship/kernel_21"

    # =============================================
    #   JPEG Compression  (quality_10, 25, 50, 75)
    # =============================================

    "${JPEG_DIR}|${JPEG_LOG}|chair/quality_10"
    "${JPEG_DIR}|${JPEG_LOG}|chair/quality_25"
    "${JPEG_DIR}|${JPEG_LOG}|chair/quality_50"
    "${JPEG_DIR}|${JPEG_LOG}|chair/quality_75"

    "${JPEG_DIR}|${JPEG_LOG}|drums/quality_10"
    "${JPEG_DIR}|${JPEG_LOG}|drums/quality_25"
    "${JPEG_DIR}|${JPEG_LOG}|drums/quality_50"
    "${JPEG_DIR}|${JPEG_LOG}|drums/quality_75"

    "${JPEG_DIR}|${JPEG_LOG}|ficus/quality_10"
    "${JPEG_DIR}|${JPEG_LOG}|ficus/quality_25"
    "${JPEG_DIR}|${JPEG_LOG}|ficus/quality_50"
    "${JPEG_DIR}|${JPEG_LOG}|ficus/quality_75"

    "${JPEG_DIR}|${JPEG_LOG}|hotdog/quality_10"
    "${JPEG_DIR}|${JPEG_LOG}|hotdog/quality_25"
    "${JPEG_DIR}|${JPEG_LOG}|hotdog/quality_50"
    "${JPEG_DIR}|${JPEG_LOG}|hotdog/quality_75"

    "${JPEG_DIR}|${JPEG_LOG}|lego/quality_10"
    "${JPEG_DIR}|${JPEG_LOG}|lego/quality_25"
    "${JPEG_DIR}|${JPEG_LOG}|lego/quality_50"
    "${JPEG_DIR}|${JPEG_LOG}|lego/quality_75"

    "${JPEG_DIR}|${JPEG_LOG}|materials/quality_10"
    "${JPEG_DIR}|${JPEG_LOG}|materials/quality_25"
    "${JPEG_DIR}|${JPEG_LOG}|materials/quality_50"
    "${JPEG_DIR}|${JPEG_LOG}|materials/quality_75"

    "${JPEG_DIR}|${JPEG_LOG}|mic/quality_10"
    "${JPEG_DIR}|${JPEG_LOG}|mic/quality_25"
    "${JPEG_DIR}|${JPEG_LOG}|mic/quality_50"
    "${JPEG_DIR}|${JPEG_LOG}|mic/quality_75"

    "${JPEG_DIR}|${JPEG_LOG}|ship/quality_10"
    "${JPEG_DIR}|${JPEG_LOG}|ship/quality_25"
    "${JPEG_DIR}|${JPEG_LOG}|ship/quality_50"
    "${JPEG_DIR}|${JPEG_LOG}|ship/quality_75"

    # =============================================
    #   Gaussian Noise  (stddev_5, 10, 25, 50)
    # =============================================

    "${NOISE_DIR}|${NOISE_LOG}|chair/stddev_5"
    "${NOISE_DIR}|${NOISE_LOG}|chair/stddev_10"
    "${NOISE_DIR}|${NOISE_LOG}|chair/stddev_25"
    "${NOISE_DIR}|${NOISE_LOG}|chair/stddev_50"

    "${NOISE_DIR}|${NOISE_LOG}|drums/stddev_5"
    "${NOISE_DIR}|${NOISE_LOG}|drums/stddev_10"
    "${NOISE_DIR}|${NOISE_LOG}|drums/stddev_25"
    "${NOISE_DIR}|${NOISE_LOG}|drums/stddev_50"

    "${NOISE_DIR}|${NOISE_LOG}|ficus/stddev_5"
    "${NOISE_DIR}|${NOISE_LOG}|ficus/stddev_10"
    "${NOISE_DIR}|${NOISE_LOG}|ficus/stddev_25"
    "${NOISE_DIR}|${NOISE_LOG}|ficus/stddev_50"

    "${NOISE_DIR}|${NOISE_LOG}|hotdog/stddev_5"
    "${NOISE_DIR}|${NOISE_LOG}|hotdog/stddev_10"
    "${NOISE_DIR}|${NOISE_LOG}|hotdog/stddev_25"
    "${NOISE_DIR}|${NOISE_LOG}|hotdog/stddev_50"

    "${NOISE_DIR}|${NOISE_LOG}|lego/stddev_5"
    "${NOISE_DIR}|${NOISE_LOG}|lego/stddev_10"
    "${NOISE_DIR}|${NOISE_LOG}|lego/stddev_25"
    "${NOISE_DIR}|${NOISE_LOG}|lego/stddev_50"

    "${NOISE_DIR}|${NOISE_LOG}|materials/stddev_5"
    "${NOISE_DIR}|${NOISE_LOG}|materials/stddev_10"
    "${NOISE_DIR}|${NOISE_LOG}|materials/stddev_25"
    "${NOISE_DIR}|${NOISE_LOG}|materials/stddev_50"

    "${NOISE_DIR}|${NOISE_LOG}|mic/stddev_5"
    "${NOISE_DIR}|${NOISE_LOG}|mic/stddev_10"
    "${NOISE_DIR}|${NOISE_LOG}|mic/stddev_25"
    "${NOISE_DIR}|${NOISE_LOG}|mic/stddev_50"

    "${NOISE_DIR}|${NOISE_LOG}|ship/stddev_5"
    "${NOISE_DIR}|${NOISE_LOG}|ship/stddev_10"
    "${NOISE_DIR}|${NOISE_LOG}|ship/stddev_25"
    "${NOISE_DIR}|${NOISE_LOG}|ship/stddev_50"

    # =============================================
    #   Geometric — Shear
    #   (shear_x_0.2, shear_y_0.2, shear_xy_0.15, shear_random)
    # =============================================

    "${GEO_DIR}|${GEO_LOG}|chair/shear_x_0.2"
    "${GEO_DIR}|${GEO_LOG}|chair/shear_y_0.2"
    "${GEO_DIR}|${GEO_LOG}|chair/shear_xy_0.15"
    "${GEO_DIR}|${GEO_LOG}|chair/shear_random"

    "${GEO_DIR}|${GEO_LOG}|drums/shear_x_0.2"
    "${GEO_DIR}|${GEO_LOG}|drums/shear_y_0.2"
    "${GEO_DIR}|${GEO_LOG}|drums/shear_xy_0.15"
    "${GEO_DIR}|${GEO_LOG}|drums/shear_random"

    "${GEO_DIR}|${GEO_LOG}|ficus/shear_x_0.2"
    "${GEO_DIR}|${GEO_LOG}|ficus/shear_y_0.2"
    "${GEO_DIR}|${GEO_LOG}|ficus/shear_xy_0.15"
    "${GEO_DIR}|${GEO_LOG}|ficus/shear_random"

    "${GEO_DIR}|${GEO_LOG}|hotdog/shear_x_0.2"
    "${GEO_DIR}|${GEO_LOG}|hotdog/shear_y_0.2"
    "${GEO_DIR}|${GEO_LOG}|hotdog/shear_xy_0.15"
    "${GEO_DIR}|${GEO_LOG}|hotdog/shear_random"

    "${GEO_DIR}|${GEO_LOG}|lego/shear_x_0.2"
    "${GEO_DIR}|${GEO_LOG}|lego/shear_y_0.2"
    "${GEO_DIR}|${GEO_LOG}|lego/shear_xy_0.15"
    "${GEO_DIR}|${GEO_LOG}|lego/shear_random"

    "${GEO_DIR}|${GEO_LOG}|materials/shear_x_0.2"
    "${GEO_DIR}|${GEO_LOG}|materials/shear_y_0.2"
    "${GEO_DIR}|${GEO_LOG}|materials/shear_xy_0.15"
    "${GEO_DIR}|${GEO_LOG}|materials/shear_random"

    "${GEO_DIR}|${GEO_LOG}|mic/shear_x_0.2"
    "${GEO_DIR}|${GEO_LOG}|mic/shear_y_0.2"
    "${GEO_DIR}|${GEO_LOG}|mic/shear_xy_0.15"
    "${GEO_DIR}|${GEO_LOG}|mic/shear_random"

    "${GEO_DIR}|${GEO_LOG}|ship/shear_x_0.2"
    "${GEO_DIR}|${GEO_LOG}|ship/shear_y_0.2"
    "${GEO_DIR}|${GEO_LOG}|ship/shear_xy_0.15"
    "${GEO_DIR}|${GEO_LOG}|ship/shear_random"

    # =============================================
    #   Geometric — Rotation
    #   (rotate_15, rotate_30, rotate_45, rotate_random)
    # =============================================

    "${GEO_DIR}|${GEO_LOG}|chair/rotate_15"
    "${GEO_DIR}|${GEO_LOG}|chair/rotate_30"
    "${GEO_DIR}|${GEO_LOG}|chair/rotate_45"
    "${GEO_DIR}|${GEO_LOG}|chair/rotate_random"

    "${GEO_DIR}|${GEO_LOG}|drums/rotate_15"
    "${GEO_DIR}|${GEO_LOG}|drums/rotate_30"
    "${GEO_DIR}|${GEO_LOG}|drums/rotate_45"
    "${GEO_DIR}|${GEO_LOG}|drums/rotate_random"

    "${GEO_DIR}|${GEO_LOG}|ficus/rotate_15"
    "${GEO_DIR}|${GEO_LOG}|ficus/rotate_30"
    "${GEO_DIR}|${GEO_LOG}|ficus/rotate_45"
    "${GEO_DIR}|${GEO_LOG}|ficus/rotate_random"

    "${GEO_DIR}|${GEO_LOG}|hotdog/rotate_15"
    "${GEO_DIR}|${GEO_LOG}|hotdog/rotate_30"
    "${GEO_DIR}|${GEO_LOG}|hotdog/rotate_45"
    "${GEO_DIR}|${GEO_LOG}|hotdog/rotate_random"

    "${GEO_DIR}|${GEO_LOG}|lego/rotate_15"
    "${GEO_DIR}|${GEO_LOG}|lego/rotate_30"
    "${GEO_DIR}|${GEO_LOG}|lego/rotate_45"
    "${GEO_DIR}|${GEO_LOG}|lego/rotate_random"

    "${GEO_DIR}|${GEO_LOG}|materials/rotate_15"
    "${GEO_DIR}|${GEO_LOG}|materials/rotate_30"
    "${GEO_DIR}|${GEO_LOG}|materials/rotate_45"
    "${GEO_DIR}|${GEO_LOG}|materials/rotate_random"

    "${GEO_DIR}|${GEO_LOG}|mic/rotate_15"
    "${GEO_DIR}|${GEO_LOG}|mic/rotate_30"
    "${GEO_DIR}|${GEO_LOG}|mic/rotate_45"
    "${GEO_DIR}|${GEO_LOG}|mic/rotate_random"

    "${GEO_DIR}|${GEO_LOG}|ship/rotate_15"
    "${GEO_DIR}|${GEO_LOG}|ship/rotate_30"
    "${GEO_DIR}|${GEO_LOG}|ship/rotate_45"
    "${GEO_DIR}|${GEO_LOG}|ship/rotate_random"
)

TOTAL=${#VARIANTS[@]}

echo "========================================"
echo "  Geometric Baselines Processing Pipeline"
echo "========================================"
echo "Total variants : ${TOTAL}"
echo "GPUs           : ${NUM_GPUS}"
echo ""

if [ ! -d "${CLEAN_ROOT}" ]; then
    echo "ERROR: Clean dataset directory not found: ${CLEAN_ROOT}"
    exit 1
fi

# ================================================================
#  Helper: send progress notification
# ================================================================


# ================================================================
#  Function Definitions
# ================================================================

run_benchmark() {
    local gpu=$1
    local dataset_path=$2
    local log_path=$3

    echo ">>> [GPU ${gpu}] Benchmarking ${dataset_path} ..."
    mkdir -p "${log_path}"
    python victim/gaussian-splatting/benchmark.py --gpu ${gpu} --exp_runs 3 \
        -s ${dataset_path}/ \
        -m ${log_path}/
    echo ">>> [GPU ${gpu}] Benchmarking completed for ${dataset_path}"
}

run_eval_clean() {
    local gpu=$1
    local dataset_path=$2
    local log_path=$3
    local variant=$4
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

# ================================================================
#   STEP 0: Generate datasets if they don't exist
# ================================================================

echo ""
echo "########################################"
echo "  STEP 0: Dataset Generation (if needed)"
echo "########################################"


if [ ! -d "${BLUR_DIR}" ] || [ -z "$(ls -A "${BLUR_DIR}" 2>/dev/null)" ]; then
    echo "Gaussian Blur dataset not found at ${BLUR_DIR}. Generating..."
    python main/make_gaussian_blur.py
    echo "Gaussian Blur dataset generation completed."
else
    echo "Gaussian Blur dataset already exists at ${BLUR_DIR}. Skipping generation."
fi

if [ ! -d "${JPEG_DIR}" ] || [ -z "$(ls -A "${JPEG_DIR}" 2>/dev/null)" ]; then
    echo "JPEG Compression dataset not found at ${JPEG_DIR}. Generating..."
    python main/make_jpeg_compression.py
    echo "JPEG Compression dataset generation completed."
else
    echo "JPEG Compression dataset already exists at ${JPEG_DIR}. Skipping generation."
fi

if [ ! -d "${NOISE_DIR}" ] || [ -z "$(ls -A "${NOISE_DIR}" 2>/dev/null)" ]; then
    echo "Gaussian Noise dataset not found at ${NOISE_DIR}. Generating..."
    python main/make_gaussian_noise.py
    echo "Gaussian Noise dataset generation completed."
else
    echo "Gaussian Noise dataset already exists at ${NOISE_DIR}. Skipping generation."
fi

if [ ! -d "${GEO_DIR}" ] || [ -z "$(ls -A "${GEO_DIR}" 2>/dev/null)" ]; then
    echo "Geometric transforms dataset not found at ${GEO_DIR}. Generating..."
    python main/make_geometric.py
    echo "Geometric transforms dataset generation completed."
else
    echo "Geometric transforms dataset already exists at ${GEO_DIR}. Skipping generation."
fi


# ================================================================
#   STEP 1: COLMAP Processing (Sequential)
# ================================================================

echo ""
echo "########################################"
echo "  STEP 1: COLMAP Processing (Sequential)"
echo "########################################"


COLMAP_DONE=0
COLMAP_FAILED=0
FAILED_VARIANTS=()

for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r DATA_DIR LOG_DIR VARIANT <<< "${entry}"
    DATASET_PATH="${DATA_DIR}/${VARIANT}"

    echo ""
    echo "=== Running COLMAP for ${VARIANT} (${DATA_DIR}) ==="

    if [ ! -d "${DATASET_PATH}" ]; then
        echo "WARNING: Dataset not found at ${DATASET_PATH}. Skipping..."
        COLMAP_FAILED=$((COLMAP_FAILED + 1))
        FAILED_VARIANTS+=("${VARIANT} (dataset not found)")
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
        # Run COLMAP in a subshell so failures don't kill the whole script
        if (
            set -e
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
        ); then
            echo "COLMAP processing completed for ${VARIANT}!"
        else
            echo "ERROR: COLMAP FAILED for ${VARIANT}! Continuing with next variant..."
            COLMAP_FAILED=$((COLMAP_FAILED + 1))
            FAILED_VARIANTS+=("${VARIANT}")
        fi
    fi

    cd ~/patch-poison/

    COLMAP_DONE=$((COLMAP_DONE + 1))
done

if [ ${COLMAP_FAILED} -gt 0 ]; then
    echo ""
    echo "WARNING: COLMAP failed for ${COLMAP_FAILED} variant(s):"
    for fv in "${FAILED_VARIANTS[@]}"; do
        echo "  - ${fv}"
    done
    echo "These variants will be skipped in benchmarking/evaluation if sparse/0 is missing."
fi

echo ""
echo "=== All COLMAP processing completed ==="

# ================================================================
#   STEP 2: Benchmarking (Multi-GPU Parallel)
# ================================================================

echo ""
echo "########################################"
echo "  STEP 2: Benchmarking (Multi-GPU Parallel)"
echo "########################################"

BATCHES=$(((TOTAL + NUM_GPUS - 1) / NUM_GPUS))

for ((i=0; i<$TOTAL; i+=NUM_GPUS)); do
    batch=$((i/NUM_GPUS + 1))
    echo "---- Starting benchmark batch ${batch} ----"
    for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
        idx=$((i+gpu))
        if [ $idx -lt $TOTAL ]; then
            IFS='|' read -r DATA_DIR LOG_DIR VARIANT <<< "${VARIANTS[$idx]}"
            run_benchmark $gpu "${DATA_DIR}/${VARIANT}" "${LOG_DIR}/${VARIANT}" &
        fi
    done
    wait
done

echo "=== All benchmarking runs completed ==="

# ================================================================
#   STEP 3: Evaluation on Clean Data (Multi-GPU Parallel)
# ================================================================

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
            IFS='|' read -r DATA_DIR LOG_DIR VARIANT <<< "${VARIANTS[$idx]}"
            run_eval_clean $gpu "${DATA_DIR}/${VARIANT}" "${LOG_DIR}/${VARIANT}" "${VARIANT}" &
        fi
    done
    wait
done

echo "=== All evaluations completed ==="

# ================================================================
#   Summary
# ================================================================

echo ""
echo "########################################"
echo "  All Geometric Baselines Completed!"
echo "########################################"
echo ""
echo "Results saved to:"
echo "  Gaussian Blur     : ${BLUR_LOG}/"
echo "  JPEG Compression  : ${JPEG_LOG}/"
echo "  Gaussian Noise    : ${NOISE_LOG}/"
echo "  Geometric (S+R)   : ${GEO_LOG}/"
echo ""
echo "Processed variants:"
for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r DATA_DIR LOG_DIR VARIANT <<< "${entry}"
    echo "  - ${VARIANT} -> ${LOG_DIR}/${VARIANT}/"
done
echo ""

