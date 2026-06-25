#!/bin/bash
#SBATCH -n 36
#SBATCH --gres=gpu:3
#SBATCH --partition u22
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --job-name=render_clean_size100
#SBATCH --output=render_clean_size100.log

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

# ---- Paths ----
# Clean baseline
CLEAN_DATASET="/ssd_scratch/prajas/dataset/ns_clean"
CLEAN_LOG="/ssd_scratch/prajas/logs/ns_clean"
CLEAN_RENDER="/ssd_scratch/prajas/renders/ns_clean"

# Size experiment (patch_size_100 = checkerboard, block_size 4, size 100)
SIZE_DATASET="/ssd_scratch/prajas/dataset/ns_size"
SIZE_LOG="/ssd_scratch/prajas/logs/ns_size"
SIZE_RENDER="/ssd_scratch/prajas/renders/ns_size"

NUM_GPUS=3

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

# Only render patch_size_100 (checkerboard, freq 4)
SIZE_VARIANT="patch_size_100"

echo "========================================================"
echo "  Combined Render Pipeline: Clean + Size-100 Checkerboard"
echo "========================================================"
echo "Clean dataset    : ${CLEAN_DATASET}"
echo "Clean log/models : ${CLEAN_LOG}"
echo "Clean renders    : ${CLEAN_RENDER}"
echo ""
echo "Size dataset     : ${SIZE_DATASET}"
echo "Size log/models  : ${SIZE_LOG}"
echo "Size renders     : ${SIZE_RENDER}"
echo "Size variant     : ${SIZE_VARIANT}"
echo ""
echo "Number of scenes : ${#SCENES[@]}"
echo "Number of GPUs   : ${NUM_GPUS}"
echo "Scenes: ${SCENES[*]}"
echo ""



# ============================
#   Function Definitions
# ============================

run_render_clean() {
    local gpu=$1
    local scene=$2
    local exp_run=$3
    local dataset_path="${CLEAN_DATASET}/${scene}"
    local model_path="${CLEAN_LOG}/${scene}/exp_run_${exp_run}"
    local output_path="${CLEAN_RENDER}/${scene}/exp_run_${exp_run}"

    if [ ! -f "${model_path}/victim_model.ply" ]; then
        echo ">>> [GPU ${gpu}] WARNING: Clean model not found at ${model_path}/victim_model.ply. Skipping..."
        return
    fi

    echo ">>> [GPU ${gpu}] Rendering CLEAN ${scene} exp_run_${exp_run} ..."
    mkdir -p "${output_path}"
    python victim/gaussian-splatting/render_model.py --gpu ${gpu} \
        -s ${dataset_path}/ \
        -m ${model_path} \
        --output_path ${output_path} \
        --skip_test
    echo ">>> [GPU ${gpu}] Rendering completed for CLEAN ${scene} exp_run_${exp_run}"
}

run_render_size() {
    local gpu=$1
    local scene=$2
    local exp_run=$3
    local variant="${scene}/${SIZE_VARIANT}"
    local dataset_path="${SIZE_DATASET}/${variant}"
    local model_path="${SIZE_LOG}/${variant}/exp_run_${exp_run}"
    local output_path="${SIZE_RENDER}/${variant}/exp_run_${exp_run}"

    if [ ! -f "${model_path}/victim_model.ply" ]; then
        echo ">>> [GPU ${gpu}] WARNING: Size model not found at ${model_path}/victim_model.ply. Skipping..."
        return
    fi

    echo ">>> [GPU ${gpu}] Rendering SIZE-100 ${scene} exp_run_${exp_run} ..."
    mkdir -p "${output_path}"
    python victim/gaussian-splatting/render_model.py --gpu ${gpu} \
        -s ${dataset_path}/ \
        -m ${model_path} \
        --output_path ${output_path} \
        --skip_test
    echo ">>> [GPU ${gpu}] Rendering completed for SIZE-100 ${scene} exp_run_${exp_run}"
}

# ============================
#   PART 1: Render Clean Baseline
# ============================

echo ""
echo "########################################"
echo "  PART 1: Rendering Clean Baseline"
echo "########################################"

mkdir -p "${CLEAN_RENDER}"

for EXP_RUN in 1 2 3; do
    echo ""
    echo "========================================"
    echo "  Clean: Rendering exp_run_${EXP_RUN}"
    echo "========================================"

    TOTAL=${#SCENES[@]}
    BATCHES=$(((TOTAL + NUM_GPUS - 1) / NUM_GPUS))

    for ((i=0; i<$TOTAL; i+=NUM_GPUS)); do
        batch=$((i/NUM_GPUS + 1))
        echo "---- Starting clean render batch ${batch}/${BATCHES} (exp_run_${EXP_RUN}) ----"
        for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
            idx=$((i+gpu))
            if [ $idx -lt $TOTAL ]; then
                run_render_clean $gpu "${SCENES[$idx]}" ${EXP_RUN} &
            fi
        done
        wait
    done

done

echo ""
echo "=== Clean rendering completed ==="

# ============================
#   PART 2: Render Size-100 (Checkerboard, Freq 4)
# ============================

echo ""
echo "########################################"
echo "  PART 2: Rendering Size-100 (Checkerboard Freq=4)"
echo "########################################"

mkdir -p "${SIZE_RENDER}"

for EXP_RUN in 1 2 3; do
    echo ""
    echo "========================================"
    echo "  Size-100: Rendering exp_run_${EXP_RUN}"
    echo "========================================"

    TOTAL=${#SCENES[@]}
    BATCHES=$(((TOTAL + NUM_GPUS - 1) / NUM_GPUS))

    for ((i=0; i<$TOTAL; i+=NUM_GPUS)); do
        batch=$((i/NUM_GPUS + 1))
        echo "---- Starting size-100 render batch ${batch}/${BATCHES} (exp_run_${EXP_RUN}) ----"
        for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
            idx=$((i+gpu))
            if [ $idx -lt $TOTAL ]; then
                run_render_size $gpu "${SCENES[$idx]}" ${EXP_RUN} &
            fi
        done
        wait
    done

done

echo ""
echo "=== Size-100 rendering completed ==="

# ============================
#   Summary
# ============================

echo ""
echo "########################################"
echo "  All Renders Completed!"
echo "########################################"
echo ""
echo "Clean renders saved to: ${CLEAN_RENDER}/"
echo "Size-100 renders saved to: ${SIZE_RENDER}/"
echo ""
echo "Rendered scenes:"
for SCENE in "${SCENES[@]}"; do
    echo "  CLEAN    : ${SCENE} -> ${CLEAN_RENDER}/${SCENE}/"
    echo "  SIZE-100 : ${SCENE} -> ${SIZE_RENDER}/${SCENE}/${SIZE_VARIANT}/"
done
echo ""

