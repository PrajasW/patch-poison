# PatchPoison: Poisoning Multi-View Datasets to Degrade 3D Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

This repository is the official code release for **PatchPoison**, a framework for evaluating the vulnerability of 3D Gaussian Splatting (3DGS) to dataset poisoning attacks. 

We demonstrate that by applying structured perturbations (such as checkerboard patterns, noise, and geometric transformations) to training images, an attacker can severely degrade the novel view synthesis quality of 3DGS while keeping the poisoned images perceptually similar to the clean dataset.

---

## Overview

**PatchPoison** provides an automated, end-to-end pipeline to:
1. Generate poisoned variants of standard 3D datasets (e.g., NeRF-Synthetic).
2. Process the poisoned data (including COLMAP SfM).
3. Train the victim 3D Gaussian Splatting models.
4. Benchmark the attack success by measuring both **imperceptibility** (stealthiness in the 2D training data) and **reconstruction degradation** (impact on the 3D rendered novel views).

---

## Installation & Environment Setup

The repository uses Conda to manage dependencies. We provide a setup script to install the environment and the required CUDA extensions.

```bash
# Clone the repository
git clone git@github.com:PrajasW/patch-poison.git
cd patch-poison

# Run the setup script to install dependencies and compile submodules
bash setup.sh
```

This will create a Conda environment named `poison_splat`, install `torch==2.1.0+cu118`, and compile the `diff-gaussian-rasterization` submodule required by 3DGS.

> **Hardware Prerequisites:** 3D Gaussian Splatting is VRAM-intensive. To run the full poisoning and training pipeline, we recommend a Linux environment (Ubuntu 22.04 tested) with an NVIDIA GPU. For the NeRF-Synthetic dataset, at least 12GB of VRAM is recommended. For the Mip-NeRF 360 dataset, 48GB of VRAM is recommended.

---

## Dataset Preparation

We benchmark our attacks primarily on the **NeRF-Synthetic** and **Mip-NeRF 360** datasets. Download them automatically using the provided script:

```bash
bash get_dataset.sh
```

This script will download and extract the datasets into the appropriate directories for the pipeline to use.

---

## Experimental Pipeline

All orchestration scripts for generating poisoned data, training, and benchmarking are located in the `main/` directory.

### 1. Dataset Poisoning (`make_*.py`)
The `make_*.py` scripts are responsible for injecting specific types of perturbations into the clean datasets. 
For example, to generate color-based checkerboard attacks:
```bash
python main/make_colour.py
```
This generates variants in your configured dataset path with varying intensities and block sizes.

### 2. End-to-End Pipeline (`*.sh`)
The shell scripts in `main/` automate the full experiment lifecycle for different attack vectors. For example, to run the color attack pipeline:
```bash
bash main/colour.sh
```
**What this does under the hood:**
1. Generates the poisoned dataset variants.
2. Runs COLMAP on the poisoned datasets to estimate poses (if necessary).
3. Trains the victim 3DGS model for multiple experimental runs.
4. Evaluates the resulting model.

**Available attack pipelines in `main/` include:**
- `colour.sh`: Tests attacks manipulating color intensities (e.g., checkerboard color differences).
- `geometric_baselines.sh`: Tests baseline geometric attacks and standard image corruptions (e.g., Gaussian blur, JPEG compression, Gaussian noise).
- `contrast.sh`: Tests attacks that modify image contrast.
- `freq.sh`: Tests attacks modifying frequency components (e.g., high-frequency patterns).
- `ratio.sh`: Tests the effectiveness of the attack when varying the ratio of poisoned data to clean data.
- `size.sh`: Tests the impact of varying the size of the injected poison patch.
- `alpha.sh`: Tests attacks altering alpha (transparency) channels.
- `variants.sh`: Tests various combinations and other specific attack variants.

---

## Benchmarking & Evaluation

Evaluation is handled by `victim/gaussian-splatting/custom_benchmark.py`. Unlike standard 3DGS evaluation, our benchmark measures two critical dimensions of a poisoning attack:

1. **Imperceptibility**: Measures PSNR, SSIM, and LPIPS between the *clean dataset* and the *poisoned dataset*. This ensures the attack remains stealthy to human observers inspecting the training data.
2. **Reconstruction Degradation**: Measures PSNR, SSIM, and LPIPS between the *final 3DGS renders* and the *clean ground truth views*. This quantifies how successfully the attack broke the novel view synthesis.

The orchestration scripts automatically trigger this benchmark and save the metrics into `benchmark_dataset_comparison.log` and `benchmark_recon_quality.log` within the output directories.

---

## Generating Results & Tables

To reproduce the exact tables and results reported in the paper, we provide parsing scripts in the `tables/` directory. 

These scripts read the logs produced by the benchmarking pipeline and format them into readable console output and LaTeX tables:

```bash
# Generate the main results table
python tables/generate_results_table.py

# Generate clean/full dataset specific tables
python tables/generate_clean_dataset_tables.py
python tables/generate_full_dataset_tables.py
```

Outputs are aggregated from the subdirectories within the `results/` folder (e.g., `results_clean`, `results_freq`).

---

## Repository Structure

```text
patch-poison/
├── main/                   # scripts
│   ├── make_*.py           # Poisoned dataset generators
│   └── *.sh                # End-to-end attack pipelines
├── results/                # Output directory for evaluation results
├── tables/                 # Scripts to generate paper tables from logs
├── victim/
│   └── gaussian-splatting/ # The victim 3DGS implementation
│       ├── train.py        # Standard 3DGS training script
│       ├── benchmark.py    # Trains and Benchmarks the model on given dataset
│       └── custom_benchmark.py # Evaluates the model for our evaluation function
├── setup.sh                # Environment and dependency setup
└── get_dataset.sh          # Dataset download utility
```

---

## Acknowledgements

This repository heavily builds upon the official [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) implementation. We thank the authors for their foundational work and open-source contributions.

The benchmarking framework and evaluation pipeline are adapted from the [Poison-Splat](https://github.com/jiahaolu97/poison-splat) project. We acknowledge the authors for releasing their benchmarking scripts, which served as the foundation for our experimental infrastructure.


---

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{wadekar2026patchpoison,
  title={PatchPoison: Poisoning Multi-View Datasets to Degrade 3D Reconstruction},
  author={Wadekar, Prajas and Bachina, Venkata Sai Pranav and Bhosikar, Kunal and Gangwal, Ankit and Sharma, Charu},
  journal={arXiv preprint arXiv:2604.13153},
  year={2026}
}
```
