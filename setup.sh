
conda create -n poison_splat python=3.11 -y
conda activate poison_splat

conda install -c conda-forge mamba
mamba install colmap -c conda-forge 
mamba install -c nvidia cuda-toolkit=11.8
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 
pip install plyfile tqdm matplotlib lpips einops colorama jaxtyping opencv
pip install gpuinfo
# mamba install --upgrade setuptools wheel torch
# install diffuser and simple
pip install setuptools

pip install git+https://github.com/shuyueW1991/simple_knn_illustration.git --no-build-isolation
cd victim/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install . --no-build-isolation
cd ~/patch-poison/
pip uninstall -y numpy
pip install numpy==1.26.4
