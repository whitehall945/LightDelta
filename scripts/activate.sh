#!/usr/bin/env bash
# Source this file from either the control host or an mlx worker.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo 'Use: source scripts/activate.sh' >&2
    exit 2
fi
source /your_home_dir/miniconda3/etc/profile.d/conda.sh
conda activate lightdelta
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONNOUSERSITE=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export LIGHTDELTA_MODEL_PATH="${LIGHTDELTA_MODEL_PATH:-your_abs_path}"
