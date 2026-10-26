#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
config="configs/AsyDiff/WV3/l1loss_stationary_wavelet_model_with_msca_spectral_without_dense_loss_ablation102_scene_weight/AsyDiff_Train_Stage1_trans_model_predx0_miniUnet_multidataset_kernel9_60.yaml"
mkdir -p log_file
exec python main_20epoch_prompt_multidataset.py --base "$config" -t --gpus 0, "$@"
