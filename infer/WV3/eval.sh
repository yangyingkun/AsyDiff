#!/usr/bin/env bash
# Retains the original location; the selected cross-source test dataset is WV2.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the checkpoints directory of your training run}"
reduced_data="${REDUCED_DATA:-data/WV2/test_wv2_data_RR.h5}"
full_data="${FULL_DATA:-data/WV2/test_wv2_data_FR.h5}"
output_root="${OUTPUT_ROOT:-results/WV2}"
epoch_file="${EPOCH_FILE:-$output_root/current_different_source_1.txt}"
end_step="${END_STEP:-81000}"
step_stride="${STEP_STRIDE:-10000}"
current_step="${START_STEP:-9}"
if [[ -z "${START_STEP:-}" && -f "$epoch_file" ]]; then
    read -r current_step < "$epoch_file"
fi
for value in "$current_step" "$end_step" "$step_stride"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Step values must be nonnegative integers.' >&2; exit 2; }
done
current_step=$((10#$current_step))
end_step=$((10#$end_step))
step_stride=$((10#$step_stride))
((step_stride > 0)) || { echo 'STEP_STRIDE must be positive.' >&2; exit 2; }
for path in "$reduced_data" "$full_data"; do
    [[ -f "$path" ]] || { echo "Missing dataset: $path" >&2; exit 1; }
done
config_dir="configs/AsyDiff/WV3/config"
mkdir -p "$output_root" "$(dirname -- "$epoch_file")"
while ((current_step < end_step)); do
    printf -v checkpoint '%s/step=%09d.ckpt' "$CHECKPOINT_DIR" "$current_step"
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint (set START_STEP to an existing step)" >&2; exit 1; }
    python scripts/WV3/eval.py \
        --pansharpening_reduced_path "$reduced_data" \
        --pansharpening_full_path "$full_data" \
        --path_config_rec "$config_dir/Eval_rec.yaml" \
        --path_rec_model "$checkpoint" \
        --path_config_trans "$config_dir/Eval_trans.yaml" \
        --path_trans_model "$checkpoint" \
        --save_reduced_mat_path "$output_root/reduced/step-$current_step" \
        --save_full_mat_path "$output_root/full/step-$current_step"
    current_step=$((current_step + step_stride))
    printf '%s\n' "$current_step" > "$epoch_file"
done
echo 'Finished all requested checkpoints.'
