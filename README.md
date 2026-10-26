# AsyDiff

This repository contains the multi-dataset pansharpening implementation of **AsyDiff**, with joint training on **WorldView-3 (WV3), QuickBird (QB), and GaoFen-2 (GF2)**. Evaluation uses the reduced-resolution (RR) and full-resolution (FR) test sets of **WV3, WV2, QB, and GF2**.

The release includes the requested training/inference entry points, their configuration files, and the required model/data-loading code. Other ablations, datasets, pretrained weights, checkpoints, and generated results are not included. Repository documentation follows the installation → data → training → evaluation organization of [SCL-IOD](https://github.com/LuowenDragon/SCL-IOD).

## Requirements and installation

The source environment uses Python **3.8.20**, PyTorch **2.4.1 + CUDA 12.1**, torchvision **0.19.1**, PyTorch Lightning **1.6.5**. No text encoder or Transformers installation is required. Use a CUDA-capable NVIDIA GPU for training and the supplied evaluation entry point.

```bash
git clone https://github.com/yangyingkun/AsyDiff.git
cd AsyDiff
conda env create -f environment.yaml
conda activate AsyDiff
```

Alternatively, create the environment explicitly:

```bash
conda create -n AsyDiff python=3.8.20 pip=24.0 -y
conda activate AsyDiff
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install xformers==0.0.28.post1
python -m pip install -r requirements.txt
python -m pip install -e .
```

Keep `pip<24.1`: newer pip releases reject Lightning 1.6.5's legacy package metadata. Do not upgrade Lightning to 2.x; these entry points use its 1.x Trainer API. xformers matches the source environment; the code also includes a fallback when it is unavailable. The requirements are a runtime subset, not a dump of unrelated packages from the source environment. A fresh-environment installation has not yet been validated.

## Data preparation

This project uses the **[PanCollection](https://github.com/liangjiandeng/PanCollection)** pansharpening datasets: WV3, QB, and GF2 for training, and the RR/FR test sets of WV3, WV2, QB, and GF2 for evaluation.

Download the corresponding HDF5 datasets by following the instructions and download links in the **[PanCollection repository](https://github.com/liangjiandeng/PanCollection)**. Please follow the dataset providers' usage terms and citation instructions. Dataset files are not included in this repository.

## Training: WV3 + QB + GF2

Run commands from the repository root:

```bash
mkdir -p log_file
CUDA_VISIBLE_DEVICES=1 bash tools/train.sh \
  > log_file/stage1_trans_multidataset_kernel9_60.log 2>&1
```

The corresponding direct command is:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --base configs/AsyDiff/WV3/config/Trans_trans.yaml \
  -t --gpus 0,
```

`CUDA_VISIBLE_DEVICES=1` selects physical GPU 1; `--gpus 0,` selects its remapped logical GPU 0. Change the former for another physical GPU.

Training defaults to **100,000 optimizer steps**. The batch size is **8**, and each batch comes from one sensor. Dataset sampling probabilities are proportional to `8 × N_WV3`, `4 × N_QB`, and `4 × N_GF2`. The base learning rate is `2e-5`; default learning-rate scaling yields **1.6e-4** for one GPU, batch size 8, and accumulation factor 1. The model predicts `x0` with L1 loss, 1,000 diffusion timesteps, and transfer offsets `edit_t1=60`, `edit_t2=120`.

Runs are saved under timestamped `logs/` directories. Checkpoints are saved every 10,000 training steps; the callback also enables epoch-end and last-checkpoint saving. Use the actual filenames under your run's `checkpoints/` directory, not a manually assumed run name. For a short training check or alternate data locations:

```bash
CUDA_VISIBLE_DEVICES=1 bash tools/train.sh --max_steps 2 \
  wv3_data.params.train.params.h5py_path=/path/to/train_wv3.h5 \
  qb_data.params.train.params.h5py_path=/path/to/train_qb.h5 \
  gf2_data.params.train.params.h5py_path=/path/to/train_gf2.h5
```

## Evaluation: reduced and full resolution

Choose the RR and FR test files for the sensor you want to evaluate. The supplied `infer/WV3/eval.sh` uses **WV2 as its default example**; its directory name does not select the test sensor. No prompt argument is required or accepted. To run the provided WV2 example, set the checkpoint directory and a starting step that actually exists:

```bash
CUDA_VISIBLE_DEVICES=0 \
CHECKPOINT_DIR=logs/YOUR_RUN/checkpoints \
START_STEP=10000 END_STEP=80001 STEP_STRIDE=10000 \
bash infer/WV3/eval.sh
```

The original sweep defaults (`START_STEP=9`, `END_STEP=81000`, `STEP_STRIDE=10000`) are preserved if no overrides are provided. They only work if files such as `step=000000009.ckpt` exist; set `START_STEP` to match your checkpoints. `END_STEP` is exclusive.

Select the sensor's RR/FR files with `REDUCED_DATA` and `FULL_DATA`, and use a separate `OUTPUT_ROOT` for each dataset. The supplied Python evaluator defaults to eight bands and a normalization scale of 2047. Match its channel handling and normalization to the selected sensor before running it; changing file paths alone is not sufficient for every sensor.

Optional environment variables are `REDUCED_DATA`, `FULL_DATA`, `OUTPUT_ROOT` (default `results/WV2`), and `EPOCH_FILE`. The sweep stops on a missing input or failed inference and advances its resume marker **only after successful inference**. Leave `START_STEP` unset when resuming from that marker.

The evaluator uses DDIM with `skip=20`, writes generated `.mat` files with the `sr` key, and preserves the original residual reconstruction and clipping. Its reconstruction-config/checkpoint arguments are compatibility inputs: it loads only the **transfer model**. The reconstruction YAML is therefore an empty documented placeholder. This entry point exports predictions; it does not calculate benchmark metrics.

## Acknowledgments and license

The implementation builds on [Latent Diffusion](https://github.com/CompVis/latent-diffusion) and [Stable Diffusion](https://github.com/Stability-AI/stablediffusion).

No AsyDiff paper metadata, pretrained release, or benchmark scores are supplied in this source release; no publication citation is invented.
