# AsyDiff

This repository contains the multi-dataset pansharpening implementation of **AsyDiff**, with joint training on **WorldView-3 (WV3), QuickBird (QB), and GaoFen-2 (GF2)**. Evaluation uses the reduced-resolution (RR) and full-resolution (FR) test sets of **WV3, WV2, QB, and GF2**.

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

## Training

Set the WV3, QB, and GF2 training data paths in [Trans_trans.yaml](configs/AsyDiff/WV3/config/Trans_trans.yaml), then run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --base configs/AsyDiff/WV3/config/Trans_trans.yaml \
  -t --gpus 0, --no-test true
```

Training checkpoints are saved under `logs/`. Change `CUDA_VISIBLE_DEVICES` to select your GPU.

## Evaluation

Set the checkpoint directory and the RR/FR test data paths:

```bash
CUDA_VISIBLE_DEVICES=0 \
CHECKPOINT_DIR=logs/YOUR_RUN/checkpoints \
REDUCED_DATA=/path/to/test_RR.h5 \
FULL_DATA=/path/to/test_FR.h5 \
START_STEP=10000 END_STEP=10001 \
bash infer/WV3/eval.sh
```

This evaluates `step=000010000.ckpt`; adjust the paths and step to your checkpoint. Results are saved as `.mat` files under `results/`. The evaluator currently uses eight bands and a normalization scale of 2047; adjust these settings in `scripts/WV3/eval.py` when switching sensors.

## Acknowledgments and license

The implementation builds on [Latent Diffusion](https://github.com/CompVis/latent-diffusion) and [Stable Diffusion](https://github.com/Stability-AI/stablediffusion).

No AsyDiff paper metadata, pretrained release, or benchmark scores are supplied in this source release; no publication citation is invented.
