# AsyDiff

This repository contains the selected multi-dataset pansharpening experiment for **AsyDiff**. It trains the same transfer diffusion model on **WorldView-3 (WV3), QuickBird (QB), and GaoFen-2 (GF2)** and runs cross-source inference on **WorldView-2 (WV2)** at reduced and full resolution.

The release includes the requested training/inference entry points, their configuration files, and the required model/data-loading code. Other ablations, datasets, pretrained weights, checkpoints, and generated results are not included. Repository documentation follows the installation → data → training → evaluation organization of [SCL-IOD](https://github.com/LuowenDragon/SCL-IOD).

## Requirements and installation

The source environment uses Python **3.8.20**, PyTorch **2.4.1 + CUDA 12.1**, torchvision **0.19.1**, PyTorch Lightning **1.6.5**, and transformers **4.46.0**. Use a CUDA-capable NVIDIA GPU for training and the supplied evaluation entry point.

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

## Data and pretrained text encoder

Prepare the datasets in HDF5 format. Obtain data from its original providers or the [PanCollection project](https://github.com/liangjiandeng/PanCollection), subject to their terms. No datasets are redistributed here.

```text
AsyDiff/
├── data/
│   ├── WV3/train_wv3.h5
│   ├── QB/train_qb.h5
│   ├── GF2/train_gf2.h5
│   └── WV2/
│       ├── test_wv2_data_RR.h5
│       └── test_wv2_data_FR.h5
└── clip-vit-large-patch14-local/
    ├── config.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── vocab.json
    ├── merges.txt
    └── model.safetensors (or pytorch_model.bin)
```

HDF5 arrays use **N × C × H × W** layout. Training and reduced-resolution test files require `gt`, `lms`, `ms`, and `pan`. Full-resolution files require `lms`, `ms`, and `pan`. `lms` is upsampled multispectral imagery aligned with PAN; `ms` is the original low-resolution multispectral image. PAN has one channel. WV3/WV2 use eight spectral bands; QB/GF2 use four. Training patches are configured at 64 × 64. Values are divided by **2047** for WV3/QB and **1023** for GF2; the selected WV2 evaluator uses **2047**.

Set each `*.params.train.params.h5py_path` in the [training configuration](configs/AsyDiff/WV3/l1loss_stationary_wavelet_model_with_msca_spectral_without_dense_loss_ablation102_scene_weight/AsyDiff_Train_Stage1_trans_model_predx0_miniUnet_multidataset_kernel9_60.yaml), or override it on the command line. Font files and scene JSONL files are not needed: neither contributes to the selected loader's returned samples.

Prepare the Hugging Face [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) tokenizer and text-encoder files locally. For example, download them separately:

```bash
huggingface-cli download openai/clip-vit-large-patch14 \
  --include config.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt tokenizer.json pytorch_model.bin \
  --local-dir clip-vit-large-patch14-local
```

To reuse an existing local copy, set `export ASYDIFF_CLIP_PATH=/path/to/clip-vit-large-patch14-local`. The model also accepts `clip_text_path` in its UNet configuration. These files are not uploaded to GitHub. Use only trusted checkpoint sources and follow their model licenses.

## Training: WV3 + QB + GF2

Run commands from the repository root:

```bash
mkdir -p log_file
CUDA_VISIBLE_DEVICES=1 bash tools/train.sh \
  > log_file/stage1_trans_multidataset_kernel9_60.log 2>&1
```

The corresponding direct command is:

```bash
CUDA_VISIBLE_DEVICES=1 python main_20epoch_prompt_multidataset.py \
  --base configs/AsyDiff/WV3/l1loss_stationary_wavelet_model_with_msca_spectral_without_dense_loss_ablation102_scene_weight/AsyDiff_Train_Stage1_trans_model_predx0_miniUnet_multidataset_kernel9_60.yaml \
  -t --gpus 0,
```

`CUDA_VISIBLE_DEVICES=1` selects physical GPU 1; `--gpus 0,` selects its remapped logical GPU 0. Change the former for another physical GPU.

The filename is historical: training defaults to **100,000 optimizer steps**, not 20 epochs. The batch size is **8**, and each batch comes from one sensor. Dataset sampling probabilities are proportional to `8 × N_WV3`, `4 × N_QB`, and `4 × N_GF2`. The base learning rate is `2e-5`; default learning-rate scaling yields **1.6e-4** for one GPU, batch size 8, and accumulation factor 1. The model predicts `x0` with L1 loss, 1,000 diffusion timesteps, and transfer offsets `edit_t1=60`, `edit_t2=120`.

Runs are saved under timestamped `logs/` directories. Checkpoints are saved every 10,000 training steps; the callback also enables epoch-end and last-checkpoint saving. Use the actual filenames under your run's `checkpoints/` directory, not a manually assumed run name. For a short training check or alternate data locations:

```bash
CUDA_VISIBLE_DEVICES=1 bash tools/train.sh --max_steps 2 \
  wv3_data.params.train.params.h5py_path=/path/to/train_wv3.h5 \
  qb_data.params.train.params.h5py_path=/path/to/train_qb.h5 \
  gf2_data.params.train.params.h5py_path=/path/to/train_gf2.h5
```

## Evaluation: WV2 reduced/full resolution

The script remains under `infer/WV3/` to preserve the original entry-point location, but **its test data and default satellite prompt are WV2**. Set the checkpoint directory and a starting step that actually exists:

```bash
CUDA_VISIBLE_DEVICES=0 \
CHECKPOINT_DIR=logs/YOUR_RUN/checkpoints \
START_STEP=10000 END_STEP=80001 STEP_STRIDE=10000 \
bash infer/WV3/infer_different_source_1.sh
```

The original sweep defaults (`START_STEP=9`, `END_STEP=81000`, `STEP_STRIDE=10000`) are preserved if no overrides are provided. They only work if files such as `step=000000009.ckpt` exist; set `START_STEP` to match your checkpoints. `END_STEP` is exclusive. Do not change the dataset to another sensor without adapting this eight-band, 2047-normalized evaluator.

Optional environment variables are `REDUCED_DATA`, `FULL_DATA`, `OUTPUT_ROOT` (default `results/WV2`), and `EPOCH_FILE`. The sweep stops on a missing input or failed inference and advances its resume marker **only after successful inference**. Leave `START_STEP` unset when resuming from that marker.

The evaluator uses DDIM with `skip=20`, writes generated `.mat` files with the `sr` key, and preserves the original residual reconstruction and clipping. Its reconstruction-config/checkpoint arguments are compatibility inputs: it loads only the **transfer model**. The reconstruction YAML is therefore an empty documented placeholder. This entry point exports predictions; it does not calculate benchmark metrics.

## Repository layout and checks

```text
configs/AsyDiff/    Selected training and inference configuration files
ldm/               Required diffusion, network, and dataset implementation
main*.py           Training entry point and its callback/data-module dependencies
tools/train.sh     Training launcher
infer/WV3/         Resumable WV2 checkpoint sweep
scripts/WV3/       Reduced/full-resolution inference implementation
tests/             Synthetic data and CPU smoke tests
docs/              Release scope and compatibility notes
```

```bash
python -m unittest discover -s tests -v
```

The smoke tests use synthetic data and a stand-in text encoder; they do not download weights, train the full experiment, or establish numerical benchmark performance. See [release notes](docs/release_notes.md) for packaging changes and retained source behavior.

## Acknowledgments and license

The implementation builds on [Latent Diffusion](https://github.com/CompVis/latent-diffusion), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), the CVPR 2024 work *Generate Like Experts: Multi-Stage Font Generation by Incorporating Font Transfer Process into Diffusion Models* by Bin Fu et al., and [FFG-benchmarks](https://github.com/clovaai/fewshot-font-generation). CLIP is provided by [OpenAI](https://github.com/openai/CLIP) and Hugging Face Transformers. Original copyright notices are retained in [LICENSE](LICENSE) and [licenses/FFG-MIT.txt](licenses/FFG-MIT.txt).

No AsyDiff paper metadata, pretrained release, or benchmark scores are supplied in this source release; no publication citation is invented.
