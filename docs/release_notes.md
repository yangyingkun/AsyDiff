# Release scope and compatibility

## Included

- The multi-dataset training entry point and its YAML configuration.
- The selected WV2 reduced/full-resolution evaluator, its transfer configuration, and the resumable checkpoint sweep.
- Model variant `AsyDiff_Model`, the effective definitions of its blocks, and their recursive Python dependencies.
- Training callbacks, the original autoencoder/style-encoder dependencies, and the HDF5 loader needed by this experiment.

Other model variants, general font-generation tooling, dataset files, pretrained CLIP files, training checkpoints, logs, and results are excluded. The `ldm` package name and parameter names are retained for compatibility with existing state dictionaries. Project-specific modules/classes/configurations use the AsyDiff name.

## Packaging changes

1. Source extraction keeps the selected top-level definitions and imported dependencies. When the source declares the same class twice, only the final effective declaration is included.
2. Data paths are repository-relative or user-supplied. The unused CLIP loading and text-conditioning path have been removed; no pretrained text encoder is needed.
3. The HDF5 loader is named `PansharpeningDataset`. It no longer performs inherited font initialization, reads unused scene descriptions, or accepts/returns a prompt. Its image tensors, 9 × 9 regression weighting, and normalization are preserved.
4. Training, image logging, and evaluation use only image conditioning (`zf`, `zcs`, `zcf`). The evaluator converts the residual tensor to CPU before NumPy export. Import resolution works when invoking the script by filename.
5. The sweep exports GPU visibility, quotes paths, validates inputs, stops on failure, and records progress only after success. The default step sequence is unchanged; users can override it to match available checkpoints.
6. Training still defaults to 100,000 steps, but an explicit positive `--max_steps` now takes effect. Text logging falls back to Pillow's default font rather than requiring a bundled font asset.
7. The unused reconstruction configuration is a compatibility placeholder; no reconstruction model is instantiated by this evaluator.

## Intentionally retained behavior

This is a focused code release, not an algorithm redesign. The following source behaviors are preserved and should be considered when interpreting experiments:

- The original effective `Middle_Block.forward` computed text projections but overwrote their outputs with time embeddings. The unused computations are now removed; both modulation stages still use time embeddings. Tests compare outputs and gradients exactly against the original effective calculation for four/eight bands and different text inputs.
- Unused `embed2` and `text_embed1`/`text_embed2` parameter modules are retained for compatibility with existing model/optimizer state structure, but are never called. The frozen CLIP subtree is no longer instantiated. Loading older state dictionaries discards only `clip_text_model.*` keys at each backbone prefix; all other keys are validated normally. Use the new YAMLs rather than old configurations containing dataset `prompt` or backbone `clip_text_path`/`clip_device` arguments. Full optimizer/trainer resume from a production checkpoint has not been validated.
- Removing CLIP initialization may change random-number consumption during a fresh training run, so same-seed retraining is not claimed to be bitwise identical. With fixed effective weights/inputs, the time-modulation computation is unchanged.
- The backbone's retained name mentions stationary wavelets, but the selected implementation does not invoke a wavelet transform. Unused wavelet libraries and other ablation branches are not included.
- The base diffusion module initializes EMA before the subclass replaces its model wrapper; the source ordering is retained.
- The training path still instantiates the configured first-stage/style models. They are retained to preserve initialization and checkpoint structure even though the selected diffusion computation uses image-space residuals.

The scientific effects of these inherited behaviors are outside the packaging task. Correcting them should be treated as a separately versioned experiment with new validation.

## Validation limits

Import checks, configuration target resolution, syntax checks, synthetic HDF5 output checks, four/eight-band CPU forward/backward smoke tests, prompt-free batch training/logging, sampling, and legacy modulation/state-dictionary compatibility checks are provided. Tests do not use production weights or real data. Full GPU training and real-checkpoint WV2 evaluation must still be run in a prepared CUDA environment. Environment versions were read from the working source environment; a clean dependency installation was not performed.
