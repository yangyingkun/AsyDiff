# Release scope and compatibility

## Included

- The multi-dataset training entry point and its YAML configuration.
- The selected WV2 reduced/full-resolution evaluator, its transfer configuration, and the resumable checkpoint sweep.
- Model variant `Stationary_Wavelet_Model_With_MSCA_Spectral_Without_Dense_102`, the effective definitions of its blocks, and their recursive Python dependencies.
- Training callbacks, the original autoencoder/style-encoder dependencies, and the HDF5 loader needed by this experiment.

Other model variants, general font-generation tooling, dataset files, pretrained CLIP files, training checkpoints, logs, and results are excluded. The `ldm` package name and parameter names are retained for compatibility with existing state dictionaries. Project-specific modules/classes/configurations use the AsyDiff name.

## Packaging changes

1. Source extraction keeps the selected top-level definitions and imported dependencies. When the source declares the same class twice, only the final effective declaration is included.
2. Data paths are repository-relative or user-supplied. CLIP's location is configurable with `ASYDIFF_CLIP_PATH` or `clip_text_path`.
3. The HDF5 loader no longer performs inherited font initialization or reads unused scene descriptions. Its returned tensors, 9 × 9 regression weighting, sensor labels, and normalization are preserved.
4. The inference entry point explicitly supplies the missing WV2 prompt in its conditioning dictionary and converts the residual tensor to CPU before NumPy export. Import resolution works when invoking the script by filename.
5. The sweep exports GPU visibility, quotes paths, validates inputs, stops on failure, and records progress only after success. The default step sequence is unchanged; users can override it to match available checkpoints.
6. Training still defaults to 100,000 steps, but an explicit positive `--max_steps` now takes effect. Text logging falls back to Pillow's default font rather than requiring a bundled font asset.
7. The unused reconstruction configuration is a compatibility placeholder; no reconstruction model is instantiated by this evaluator.

## Intentionally retained behavior

This is a focused code release, not an algorithm redesign. The following source behaviors are preserved and should be considered when interpreting experiments:

- The WV2 prompt text in the original model describes GaoFen-2/four bands. It has not been rewritten here, because changing text can change embeddings and experimental behavior.
- The effective `MSCA_Block_GatedConv_With_SMFA_62.forward` computes text embeddings but then overwrites the reshaped text variables with time embeddings. This release does not silently change that modulation behavior.
- The backbone's retained name mentions stationary wavelets, but the selected implementation does not invoke a wavelet transform. Unused wavelet libraries and other ablation branches are not included.
- The base diffusion module initializes EMA before the subclass replaces its model wrapper; the source ordering is retained.
- The training path still instantiates the configured first-stage/style models. They are retained to preserve initialization and checkpoint structure even though the selected diffusion computation uses image-space residuals.

The scientific effects of these inherited behaviors are outside the packaging task. Correcting them should be treated as a separately versioned experiment with new validation.

## Validation limits

Import checks, configuration target resolution, syntax checks, synthetic HDF5 output checks, four/eight-band CPU forward/backward smoke tests, and short mocked-encoder sampling checks are provided. Tests do not use production weights or real data. Full GPU training and real-checkpoint WV2 evaluation must still be run in a prepared CUDA environment. Environment versions were read from the working source environment; a clean dependency installation was not performed.
