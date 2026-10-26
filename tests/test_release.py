"""Offline CPU smoke tests; synthetic data and a stand-in text encoder only."""
import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from ldm.data.AsyDiff_data import PansharpeningTrain_WV3_Res_Scene_Weight_Prompt_9
from ldm.modules.diffusionmodules import ablation
from ldm.util import get_obj_from_str, instantiate_from_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs/AsyDiff/WV3/config'
TRAIN_CONFIG = CONFIG_DIR / 'Trans_trans.yaml'
EVAL_CONFIG = CONFIG_DIR / 'Eval_trans.yaml'
EVAL_MODULE = 'scripts.WV3.AsyDiff_eval_Stationary_Wavelet_Model_With_MSCA_spectral_res_ddim_scene_different_source_2_9'


class StandInTextEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, text):
        return torch.zeros(1, 768)


def small_model(path):
    config = OmegaConf.load(path).model
    config.params.unet_config.params.middle_channels = 8
    config.params.unet_config.params.clip_device = 'cpu'
    config.params.first_stage_config = '__is_first_stage__'
    config.params.style_stage_config = '__is_first_stage__'
    with patch.object(ablation, 'FrozenCLIPEmbedder', StandInTextEncoder):
        return instantiate_from_config(config)


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(23)

    def test_python_syntax_and_config_targets(self):
        for path in ROOT.rglob('*.py'):
            if '.git' not in path.parts:
                ast.parse(path.read_text(), filename=str(path))

        def check(value):
            if isinstance(value, dict):
                if 'target' in value:
                    obj = get_obj_from_str(value['target'])
                    self.assertTrue(callable(obj), value['target'])
                    if value['target'].startswith(('ldm.', 'main.')):
                        module = importlib.import_module(obj.__module__)
                        self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT) if hasattr(Path('.'), 'is_relative_to')
                                        else str(Path(module.__file__).resolve()).startswith(str(ROOT) + '/'))
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)

        for path in CONFIG_DIR.glob('*.yaml'):
            check(OmegaConf.to_container(OmegaConf.load(path)))
        import main
        import train
        main.ImageLogger(batch_frequency=5000, max_images=8)
        callback = instantiate_from_config({'target': 'train.LearningRateLogger'})
        self.assertIsInstance(callback, train.LearningRateLogger)

    def test_training_entrypoint_help(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'train.py'), '--help'],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--base', result.stdout)
        self.assertIn('--gpus', result.stdout)
        for name in ('Eval_rec.yaml', 'Eval_trans.yaml', 'Trans_trans.yaml'):
            self.assertTrue((CONFIG_DIR / name).is_file())

    def test_synthetic_hdf5_and_regression_parity(self):
        evaluate = importlib.import_module(EVAL_MODULE)
        rng = np.random.RandomState(23)
        with tempfile.TemporaryDirectory() as directory:
            for channels, scale, prompt in [(4, 1023, 'GF2'), (8, 2047, 'WV3')]:
                path = Path(directory) / (prompt + '.h5')
                arrays = {
                    'gt': rng.rand(2, channels, 64, 64).astype('float32') * scale,
                    'lms': rng.rand(2, channels, 64, 64).astype('float32') * scale,
                    'ms': rng.rand(2, channels, 16, 16).astype('float32') * scale,
                    'pan': rng.rand(2, 1, 64, 64).astype('float32') * scale,
                }
                with h5py.File(path, 'w') as handle:
                    for name, array in arrays.items():
                        handle[name] = array
                dataset = PansharpeningTrain_WV3_Res_Scene_Weight_Prompt_9(
                    str(path), scale, prompt)
                sample = dataset[0]
                self.assertEqual(len(dataset), 2)
                self.assertEqual(sample['prompt'], prompt)
                self.assertEqual(tuple(sample['image'].shape), (64, 64, channels))
                expected_gt = (arrays['gt'][0] / scale).transpose(1, 2, 0)
                expected_lms = (arrays['lms'][0] / scale).transpose(1, 2, 0)
                np.testing.assert_allclose(sample['image'], expected_gt)
                np.testing.assert_allclose(sample['gt_branch'], expected_gt - expected_lms)
                weights, detail = evaluate.calculate_regression_weights_advanced_batch_cv2(
                    expected_lms[None], (arrays['pan'][:1] / scale).transpose(0, 2, 3, 1))
                np.testing.assert_allclose(sample['pan_branch'], weights[0] * detail[0, :, :, None], rtol=1e-6)
                self.assertTrue(torch.isfinite(sample['pan_branch']).all())

    def test_four_and_eight_band_forward_backward(self):
        with patch.object(ablation, 'FrozenCLIPEmbedder', StandInTextEncoder):
            model = ablation.Stationary_Wavelet_Model_With_MSCA_Spectral_Without_Dense_102(
                in_channels=1, out_channels=1, middle_channels=8, clip_device='cpu')
        for channels, prompt in [(4, 'QB'), (8, 'WV3')]:
            x = torch.randn(1, channels, 64, 64)
            output = model(x, x, [prompt], x, torch.tensor([50]))
            self.assertEqual(output.shape, x.shape)
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertIsNotNone(model.rebuild.weight.grad)
            self.assertTrue(torch.isfinite(model.rebuild.weight.grad).all())
            model.zero_grad(set_to_none=True)

    def test_diffusion_loss_and_short_sampling(self):
        x = torch.randn(1, 8, 64, 64) * 0.1
        cond = dict(zf=[x], zcs=[x], zcf=[x], prompt=[['WV2']])
        model = small_model(TRAIN_CONFIG)
        loss, metrics = model.p_losses(x, cond, torch.tensor([100]))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('train/loss', metrics)
        loss.backward()
        model = small_model(EVAL_CONFIG).eval()
        with torch.no_grad():
            result, _ = model.ddim_progressive_denoising(
                cond, cond, shape=list(x.shape), skip=20, verbose=False)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_sweep_progress_is_transactional(self):
        script = ROOT / 'infer/WV3/infer_different_source_1.sh'
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            fake_python = tmp / 'python'
            fake_python.write_text('#!/bin/sh\nexit "${FAKE_EXIT:-0}"\n')
            fake_python.chmod(0o755)
            for name in ['rr.h5', 'fr.h5', 'step=000000009.ckpt']:
                (tmp / name).touch()
            marker = tmp / 'current.txt'
            env = dict(os.environ, PATH=str(tmp) + os.pathsep + os.environ['PATH'],
                       CHECKPOINT_DIR=str(tmp), REDUCED_DATA=str(tmp / 'rr.h5'),
                       FULL_DATA=str(tmp / 'fr.h5'), OUTPUT_ROOT=str(tmp / 'results'),
                       EPOCH_FILE=str(marker), START_STEP='9', END_STEP='10', STEP_STRIDE='10000')
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text().strip(), '10009')
            marker.write_text('9\n')
            result = subprocess.run(['bash', str(script)], env=dict(env, FAKE_EXIT='7'), capture_output=True)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(marker.read_text().strip(), '9')
            # Use a range containing the missing checkpoint.
            result = subprocess.run(['bash', str(script)], env=dict(env, START_STEP='10000', END_STEP='10001'), capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text().strip(), '9')


if __name__ == '__main__':
    unittest.main()
