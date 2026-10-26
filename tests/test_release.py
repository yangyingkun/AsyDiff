"""Offline CPU smoke tests using synthetic data; no text encoder or downloads."""
import ast
import copy
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import default_collate

from ldm.data.AsyDiff_data import PansharpeningDataset
from ldm.modules.diffusionmodules import ablation
from ldm.modules.diffusionmodules.middle_block import Middle_Block
from ldm.util import get_obj_from_str, instantiate_from_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs/AsyDiff/WV3/config'
TRAIN_CONFIG = CONFIG_DIR / 'Trans_trans.yaml'
EVAL_CONFIG = CONFIG_DIR / 'Eval_trans.yaml'
EVAL_MODULE = 'scripts.WV3.eval'


def small_model(path):
    config = OmegaConf.load(path).model
    config.params.unet_config.params.middle_channels = 8
    config.params.first_stage_config = '__is_first_stage__'
    config.params.style_stage_config = '__is_first_stage__'
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
                    if value['target'] == 'ldm.data.AsyDiff_data.PansharpeningDataset':
                        inspect.signature(obj).bind(**value.get('params', {}))
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
            model = small_model(TRAIN_CONFIG)
            for channels, scale, sensor in [(4, 1023, 'GF2'), (8, 2047, 'WV3')]:
                path = Path(directory) / (sensor + '.h5')
                arrays = {
                    'gt': rng.rand(2, channels, 64, 64).astype('float32') * scale,
                    'lms': rng.rand(2, channels, 64, 64).astype('float32') * scale,
                    'ms': rng.rand(2, channels, 16, 16).astype('float32') * scale,
                    'pan': rng.rand(2, 1, 64, 64).astype('float32') * scale,
                }
                with h5py.File(path, 'w') as handle:
                    for name, array in arrays.items():
                        handle[name] = array
                dataset = PansharpeningDataset(str(path), scale)
                sample = dataset[0]
                self.assertEqual(len(dataset), 2)
                self.assertNotIn('prompt', sample)
                self.assertEqual(tuple(sample['image'].shape), (64, 64, channels))
                expected_gt = (arrays['gt'][0] / scale).transpose(1, 2, 0)
                expected_lms = (arrays['lms'][0] / scale).transpose(1, 2, 0)
                np.testing.assert_allclose(sample['image'], expected_gt)
                np.testing.assert_allclose(sample['gt_branch'], expected_gt - expected_lms)
                weights, detail = evaluate.calculate_regression_weights_advanced_batch_cv2(
                    expected_lms[None], (arrays['pan'][:1] / scale).transpose(0, 2, 3, 1))
                np.testing.assert_allclose(sample['pan_branch'], weights[0] * detail[0, :, :, None], rtol=1e-6)
                self.assertTrue(torch.isfinite(sample['pan_branch']).all())
                batch = default_collate([sample])
                loss, _ = model.shared_step(batch)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                model.zero_grad(set_to_none=True)
                logged = model.log_images(batch, N=1)
                self.assertIn('inputs', logged)

    def test_four_and_eight_band_forward_backward(self):
        model = ablation.AsyDiff_Model(
            in_channels=1, out_channels=1, middle_channels=8)
        self.assertFalse(hasattr(model, 'clip_text_model'))
        for channels in (4, 8):
            x = torch.randn(1, channels, 64, 64)
            output = model(x, x, x, torch.tensor([50]))
            self.assertEqual(output.shape, x.shape)
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertIsNotNone(model.rebuild.weight.grad)
            self.assertTrue(torch.isfinite(model.rebuild.weight.grad).all())
            model.zero_grad(set_to_none=True)

    def test_diffusion_loss_and_short_sampling(self):
        x = torch.randn(1, 8, 64, 64) * 0.1
        cond = dict(zf=[x], zcs=[x], zcf=[x])
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

    def test_time_modulation_matches_legacy_outputs_and_gradients(self):
        # The old implementation discarded both text projections. Preserve its
        # effective math rather than replacing time modulation with new behavior.
        block = Middle_Block(8, 8, 8, 8, 128)
        for bands in (4, 8):
            for text_scale in (0.0, 100.0):
                reference = copy.deepcopy(block)
                x = torch.randn(1, 8, bands, 8, 8, requires_grad=True)
                old_x = x.detach().clone().requires_grad_(True)
                t = torch.randn(1, 8, requires_grad=True)
                old_t = t.detach().clone().requires_grad_(True)
                text = torch.randn(1, 128) * text_scale
                time1 = reference.tim_embed1(old_t).reshape(1, 8, 1, 1, 1)
                time2 = reference.tim_embed2(old_t).reshape(1, 8, 1, 1, 1)
                text1 = reference.text_embed1(text)
                text1 = time1.reshape(1, 8, 1, 1, 1)
                text2 = reference.text_embed2(text)
                text2 = time2.reshape(1, 8, 1, 1, 1)
                data = reference.smfa1(reference.norm1(old_x) * time1 + time2) + old_x
                expected = reference.smfa2(reference.norm2(data) * text1 + text2)
                actual = block(x, t)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.square().mean().backward()
                expected.square().mean().backward()
                torch.testing.assert_close(x.grad, old_x.grad, rtol=0, atol=0)
                torch.testing.assert_close(t.grad, old_t.grad, rtol=0, atol=0)
                for name, parameter in block.named_parameters():
                    old_parameter = dict(reference.named_parameters())[name]
                    if parameter.grad is None:
                        self.assertIsNone(old_parameter.grad, name)
                    else:
                        torch.testing.assert_close(parameter.grad, old_parameter.grad, rtol=0, atol=0)
                block.zero_grad(set_to_none=True)

    def test_legacy_text_encoder_keys_are_ignored_only(self):
        model = small_model(EVAL_CONFIG)
        state = model.state_dict()
        legacy_key = 'model.diffusion_model.clip_text_model.transformer.dummy_weight'
        state[legacy_key] = torch.zeros(2)
        result = model.load_state_dict(state, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        state['model.diffusion_model.unknown_weight'] = torch.zeros(2)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state, strict=True)
        del state['model.diffusion_model.unknown_weight']
        del state['model.diffusion_model.rebuild.weight']
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state, strict=True)

    def test_evaluation_help_without_prompt(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/WV3/eval.py'), '--help'],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--prompt', result.stdout)

    def test_sweep_progress_is_transactional(self):
        script = ROOT / 'infer/WV3/eval.sh'
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
