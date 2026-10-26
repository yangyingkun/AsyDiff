import os

import argparse

import os

import sys

import glob

from omegaconf import OmegaConf

from PIL import Image

from tqdm import tqdm

import numpy as np

import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ldm.util import instantiate_from_config

import json

import random

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import torchvision.transforms as transforms

from einops import rearrange, repeat

from torchvision.utils import make_grid

import h5py

from scipy.io import savemat

import cv2

from pytorch_lightning import seed_everything

seed_everything(23)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def calculate_regression_weights_advanced_batch_cv2(LRMS_up_batch, PAN_batch, window_size=9):
    """
    使用cv2.blur的批次版本回归权重计算
    """
    b, h, w, c = LRMS_up_batch.shape

    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1

    # 初始化输出数组
    weights_batch = np.zeros((b, h, w, c))
    PAN_detail_batch = np.zeros((b, h, w))

    # 处理PAN批次维度
    if len(PAN_batch.shape) == 4 and PAN_batch.shape[3] == 1:
        PAN_batch = PAN_batch[:, :, :, 0]  # 从 [b, h, w, 1] 转换为 [b, h, w]

    # 对每个批次进行处理
    for batch_idx in range(b):
        LRMS_up = LRMS_up_batch[batch_idx]  # [h, w, c]
        PAN = PAN_batch[batch_idx]          # [h, w]

        # 使用cv2.blur计算PAN的低通版本
        PAN_low = cv2.blur(PAN, (window_size, window_size))
        PAN_detail = PAN - PAN_low
        PAN_detail_batch[batch_idx] = PAN_detail

        # 预计算PAN的均值（用于回归计算）
        PAN_mean = cv2.blur(PAN, (window_size, window_size))

        # 对每个波段进行向量化计算
        for band in range(c):
            band_data = LRMS_up[:, :, band]  # [h, w]

            # 计算波段的均值
            band_mean = cv2.blur(band_data, (window_size, window_size))

            # 计算协方差分子: E[(PAN - PAN_mean) * (band - band_mean)]
            cov_numerator = cv2.blur(
                (PAN - PAN_mean) * (band_data - band_mean),
                (window_size, window_size)
            )

            # 计算PAN的方差
            pan_variance = cv2.blur(
                (PAN - PAN_mean) ** 2,
                (window_size, window_size)
            )

            # 计算回归系数
            with np.errstate(divide='ignore', invalid='ignore'):
                regression_coef = np.divide(
                    cov_numerator,
                    pan_variance,
                    out=np.zeros_like(cov_numerator),
                    where=(pan_variance > 1e-6)
                )

            weights_batch[batch_idx, :, :, band] = regression_coef

        # 对当前批次权重进行后处理
        weights_current = weights_batch[batch_idx]
        weights_current = np.maximum(weights_current, 0)

        # 归一化
        weight_sum = np.sum(weights_current, axis=2, keepdims=True)
        weight_sum = np.maximum(weight_sum, 1e-6)
        weights_batch[batch_idx] = weights_current / weight_sum

    return weights_batch, PAN_detail_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        default=None,
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--pansharpening_reduced_path",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--pansharpening_full_path",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--path_config_rec",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--path_rec_model",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--path_config_trans",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--path_trans_model",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--save_reduced_mat_path",
        type=str,
        nargs="?",
        help="path to save mat file to",
    )
    parser.add_argument(
        "--save_full_mat_path",
        type=str,
        nargs="?",
        help="path to save mat file to",
    )
    opt = parser.parse_args()

    path_config_rec = opt.path_config_rec
    path_rec_model = opt.path_rec_model
    config1 = OmegaConf.load(path_config_rec)

    path_config_trans = opt.path_config_trans
    path_trans_model = opt.path_trans_model
    config2 = OmegaConf.load(path_config_trans)
    trans_model = instantiate_from_config(config2.model)

    sd = torch.load(path_trans_model, map_location="cpu", weights_only=True)

    if "state_dict" in list(sd.keys()):
        sd = sd["state_dict"]
    keys = list(sd.keys())
    ignore_keys = []
    for k in keys:
        if 'first_stage_model' in k:
            ignore_keys.append(k)
        if 'cond_stage_model' in k:
            ignore_keys.append(k)

    for k in keys:
        for ik in ignore_keys:
            if k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    trans_model.load_state_dict(sd, strict=False)


    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    trans_model = trans_model.to(device)

    trans_model = trans_model.eval()

    # 降分辨率测试
    print(opt.pansharpening_reduced_path)
    data = h5py.File(opt.pansharpening_reduced_path)  # NxCxHxW = 0x1x2x3=8806x8x64x64
    print('数据读取完成')

    gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
    gt1 = np.array(gt1, dtype=np.float32)
    gt = gt1                       # 保持数据为numpy

    lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
    lms1 = np.array(lms1, dtype=np.float32)
    lms = lms1.astype(np.float32)

    ms1 = data["ms"][...]  # NxCxHxW=0,1,2,3
    ms1 = np.array(ms1, dtype=np.float32)  # NxHxWxC
    ms1_tmp = ms1
    ms_hp = ms1_tmp.astype(np.float32)

    pan1 = data['pan'][...]  # Nx1xHxW
    pan1 = np.array(pan1, dtype=np.float32)  # NxHxWx1
    pan_hp_tmp = pan1
    pan_hp = pan_hp_tmp.astype(np.float32)

    print(f'lms1 max:{np.max(lms1)}, min:{np.min(lms1)}')
    print(f'pan1 max:{np.max(pan1)}, min:{np.min(pan1)}')
    print(f'ms1 max:{np.max(ms1)}, min:{np.min(ms1)}')

    print(
        f"lms: {lms.shape}, pan_hp: {pan_hp.shape}, ms_hp: {ms_hp.shape}")

    gt_path = opt.save_reduced_mat_path
    print(gt_path)
    if not os.path.exists(gt_path):
        # 创建目录
        os.makedirs(gt_path)
        print(f"路径 '{gt_path}' 已创建。")
    else:
        print(f"路径 '{gt_path}' 已存在。")

    batch_size = 1


    iter = 0
    batch = 1
    with torch.no_grad():
        with trans_model.ema_scope():
            while iter < pan_hp.shape[0]:
                # 清理不需要的变量
                if iter > 0:
                    torch.cuda.empty_cache()
                if iter + batch >= pan_hp.shape[0]:
                    pan_image = pan_hp[iter:pan_hp.shape[0], :, :, :]
                    lms_image = lms[iter:pan_hp.shape[0], :, :, :]
                else:
                    pan_image = pan_hp[iter:iter + batch, :, :, :]
                    lms_image = lms[iter:iter + batch, :, :, :]

                pan_image = np.array(pan_image, dtype=np.float32) / 2047
                lms_image = np.array(lms_image, dtype=np.float32) / 2047

                pending_pan_image = pan_image
                pending_pan_image = np.transpose(pending_pan_image, (0, 2, 3, 1))
                pending_lms_image = lms_image
                pending_lms_image = np.transpose(pending_lms_image, (0, 2, 3, 1))

                pan_image = torch.from_numpy(pan_image)
                lms_image = torch.from_numpy(lms_image)
                pan_image = torch.repeat_interleave(pan_image, 8, dim=1)

                weights_batch, PAN_detail_batch = calculate_regression_weights_advanced_batch_cv2(
                    pending_lms_image, pending_pan_image
                )
                PAN_detail_batch = PAN_detail_batch[:, :, :, np.newaxis]
                residual_batch = weights_batch * PAN_detail_batch
                residual_batch = np.transpose(residual_batch, (0, 3, 1, 2))
                residual_batch = np.array(residual_batch, dtype=np.float32)
                residual_batch = torch.from_numpy(residual_batch)

                zf = residual_batch
                zsf = lms_image
                zcf = pan_image

                zf = zf.cuda()
                zsf = zsf.cuda()
                zcf = zcf.cuda()

                c1 = dict(zf=[zf], zcs=[zsf], zcf=[zcf])

                nbs, nc, nh, nw = zf.shape
                shape = [nbs, nc, nh, nw]

                z0, progressives = trans_model.ddim_progressive_denoising(
                    c1, c1, shape=shape, skip=20)

                x_gen = z0

                for gen_img in x_gen:
                    residual = residual_batch[0]
                    residual = residual.detach().cpu().numpy().astype(np.float32) * 2047.0
                    residual = np.transpose(residual, (1, 2, 0))
                    savemat(os.path.join(
                        gt_path, f'residual_output_mulExm_{str(iter)}.mat'), {"sr": residual})

                    gen_img = np.array(gen_img.cpu())
                    gen_img = np.transpose(gen_img, (1, 2, 0))
                    print(np.max(gen_img), np.min(gen_img))

                    lms_img = lms[iter, :, :, :]  # Tensor -> NumPy数组
                    lms_img = np.array(lms_img, dtype=np.float32) / 2047.0
                    lms_img = np.transpose(lms_img, (1, 2, 0))
                    print(np.max(lms_img), np.min(lms_img))

                    print(gen_img.shape)
                    print(lms_img.shape)
                    gen_img = gen_img + lms_img
                    print(np.max(gen_img), np.min(gen_img))

                    gen_img = np.clip(gen_img, 0., 1.)
                    gen_img = gen_img * 2047.0
                    print(np.max(gen_img), np.min(gen_img))

                    savemat(os.path.join(
                        gt_path, f'output_mulExm_{str(iter)}.mat'), {"sr": gen_img})


                    iter = iter + 1

    # 全分辨率测试
    print(opt.pansharpening_full_path)
    data = h5py.File(opt.pansharpening_full_path)  # NxCxHxW = 0x1x2x3=8806x8x64x64
    print('数据读取完成')


    lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
    lms1 = np.array(lms1, dtype=np.float32)
    lms = lms1.astype(np.float32)

    ms1 = data["ms"][...]  # NxCxHxW=0,1,2,3
    ms1 = np.array(ms1, dtype=np.float32)  # NxHxWxC
    ms1_tmp = ms1
    ms_hp = ms1_tmp.astype(np.float32)

    pan1 = data['pan'][...]  # Nx1xHxW
    pan1 = np.array(pan1, dtype=np.float32)  # NxHxWx1
    pan_hp_tmp = pan1
    pan_hp = pan_hp_tmp.astype(np.float32)

    print(f'lms1 max:{np.max(lms1)}, min:{np.min(lms1)}')
    print(f'pan1 max:{np.max(pan1)}, min:{np.min(pan1)}')
    print(f'ms1 max:{np.max(ms1)}, min:{np.min(ms1)}')

    print(
        f"lms: {lms.shape}, pan_hp: {pan_hp.shape}, ms_hp: {ms_hp.shape}")

    gt_path = opt.save_full_mat_path
    print(gt_path)
    if not os.path.exists(gt_path):
        # 创建目录
        os.makedirs(gt_path)
        print(f"路径 '{gt_path}' 已创建。")
    else:
        print(f"路径 '{gt_path}' 已存在。")

    batch_size = 1


    iter = 0
    batch = 1
    with torch.no_grad():
        with trans_model.ema_scope():
            while iter < pan_hp.shape[0]:
                # 清理不需要的变量
                if iter > 0:
                    torch.cuda.empty_cache()
                if iter + batch >= pan_hp.shape[0]:
                    pan_image = pan_hp[iter:pan_hp.shape[0], :, :, :]
                    lms_image = lms[iter:pan_hp.shape[0], :, :, :]
                else:
                    pan_image = pan_hp[iter:iter + batch, :, :, :]
                    lms_image = lms[iter:iter + batch, :, :, :]

                pan_image = np.array(pan_image, dtype=np.float32) / 2047
                lms_image = np.array(lms_image, dtype=np.float32) / 2047

                pending_pan_image = pan_image
                pending_pan_image = np.transpose(pending_pan_image, (0, 2, 3, 1))
                pending_lms_image = lms_image
                pending_lms_image = np.transpose(pending_lms_image, (0, 2, 3, 1))

                pan_image = torch.from_numpy(pan_image)
                lms_image = torch.from_numpy(lms_image)
                pan_image = torch.repeat_interleave(pan_image, 8, dim=1)

                weights_batch, PAN_detail_batch = calculate_regression_weights_advanced_batch_cv2(
                    pending_lms_image, pending_pan_image
                )
                PAN_detail_batch = PAN_detail_batch[:, :, :, np.newaxis]
                residual_batch = weights_batch * PAN_detail_batch
                residual_batch = np.transpose(residual_batch, (0, 3, 1, 2))
                residual_batch = np.array(residual_batch, dtype=np.float32)
                residual_batch = torch.from_numpy(residual_batch)


                zf = residual_batch
                zsf = lms_image
                zcf = pan_image

                zf = zf.cuda()
                zsf = zsf.cuda()
                zcf = zcf.cuda()

                c1 = dict(zf=[zf], zcs=[zsf], zcf=[zcf])

                nbs, nc, nh, nw = zf.shape
                shape = [nbs, nc, nh, nw]

                z0, progressives = trans_model.ddim_progressive_denoising(
                    c1, c1, shape=shape, skip=20)
                x_gen = z0

                for gen_img in x_gen:
                    gen_img = np.array(gen_img.cpu())
                    gen_img = np.transpose(gen_img, (1, 2, 0))
                    print(np.max(gen_img), np.min(gen_img))

                    lms_img = lms[iter, :, :, :]  # Tensor -> NumPy数组
                    lms_img = np.array(lms_img, dtype=np.float32) / 2047.0
                    lms_img = np.transpose(lms_img, (1, 2, 0))
                    print(np.max(lms_img), np.min(lms_img))

                    print(gen_img.shape)
                    print(lms_img.shape)
                    gen_img = gen_img + lms_img
                    print(np.max(gen_img), np.min(gen_img))

                    gen_img = np.clip(gen_img, 0., 1.)
                    gen_img = gen_img * 2047.0
                    print(np.max(gen_img), np.min(gen_img))

                    savemat(os.path.join(
                        gt_path, f'output_mulExm_{str(iter)}.mat'), {"sr": gen_img})
                    iter = iter + 1
