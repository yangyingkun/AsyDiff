from torch.utils.data import Dataset

import cv2

import torch

import numpy as np

import h5py

class PansharpeningDataset(Dataset):
    def __init__(self, h5py_path, img_scale, size=64, scene_file=None):
        super().__init__()

        data = h5py.File(h5py_path)  # NxCxHxW = 0x1x2x3=8806x8x64x64

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1.transpose(0, 2, 3, 1), dtype=np.float32) / img_scale
        # gt1 = gt1 * 2.0 - 1.0
        self.gt = torch.from_numpy(gt1)   # NxHxWxC

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1.transpose(0, 2, 3, 1), dtype=np.float32) / img_scale
        # lms1 = lms1 * 2.0 - 1.0
        self.lms = torch.from_numpy(lms1)   # NxHxWxC

        ms1 = data["ms"][...]  # NxCxHxW=0,1,2,3
        ms1 = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / img_scale  # NxHxWxC
        # ms1 = ms1 * 2.0 - 1.0
        ms1_tmp = ms1
        self.ms_hp = torch.from_numpy(ms1_tmp)  # NxHxWxC:

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / img_scale  # NxHxWx1
        # pan1 = pan1 * 2.0 - 1.0
        pan_hp_tmp = pan1
        self.pan_hp = torch.from_numpy(pan_hp_tmp)   # NxHxWx1
        data.close()

        self.length = self.gt.shape[0]
        print(f'gt: max{torch.max(self.gt)}, min{torch.min(self.gt)}')
        print(f'lms: max{torch.max(self.lms)}, min{torch.min(self.lms)}')
        print(f'pan_hp: max{torch.max(self.pan_hp)}, min{torch.min(self.pan_hp)}')
        print(f'ms_hp: max{torch.max(self.ms_hp)}, min{torch.min(self.ms_hp)}')

        print(
            f"gt: {self.gt.size()}, lms: {self.lms.size()}, pan_hp: {self.pan_hp.size()}, ms_hp: {self.ms_hp.size()} with {img_scale}")

        # self.grounding = [json.loads(q)['answer'][37:] for q in open(os.path.expanduser(grounding_file), "r")]
        self.scene = []  # Scene descriptions do not enter this model.

        # 提取所有的answer字段，并存储在一个列表中
        self.dataset_len = self.lms.shape[0]  # 目录下所有图像的数量
        print(self.dataset_len, len(self.scene))

    def __len__(self):
        return self.gt.shape[0]

    def calculate_regression_weights_advanced_single(self, LRMS_up, PAN, window_size=9):
        """
        使用cv2.blur的单张图像版本回归权重计算
        输入:
            LRMS_up: [h, w, c] 上采样后的多光谱图像
            PAN: [h, w, 1] 或 [h, w] 全色图像
            window_size: 滑动窗口大小
        输出:
            weights: [h, w, c] 回归权重
            PAN_detail: [h, w] PAN细节分量
        """
        h, w, c = LRMS_up.shape

        # 确保窗口大小为奇数
        if window_size % 2 == 0:
            window_size += 1

        # 初始化输出数组
        weights = np.zeros((h, w, c))

        # 处理PAN维度
        if len(PAN.shape) == 3 and PAN.shape[2] == 1:
            PAN = PAN[:, :, 0]  # 从 [h, w, 1] 转换为 [h, w]

        # 使用cv2.blur计算PAN的低通版本
        PAN_low = cv2.blur(PAN, (window_size, window_size))
        PAN_detail = PAN - PAN_low

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

            weights[:, :, band] = regression_coef

        # 对权重进行后处理
        weights = np.maximum(weights, 0)

        # 归一化
        weight_sum = np.sum(weights, axis=2, keepdims=True)
        weight_sum = np.maximum(weight_sum, 1e-6)
        weights = weights / weight_sum

        return weights, PAN_detail

    def __getitem__(self, index):
        example = dict()
        # lrms : [b, h, w, c]                                 --[0, 1]
        # pan : [b, h, w, 1]                                  --[0, 1]
        # hrms : [b, h, w, c]                                 --[0, 1]
        # gt(hrms - lrms) : [b, h, w, c]                      --[-1, 1]
        # pan 高通部分(pan - pan 低通部分): [b, h, w, 1]        --[-1, 1]
        # lms 图像
        style_img = self.lms[index, :, :, :].float()
        style_img = style_img.unsqueeze(0)
        channels = style_img.shape[-1]
        # pan 图像
        char_img = self.pan_hp[index, :, :, :].float()
        char_img = torch.repeat_interleave(char_img, channels, dim=2)
        # pan 图像
        grey_char_img = self.pan_hp[index, :, :, :].float()
        grey_char_img = torch.repeat_interleave(grey_char_img, channels, dim=2)
        # gt 图像
        label_image = self.gt[index, :, :, :].float()
        # gt 分支部分 -- label 标签(hrms - lrms)  -- 残差图
        example['gt_branch'] = self.gt[index, :, :, :].float() - self.lms[index, :, :, :].float()   # 128, 128, 8
        # pan 分支部分 -- label 标签(pan - pan 低通部分) -- pan 高通部分
        pending_pan_image = np.array(self.pan_hp[index, :, :, :], dtype=np.float32)
        pending_lms_image = np.array(self.lms[index, :, :, :], dtype=np.float32)
        weights_batch, PAN_detail_batch = self.calculate_regression_weights_advanced_single(
                pending_lms_image, pending_pan_image)
        PAN_detail_batch = PAN_detail_batch[:, :, np.newaxis]
        residual_batch = weights_batch * PAN_detail_batch
        residual_batch = torch.from_numpy(residual_batch)
        example['pan_branch'] = residual_batch      # 128, 128, 8

        example["image"] = label_image              # 128, 128, 8
        example["style_imgs"] = style_img           # 1, 128, 128, 8
        example["char_img"] = char_img              # 128, 128, 8
        example["grey_char_img"] = grey_char_img    # 128, 128, 8

        return example
