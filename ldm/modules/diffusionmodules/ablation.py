import os

import torch as th

import torch.nn as nn

from ldm.modules.diffusionmodules.util import timestep_embedding

from .clip import FrozenCLIPEmbedder

class Swish(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

from .middle_block import TKSCrossAttention3D

from .middle_block import Upsample3D, Downsample3D

from .middle_block import TKSCrossAttention3D

from .middle_block import Upsample3D, Downsample3D

from .middle_block import MSCA_Block_GatedConv_With_SMFA_62

class Stationary_Wavelet_Model_With_MSCA_Spectral_Without_Dense_102(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        middle_channels,
        use_checkpoint=False,
        embed_dim=128,
        clip_text_path=None,
        clip_device="cuda",
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.head = 1

        self.clip_text_path = clip_text_path or os.environ.get(
            "ASYDIFF_CLIP_PATH", "clip-vit-large-patch14-local")
        self.embed2 = nn.Sequential(nn.Linear(768, self.embed_dim * 4), Swish(),
                                    nn.Linear(self.embed_dim * 4, self.embed_dim * 4), Swish(),
                                    nn.Linear(self.embed_dim * 4, self.embed_dim))

        self.PAN_FE1 = nn.Conv3d(
            1, self.middle_channels, kernel_size=3, stride=1, padding=1)
        self.MS_FE1 = nn.Conv3d(
            1, self.middle_channels, kernel_size=3, stride=1, padding=1)

        self.FE2 = nn.Conv3d(
            3, self.middle_channels, kernel_size=3, stride=1, padding=1)

        self.pan_middle_block_1 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels, self.middle_channels, self.middle_channels, self.middle_channels, self.embed_dim)

        self.pan_down_1 = Downsample3D(self.middle_channels)
        self.pan_middle_block_2 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels * 2, self.middle_channels * 2, self.middle_channels * 2, self.middle_channels, self.embed_dim)

        self.pan_down_2 = Downsample3D(self.middle_channels * 2)
        self.pan_middle_block_3 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels * 4, self.middle_channels * 4, self.middle_channels * 4, self.middle_channels, self.embed_dim)

        self.ms_middle_block_1 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels, self.middle_channels, self.middle_channels, self.middle_channels, self.embed_dim)

        self.ms_down_1 = Downsample3D(self.middle_channels)
        self.ms_middle_block_2 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels * 2, self.middle_channels * 2, self.middle_channels * 2, self.middle_channels, self.embed_dim)

        self.ms_down_2 = Downsample3D(self.middle_channels * 2)
        self.ms_middle_block_3 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels * 4, self.middle_channels * 4, self.middle_channels * 4, self.middle_channels, self.embed_dim)

        self.gt_left_1 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels, self.middle_channels, self.middle_channels, self.middle_channels, self.embed_dim)
        self.gt_down_1 = Downsample3D(self.middle_channels)
        self.gt_left_2 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels * 2, self.middle_channels * 2, self.middle_channels * 2, self.middle_channels, self.embed_dim)
        self.gt_down_2 = Downsample3D(self.middle_channels * 2)
        self.gt_left_3 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels * 4, self.middle_channels * 4, self.middle_channels * 4, self.middle_channels, self.embed_dim)
        self.gt_down_3 = Downsample3D(self.middle_channels * 4)

        self.gt_middle = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels * 8, self.middle_channels * 8, self.middle_channels * 8, self.middle_channels, self.embed_dim)

        self.gt_up_1 = Upsample3D(self.middle_channels * 8)
        self.gt_set_1 = nn.Conv3d(in_channels = self.middle_channels * 16, out_channels = self.middle_channels * 4, kernel_size = 1)
        self.gt_right_1 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels * 4, self.middle_channels * 4, self.middle_channels * 4, self.middle_channels, self.embed_dim)
        self.gt_up_2 = Upsample3D(self.middle_channels * 4)
        self.gt_set_2 = nn.Conv3d(in_channels = self.middle_channels * 8, out_channels = self.middle_channels * 2, kernel_size = 1)
        self.gt_right_2 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels * 2, self.middle_channels * 2, self.middle_channels * 2, self.middle_channels, self.embed_dim)
        self.gt_up_3 = Upsample3D(self.middle_channels * 2)
        self.gt_set_3 = nn.Conv3d(in_channels = self.middle_channels * 4, out_channels = self.middle_channels, kernel_size = 1)
        self.gt_right_3 = MSCA_Block_GatedConv_With_SMFA_62(
            self.middle_channels, self.middle_channels, self.middle_channels, self.middle_channels, self.embed_dim)

        self.rebuild = nn.Conv3d(
            self.middle_channels, self.out_channels, kernel_size=1)

        self.cross_1 = TKSCrossAttention3D(dim=self.middle_channels, num_heads=self.head)
        self.cross_2 = TKSCrossAttention3D(dim=self.middle_channels*2, num_heads=self.head*2)
        self.cross_3 = TKSCrossAttention3D(dim=self.middle_channels*4, num_heads=self.head*4)

        self.act = Swish()
        self.clip_text_model = FrozenCLIPEmbedder(version=self.clip_text_path, device=clip_device)
        self.encode_prompt()

    def encode_prompt(self):
        self.gf2_embeding = self.clip_text_model.encode(self.get_prompt("GF2"))
        self.wv3_embeding = self.clip_text_model.encode(self.get_prompt("WV3"))
        self.qb_embeding = self.clip_text_model.encode(self.get_prompt("QB"))
        self.wv2_embeding = self.clip_text_model.encode(self.get_prompt("WV2"))
        self.wv4_embeding = self.clip_text_model.encode(self.get_prompt("WV4"))
        print(self.gf2_embeding.shape, self.wv3_embeding.shape, self.qb_embeding.shape, self.wv2_embeding.shape)

    def get_prompt(self, prompt):
        if prompt == "QB":
            return "The QuickBird satellite captures high-resolution images with notable physical properties. Its panchromatic sensor acquires images at a 0.61-meter resolution, providing crisp and detailed grayscale visuals. The multispectral sensor captures images at a 2.44-meter resolution in four spectral bands: blue (450-520 nm), green (520-600 nm), red (630-690 nm), and near-infrared (760-900 nm). These physical properties enable accurate Earth observation, supporting applications in environmental monitoring, land use planning, urban mapping, and disaster management."
        elif prompt == "WV3":
            return "The WorldView-3 satellite captures high-resolution images with exceptional physical properties. Its panchromatic sensor acquires images at a 31 cm resolution, delivering sharp and detailed grayscale visuals. The multispectral sensor captures images at a 1.24 m resolution in eight spectral bands: coastal (400-450 nm), blue (450-510 nm), green (510-580 nm), yellow (585-625 nm), red (630-690 nm), red edge (705-745 nm), near-infrared 1 (770-895 nm), and near-infrared 2 (860-1,040 nm). Additionally, WorldView-3 features a shortwave infrared (SWIR) sensor with 3.7 m resolution in eight bands (1,195-1,385 nm, 1,560-1,660 nm, 2,045-2,110 nm, etc.). These physical properties enable advanced Earth observation, supporting applications in environmental monitoring, land use planning, urban mapping, and disaster response."
        elif prompt == "GF2":
            return "The GaoFen-2 satellite captures high-resolution images with notable physical properties. Its panchromatic sensor acquires images at a 1.0-meter resolution, delivering clear and detailed grayscale visuals. The multispectral sensor captures images at a 4.0-meter resolution in four spectral bands: blue (450-520 nm), green (520-590 nm), red (630-690 nm), and near-infrared (770-890 nm). These physical properties enable accurate Earth observation, supporting applications in urban planning, environmental monitoring, disaster management, and land use analysis."
        elif prompt == "WV2":
            return "The GaoFen-2 satellite captures high-resolution images with notable physical properties. Its panchromatic sensor acquires images at a 0.5-meter resolution, delivering clear and detailed grayscale visuals. The multispectral sensor captures images at a 2.0-meter resolution in four spectral bands: blue (450-520 nm), green (520-590 nm), red (630-690 nm), and near-infrared (770-890 nm). These physical properties enable accurate Earth observation, supporting applications in urban planning, environmental monitoring, disaster management, and land use analysis."
        elif prompt == "WV4":
            return "The WorldView-4 satellite captures high-resolution images with remarkable physical properties. Its panchromatic sensor acquires images at a 31 cm resolution, providing sharp, detailed grayscale visuals. The multispectral sensor captures images at a 1.24 m resolution in four spectral bands: blue (450-510 nm), green (510-580 nm), red (630-690 nm), and near-infrared (770-895 nm). These physical properties enable precise Earth observation, facilitating applications in environmental monitoring, land use planning, and disaster response."
        return None

    def get_embeding(self, prompt):
        if prompt == "QB":
            return self.qb_embeding
        elif prompt == "WV3":
            return self.wv3_embeding
        elif prompt == "GF2":
            return self.gf2_embeding
        elif prompt == "WV2":
            return self.wv2_embeding
        elif prompt == "WV4":
            return self.wv4_embeding
        return None

    def forward(self, lrms, pan, prompts, xt, timesteps):
    # def forward(self, lrms, pan, xt, timesteps):
        t_emb = timestep_embedding(
            timesteps, self.middle_channels, repeat_only=False)

        target = prompts[0]
        # target = "WV3"
        prompt = self.get_embeding(target).repeat((xt.shape[0], 1)).to(xt.device)
        prompt = self.act(self.embed2(prompt))

        # print(xt.shape)
        lrms = lrms.unsqueeze(1)
        pan = pan.unsqueeze(1)
        xt = xt.unsqueeze(1)

        lrms_module_1 = self.MS_FE1(lrms)       # b 1 8 h w -> b c 8 h w
        pan_module_1 = self.PAN_FE1(pan)        # b 1 8 h w -> b c 8 h w

        gt = th.cat((xt, pan, lrms), dim=1)
        gt_module_1 = self.FE2(gt)              # b 3 8 h w -> b c 8 h w

        lrms_module_1 = self.ms_middle_block_1(lrms_module_1, t_emb, prompt)
        lrms_module_2 = self.ms_down_1(lrms_module_1)
        lrms_module_2 = self.ms_middle_block_2(lrms_module_2, t_emb, prompt)
        lrms_module_3 = self.ms_down_2(lrms_module_2)
        lrms_module_3 = self.ms_middle_block_3(lrms_module_3, t_emb, prompt)

        pan_module_1 = self.pan_middle_block_1(pan_module_1, t_emb, prompt)
        pan_module_2 = self.pan_down_1(pan_module_1)
        pan_module_2 = self.pan_middle_block_2(pan_module_2, t_emb, prompt)
        pan_module_3 = self.pan_down_2(pan_module_2)
        pan_module_3 = self.pan_middle_block_3(pan_module_3, t_emb, prompt)

        pan_module_1 = self.cross_1(pan_module_1, lrms_module_1)
        feature_1 = th.cat((lrms_module_1, pan_module_1), dim=1)

        pan_module_2 = self.cross_2(pan_module_2, lrms_module_2)
        feature_2 = th.cat((lrms_module_2, pan_module_2), dim=1)

        pan_module_3 = self.cross_3(pan_module_3, lrms_module_3)
        feature_3 = th.cat((lrms_module_3, pan_module_3), dim=1)

        gt_left_1 = self.gt_left_1(gt_module_1, t_emb, prompt)
        gt_down_1 = self.gt_down_1(gt_left_1)
        gt_left_2 = self.gt_left_2(gt_down_1, t_emb, prompt)
        gt_down_2 = self.gt_down_2(gt_left_2)
        gt_left_3 = self.gt_left_3(gt_down_2, t_emb, prompt)
        gt_down_3 = self.gt_down_3(gt_left_3)

        gt_middle = self.gt_middle(gt_down_3, t_emb, prompt)

        gt_up_1 = self.gt_up_1(gt_middle)
        gt_feature_3 = th.cat((gt_up_1, gt_left_3, feature_3), dim = 1)
        gt_set_1 = self.gt_set_1(gt_feature_3)
        gt_right_1 = self.gt_right_1(gt_set_1, t_emb, prompt)

        gt_up_2 = self.gt_up_2(gt_right_1)
        gt_feature_2 = th.cat((gt_up_2, gt_left_2, feature_2), dim = 1)
        gt_set_2 = self.gt_set_2(gt_feature_2)
        gt_right_2 = self.gt_right_2(gt_set_2, t_emb, prompt)

        gt_up_3 = self.gt_up_3(gt_right_2)
        gt_feature_1 = th.cat((gt_up_3, gt_left_1, feature_1), dim = 1)
        gt_set_3 = self.gt_set_3(gt_feature_1)
        gt_right_3 = self.gt_right_3(gt_set_3, t_emb, prompt)

        output = self.rebuild(gt_right_3)
        output = output.squeeze(1)
        # print(th.max(lrms), th.max(pan), th.max(xt), th.max(output), th.min(lrms), th.min(pan), th.min(xt), th.min(output), target, timesteps)

        return output
