import torch as th

import torch.nn as nn

from ldm.modules.diffusionmodules.util import timestep_embedding

class Swish(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

from .middle_block import TKSCrossAttention3D

from .middle_block import Upsample3D, Downsample3D

from .middle_block import TKSCrossAttention3D

from .middle_block import Upsample3D, Downsample3D

from .middle_block import Middle_Block

class AsyDiff_Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        middle_channels,
        use_checkpoint=False,
        embed_dim=128,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.head = 1

        # Retain unused projection parameters for existing checkpoint compatibility.
        self.embed2 = nn.Sequential(nn.Linear(768, self.embed_dim * 4), Swish(),
                                    nn.Linear(self.embed_dim * 4, self.embed_dim * 4), Swish(),
                                    nn.Linear(self.embed_dim * 4, self.embed_dim))

        self.PAN_FE1 = nn.Conv3d(
            1, self.middle_channels, kernel_size=3, stride=1, padding=1)
        self.MS_FE1 = nn.Conv3d(
            1, self.middle_channels, kernel_size=3, stride=1, padding=1)

        self.FE2 = nn.Conv3d(
            3, self.middle_channels, kernel_size=3, stride=1, padding=1)

        self.pan_middle_block_1 = Middle_Block(
            self.middle_channels, self.middle_channels, self.middle_channels, self.middle_channels, self.embed_dim)

        self.pan_down_1 = Downsample3D(self.middle_channels)
        self.pan_middle_block_2 = Middle_Block(
            self.middle_channels * 2, self.middle_channels * 2, self.middle_channels * 2, self.middle_channels, self.embed_dim)

        self.pan_down_2 = Downsample3D(self.middle_channels * 2)
        self.pan_middle_block_3 = Middle_Block(
            self.middle_channels * 4, self.middle_channels * 4, self.middle_channels * 4, self.middle_channels, self.embed_dim)

        self.ms_middle_block_1 = Middle_Block(
            self.middle_channels, self.middle_channels, self.middle_channels, self.middle_channels, self.embed_dim)

        self.ms_down_1 = Downsample3D(self.middle_channels)
        self.ms_middle_block_2 = Middle_Block(
            self.middle_channels * 2, self.middle_channels * 2, self.middle_channels * 2, self.middle_channels, self.embed_dim)

        self.ms_down_2 = Downsample3D(self.middle_channels * 2)
        self.ms_middle_block_3 = Middle_Block(
            self.middle_channels * 4, self.middle_channels * 4, self.middle_channels * 4, self.middle_channels, self.embed_dim)

        self.gt_left_1 = Middle_Block(
            self.middle_channels, self.middle_channels, self.middle_channels, self.middle_channels, self.embed_dim)
        self.gt_down_1 = Downsample3D(self.middle_channels)
        self.gt_left_2 = Middle_Block(
            self.middle_channels * 2, self.middle_channels * 2, self.middle_channels * 2, self.middle_channels, self.embed_dim)
        self.gt_down_2 = Downsample3D(self.middle_channels * 2)
        self.gt_left_3 = Middle_Block(
            self.middle_channels * 4, self.middle_channels * 4, self.middle_channels * 4, self.middle_channels, self.embed_dim)
        self.gt_down_3 = Downsample3D(self.middle_channels * 4)

        self.gt_middle = Middle_Block(
            self.middle_channels * 8, self.middle_channels * 8, self.middle_channels * 8, self.middle_channels, self.embed_dim)

        self.gt_up_1 = Upsample3D(self.middle_channels * 8)
        self.gt_set_1 = nn.Conv3d(in_channels = self.middle_channels * 16, out_channels = self.middle_channels * 4, kernel_size = 1)
        self.gt_right_1 = Middle_Block(
            self.middle_channels * 4, self.middle_channels * 4, self.middle_channels * 4, self.middle_channels, self.embed_dim)
        self.gt_up_2 = Upsample3D(self.middle_channels * 4)
        self.gt_set_2 = nn.Conv3d(in_channels = self.middle_channels * 8, out_channels = self.middle_channels * 2, kernel_size = 1)
        self.gt_right_2 = Middle_Block(
            self.middle_channels * 2, self.middle_channels * 2, self.middle_channels * 2, self.middle_channels, self.embed_dim)
        self.gt_up_3 = Upsample3D(self.middle_channels * 2)
        self.gt_set_3 = nn.Conv3d(in_channels = self.middle_channels * 4, out_channels = self.middle_channels, kernel_size = 1)
        self.gt_right_3 = Middle_Block(
            self.middle_channels, self.middle_channels, self.middle_channels, self.middle_channels, self.embed_dim)

        self.rebuild = nn.Conv3d(
            self.middle_channels, self.out_channels, kernel_size=1)

        self.cross_1 = TKSCrossAttention3D(dim=self.middle_channels, num_heads=self.head)
        self.cross_2 = TKSCrossAttention3D(dim=self.middle_channels*2, num_heads=self.head*2)
        self.cross_3 = TKSCrossAttention3D(dim=self.middle_channels*4, num_heads=self.head*4)

        self.act = Swish()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Discard only the frozen text encoder from older checkpoints.
        legacy_prefix = prefix + 'clip_text_model.'
        for key in list(state_dict):
            if key.startswith(legacy_prefix):
                del state_dict[key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, lrms, pan, xt, timesteps):
        t_emb = timestep_embedding(
            timesteps, self.middle_channels, repeat_only=False)

        lrms = lrms.unsqueeze(1)
        pan = pan.unsqueeze(1)
        xt = xt.unsqueeze(1)

        lrms_module_1 = self.MS_FE1(lrms)       # b 1 8 h w -> b c 8 h w
        pan_module_1 = self.PAN_FE1(pan)        # b 1 8 h w -> b c 8 h w

        gt = th.cat((xt, pan, lrms), dim=1)
        gt_module_1 = self.FE2(gt)              # b 3 8 h w -> b c 8 h w

        lrms_module_1 = self.ms_middle_block_1(lrms_module_1, t_emb)
        lrms_module_2 = self.ms_down_1(lrms_module_1)
        lrms_module_2 = self.ms_middle_block_2(lrms_module_2, t_emb)
        lrms_module_3 = self.ms_down_2(lrms_module_2)
        lrms_module_3 = self.ms_middle_block_3(lrms_module_3, t_emb)

        pan_module_1 = self.pan_middle_block_1(pan_module_1, t_emb)
        pan_module_2 = self.pan_down_1(pan_module_1)
        pan_module_2 = self.pan_middle_block_2(pan_module_2, t_emb)
        pan_module_3 = self.pan_down_2(pan_module_2)
        pan_module_3 = self.pan_middle_block_3(pan_module_3, t_emb)

        pan_module_1 = self.cross_1(pan_module_1, lrms_module_1)
        feature_1 = th.cat((lrms_module_1, pan_module_1), dim=1)

        pan_module_2 = self.cross_2(pan_module_2, lrms_module_2)
        feature_2 = th.cat((lrms_module_2, pan_module_2), dim=1)

        pan_module_3 = self.cross_3(pan_module_3, lrms_module_3)
        feature_3 = th.cat((lrms_module_3, pan_module_3), dim=1)

        gt_left_1 = self.gt_left_1(gt_module_1, t_emb)
        gt_down_1 = self.gt_down_1(gt_left_1)
        gt_left_2 = self.gt_left_2(gt_down_1, t_emb)
        gt_down_2 = self.gt_down_2(gt_left_2)
        gt_left_3 = self.gt_left_3(gt_down_2, t_emb)
        gt_down_3 = self.gt_down_3(gt_left_3)

        gt_middle = self.gt_middle(gt_down_3, t_emb)

        gt_up_1 = self.gt_up_1(gt_middle)
        gt_feature_3 = th.cat((gt_up_1, gt_left_3, feature_3), dim = 1)
        gt_set_1 = self.gt_set_1(gt_feature_3)
        gt_right_1 = self.gt_right_1(gt_set_1, t_emb)

        gt_up_2 = self.gt_up_2(gt_right_1)
        gt_feature_2 = th.cat((gt_up_2, gt_left_2, feature_2), dim = 1)
        gt_set_2 = self.gt_set_2(gt_feature_2)
        gt_right_2 = self.gt_right_2(gt_set_2, t_emb)

        gt_up_3 = self.gt_up_3(gt_right_2)
        gt_feature_1 = th.cat((gt_up_3, gt_left_1, feature_1), dim = 1)
        gt_set_3 = self.gt_set_3(gt_feature_1)
        gt_right_3 = self.gt_right_3(gt_set_3, t_emb)

        output = self.rebuild(gt_right_3)
        output = output.squeeze(1)

        return output
