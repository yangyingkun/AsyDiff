import torch as th

import torch.nn as nn

import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import linear

from einops import rearrange

import numbers

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = th.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(th.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / th.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = th.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(th.ones(normalized_shape))
        self.bias = nn.Parameter(th.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / th.sqrt(sigma + 1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c n h w -> b (n h w) c')

def to_4d(x, n, h, w):
    return rearrange(x, 'b (n h w) c -> b c n h w', n=n, h=h, w=w)

class IBN(nn.Module):
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm3d(self.half, affine=True)
        self.BN = nn.BatchNorm3d(planes - self.half)

    def forward(self, x):
        split = th.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = th.cat((out1, out2), 1)
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="BiasFree"):
        super(LayerNorm, self).__init__()
        self.LayerNorm_type = LayerNorm_type
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        elif LayerNorm_type == "IBN":
            self.body = IBN(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if self.LayerNorm_type == 'IBN':
            return self.body(x)
        n, h, w = x.shape[-3:]
        return to_4d(self.body(to_3d(x)), n, h, w)  # 现在B C (NHW)内部norm然后转为b c n h w

class SMFA_3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels

        # 将2D卷积改为3D卷积
        self.in_conv = nn.Conv3d(in_channels=self.in_channels,
                                out_channels=self.middle_channels * 2, kernel_size=1)

        # 3D分组卷积，注意groups参数需要能被middle_channels整除
        self.easa_conv1 = nn.Conv3d(in_channels=self.middle_channels, out_channels=self.middle_channels,
                                    kernel_size=1, groups=self.middle_channels)
        self.easa_conv2 = nn.Conv3d(
            in_channels=self.middle_channels, out_channels=self.middle_channels, kernel_size=1)

        # 3D深度可分离卷积
        self.lde_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=self.middle_channels, out_channels=self.middle_channels,
                     kernel_size=3, padding=1, groups=self.middle_channels),
            nn.Conv3d(in_channels=self.middle_channels,
                     out_channels=self.middle_channels, kernel_size=1)
        )
        self.lde_conv2 = nn.Conv3d(
            in_channels=self.middle_channels, out_channels=self.middle_channels, kernel_size=1)

        self.out_conv = nn.Conv3d(
            in_channels=self.middle_channels, out_channels=self.in_channels, kernel_size=1)

        self.gelu = nn.GELU()
        self.down_scale = 8

        # 参数维度调整为5D [1, c, 1, 1, 1]
        self.alpha = nn.Parameter(th.ones((1, self.middle_channels, 1, 1, 1)))
        self.beta = nn.Parameter(th.zeros((1, self.middle_channels, 1, 1, 1)))

    def forward(self, Xt):
        data = Xt

        # 如果输入是 [b, n, c, h, w]，需要转换为 [b, c, n, h, w]
        if data.dim() == 5 and data.shape[1] != self.in_channels:
            # 假设输入是 [b, n, c, h, w]，转换为 [b, c, n, h, w]
            data = data.permute(0, 2, 1, 3, 4)

        b, c, d, h, w = data.shape

        # 3D卷积处理
        Y, X = self.in_conv(data).chunk(2, dim=1)

        # 3D自适应池化，只在空间维度下采样，保持深度维度
        x_s = self.easa_conv1(F.adaptive_max_pool3d(
            X, (d, h // self.down_scale, w // self.down_scale)))

        # 3D方差计算，在空间维度计算方差
        x_v = th.var(X, dim=(-3, -2, -1), keepdim=True)  # 在d,h,w维度计算方差

        # 3D插值上采样
        x_l = X * F.interpolate(
            self.gelu(self.easa_conv2(x_s * self.alpha + x_v * self.beta)),
            size=(d, h, w),
            # mode='trilinear',  # 使用三线性插值
            # align_corners=True  # 添加align_corners参数
        )

        y_l = self.lde_conv2(self.gelu(self.lde_conv1(Y)))

        out = self.out_conv(x_l + y_l)

        # 如果需要保持原始输入格式 [b, n, c, h, w]，转换回去
        if Xt.dim() == 5 and Xt.shape[1] != self.in_channels:
            out = out.permute(0, 2, 1, 3, 4)

        return out

class Downsample3D(nn.Module):
    def __init__(self, n_feat):
        super(Downsample3D, self).__init__()
        self.body = nn.Conv3d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        #b n c h w-> b n//2 c h w-> b 2n c h//2 w//2
        b, n, c, h, w = x.size()
        x = self.body(x) #b, n//2, c, h, w
        x = x.transpose(1, 2).reshape(b * c, n // 2, h, w)
        x = self.unshuffle(x).reshape(b, c, n * 2, h // 2, w // 2).transpose(2, 1)  #b, 2*n, c, h//2, w//2
        return x

class Upsample3D(nn.Module):
    def __init__(self, n_feat):
        super(Upsample3D, self).__init__()
        self.body = nn.Conv3d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        b, c, n, h, w = x.size()
        x = self.body(x)
        x = x.transpose(1, 2).reshape(b * n, c * 2, h, w)
        x = self.shuffle(x).reshape(b, n, c // 2, h * 2, w * 2).transpose(2, 1)
        return x

from einops import rearrange

class TKSCrossAttention3D(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(TKSCrossAttention3D, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(th.ones(num_heads, 1, 1))

        # 3D卷积层 - 输入维度为 [b, c, d, h, w]
        self.kv = nn.Conv3d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv3d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

        # 注意力权重参数
        self.attn1 = th.nn.Parameter(th.tensor([0.2]), requires_grad=True)
        self.attn2 = th.nn.Parameter(th.tensor([0.2]), requires_grad=True)
        self.attn3 = th.nn.Parameter(th.tensor([0.2]), requires_grad=True)
        self.attn4 = th.nn.Parameter(th.tensor([0.2]), requires_grad=True)

    def forward(self, pan, ms):
        # 输入维度 [b, c, d, h, w]
        b, c, d, h, w = pan.shape

        # 3D卷积处理 - 直接处理 [b, c, d, h, w]
        kv = self.kv_dwconv(self.kv(pan))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(ms))

        # 重新排列维度，将深度维度视为序列的一部分
        # 从 [b, c, d, h, w] 转换为 [b, head, c, (d*h*w)]
        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)

        # 归一化
        q = th.nn.functional.normalize(q, dim=-1)
        k = th.nn.functional.normalize(k, dim=-1)

        _, _, C, N = q.shape  # C是每个头的通道数，N = d * h * w

        # 初始化掩码
        mask1 = th.zeros(b, self.num_heads, C, C, device=pan.device, requires_grad=False)
        mask2 = th.zeros(b, self.num_heads, C, C, device=pan.device, requires_grad=False)
        mask3 = th.zeros(b, self.num_heads, C, C, device=pan.device, requires_grad=False)
        mask4 = th.zeros(b, self.num_heads, C, C, device=pan.device, requires_grad=False)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # 创建不同稀疏度的注意力掩码
        index = th.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = th.where(mask1 > 0, attn, th.full_like(attn, float('-inf')))

        index = th.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = th.where(mask2 > 0, attn, th.full_like(attn, float('-inf')))

        index = th.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = th.where(mask3 > 0, attn, th.full_like(attn, float('-inf')))

        index = th.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = th.where(mask4 > 0, attn, th.full_like(attn, float('-inf')))

        # softmax归一化
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        # 应用注意力到value
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        # 加权融合
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        # 重新排列回原始形状 [b, c, d, h, w]
        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w',
                       head=self.num_heads, d=d, h=h, w=w)

        # 3D卷积输出投影
        out = self.project_out(out)

        return out

class MSCA_Block_GatedConv_With_SMFA_62(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, time_channels, embed_channels, LayerNorm_type='BiasFree'):
        super().__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.embed_channels = embed_channels

        self.smfa1 = SMFA_3D(
            self.in_channels, self.in_channels, self.in_channels)
        self.smfa2 = SMFA_3D(
            self.out_channels, self.out_channels, self.out_channels)

        self.norm1 = LayerNorm(self.in_channels)
        self.norm2 = LayerNorm(self.in_channels)

        self.tim_embed1 = nn.Sequential(
            linear(self.time_channels, self.in_channels),
            nn.SiLU(),
            linear(self.in_channels, self.in_channels),
        )
        self.tim_embed2 = nn.Sequential(
            linear(self.time_channels, self.in_channels),
            nn.SiLU(),
            linear(self.in_channels, self.in_channels),
        )

        # Legacy, unused parameters kept for checkpoint/optimizer compatibility.
        self.text_embed1 = nn.Sequential(
            linear(self.embed_channels, self.out_channels),
            nn.SiLU(),
            linear(self.out_channels, self.out_channels),
        )
        self.text_embed2 = nn.Sequential(
            linear(self.embed_channels, self.out_channels),
            nn.SiLU(),
            linear(self.out_channels, self.out_channels),
        )

    def forward(self, Xt, t):
        b, c, _, _, _ = Xt.shape
        time_embed1 = self.tim_embed1(t)
        time_embed1 = time_embed1.reshape(b, c, 1, 1, 1)
        time_embed2 = self.tim_embed2(t)
        time_embed2 = time_embed2.reshape(b, c, 1, 1, 1)

        skip1 = Xt
        Xt = self.norm1(Xt)
        # data = Xt + time_embed1
        data = Xt * time_embed1 + time_embed2

        data = self.smfa1(data)
        data = data + skip1

        data = self.norm2(data)
        # Preserve the original effective modulation (both stages use time).
        data = data * time_embed1 + time_embed2
        data = self.smfa2(data)

        return data
