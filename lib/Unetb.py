from functools import partial

from thop.profile import register_hooks
from torch import einsum
from einops import rearrange
from lib.simple_diffusion import ResnetBlock
import torch
from torch.nn import Module
from mmcv.cnn import ConvModule
import torch.nn as nn
import math
import torch.nn.functional as F


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SpatialEnhancement(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(SpatialEnhancement, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.convadp = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        adp_out = self.convadp(x)
        x = torch.cat([max_out, mean_out, adp_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelEnhancement(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(ChannelEnhancement, self).__init__()
        mip = min(8, in_channel // ratio)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_max = nn.Sequential(
            nn.Conv2d(in_channel, mip, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mip, in_channel, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.max_pool(x)+self.avg_pool(x)
        x = self.conv_max(x)
        return x

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=1, padding=1, groups=in_channel)
        self.point_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class DirectionalAttentionUnit(nn.Module):
    def __init__(self, channel):
        super(DirectionalAttentionUnit, self).__init__()
        self.h_conv = nn.Conv2d(channel, 1, (1, 5), padding=(0, 2))
        self.w_conv = nn.Conv2d(channel, 1, (5, 1), padding=(2, 0))
        # leading diagonal
        self.dia19_conv = nn.Conv2d(channel, 1, (5, 1), padding=(2, 0))
        # reverse diagonal
        self.dia37_conv = nn.Conv2d(channel, 1, (1, 5), padding=(0, 2))
        self.conv = nn.Sequential(
            nn.Conv2d(channel*4, channel, kernel_size=1),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.GroupNorm(8, channel),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        x1 = self.h_conv(x)*x
        x2 = self.w_conv(x)*x
        x3 = self.inv_h_transform(self.dia19_conv(self.h_transform(x)))*x
        x4 = self.inv_v_transform(self.dia37_conv(self.v_transform(x)))*x
        x = torch.cat([x1,x2,x3,x4],dim=1)
        x = self.conv(x)
        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x.permute(0, 1, 3, 2)

class SpatialAttentionUnit(nn.Module):
    def __init__(self, channel):
        super(SpatialAttentionUnit, self).__init__()
        self.conv1 = nn.Conv2d(channel, 1, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(channel, 1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channel, 1, kernel_size=1, padding=0)
        self.conv= nn.Sequential(
            nn.Conv2d(channel * 4, channel, kernel_size=1),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.GroupNorm(8, channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)*x
        x2 = self.conv2(x)*x
        x3 = self.conv3(x)*x
        x4 = self.conv4(x)*x
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv(x)
        return x

class FSDE(Module):
    def __init__(self, dim, time_embed_dim):
        super(FSDE, self).__init__()
        self.time_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, dim)
        )
        self.conv_fu = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.conv_xt = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=False)
        self.SAU = SpatialAttentionUnit(dim)
        self.DAU = DirectionalAttentionUnit(dim)
        self.SE = SpatialEnhancement(dim)
        self.CE = ChannelEnhancement(dim)
        self.conv = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x, t, xt):
        B, C, H, W = x.shape
        xt = F.interpolate(xt, size=(H, W), mode='bilinear', align_corners=False)
        xt = self.conv_xt(xt)
        time_token = self.time_embed(t)
        time_token = time_token.unsqueeze(dim=1)
        time_token = time_token.unsqueeze(dim=1)
        time_token = time_token.transpose(3, 1)
        x = x + xt + time_token
        x = self.conv_fu(x)
        x_s = self.SAU(x)
        x_d = self.DAU(x)
        x = torch.cat([x_s, x_d], dim=1)
        x = self.conv(x)
        se = self.SE(x)
        x = x * se + x
        ce = self.CE(x)
        x = x * ce + x
        return x

class PFA(Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super(PFA, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = nn.Sequential(
            DepthWiseConv(dim4, dim3),
            nn.GroupNorm(8, dim3),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample2 = nn.Sequential(
            DepthWiseConv(dim4, dim2),
            nn.GroupNorm(8, dim2),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample3 = nn.Sequential(
            DepthWiseConv(dim3, dim2),
            nn.GroupNorm(8, dim2),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample4 = nn.Sequential(
            DepthWiseConv(dim4, dim1),
            nn.GroupNorm(8, dim1),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample5 = nn.Sequential(
            DepthWiseConv(dim3, dim1),
            nn.GroupNorm(8, dim1),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample6 = nn.Sequential(
            DepthWiseConv(dim2, dim1),
            nn.GroupNorm(8, dim1),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample7 = nn.Sequential(
            DepthWiseConv(dim4, dim4),
            nn.GroupNorm(8, dim4),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample8 = nn.Sequential(
            DepthWiseConv(dim3, dim3),
            nn.GroupNorm(8, dim3),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample9 = nn.Sequential(
            DepthWiseConv(dim2, dim2),
            nn.GroupNorm(8, dim2),
            nn.ReLU(inplace=True),
        )
        self.conv_concat2 = nn.Sequential(
            DepthWiseConv(dim4+dim3, dim3),
            nn.GroupNorm(8, dim3),
            nn.ReLU(inplace=True),
        )
        self.conv_concat3 = nn.Sequential(
            DepthWiseConv(dim3 + dim2, dim2),
            nn.GroupNorm(8, dim2),
            nn.ReLU(inplace=True),
        )
        self.conv_concat4 = nn.Sequential(
            DepthWiseConv(dim2 + dim1, dim1),
            nn.GroupNorm(8, dim1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x1, x2, x3, x4):
        x4_1 = x4
        x3_1 = self.conv_upsample1(self.upsample(x4)) * x3
        x2_1 = self.conv_upsample2(self.upsample(self.upsample(x4)))* self.conv_upsample3(self.upsample(x3)) * x2
        x1_1 = (self.conv_upsample4(self.upsample(self.upsample(self.upsample(x4))))
                * self.conv_upsample5(self.upsample(self.upsample(x3))) * self.conv_upsample6(self.upsample(x2)) * x1)

        x3_2 = torch.cat((x3_1, self.conv_upsample7(self.upsample(x4_1))), 1)
        x3_2 = self.conv_concat2(x3_2)

        x2_2 = torch.cat((x2_1, self.conv_upsample8(self.upsample(x3_2))), 1)
        x2_2 = self.conv_concat3(x2_2)

        x1_2 = torch.cat((x1_1, self.conv_upsample9(self.upsample(x2_2))), 1)
        x1_2 = self.conv_concat4(x1_2)
        return [x1_2, x2_2, x3_2, x4_1]

class decoder(nn.Module):
    def __init__(self, dim_x, dim_f):
        super().__init__()
        dim1, dim2, dim3, dim4 = dim_f[0], dim_f[1], dim_f[2], dim_f[3]
        d1, d2, d3, d4 = dim_x[0], dim_x[1], dim_x[2], dim_x[3]
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1x1_fuse1 = nn.Conv2d(dim4 + d4, d4, kernel_size=3, padding=1, bias=False)
        self.conv1x1_fuse21 = nn.Conv2d(dim3 + d3, d3, kernel_size=3, padding=1, bias=False)
        self.conv1x1_fuse22 = nn.Conv2d(dim3 + d3, d3, kernel_size=3, padding=1, bias=False)
        self.conv1x1_fuse31 = nn.Conv2d(dim2 + d2, d2, kernel_size=3, padding=1, bias=False)
        self.conv1x1_fuse32 = nn.Conv2d(dim2 + d2, d2, kernel_size=3, padding=1, bias=False)
        self.conv1x1_fuse41 = nn.Conv2d(dim1 + d1, d1, kernel_size=3, padding=1, bias=False)
        self.conv1x1_fuse42 = nn.Conv2d(dim1 + d1, d1, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(dim4, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(dim3, 1, kernel_size=1)
        self.conv3 = nn.Conv2d(dim2, 1, kernel_size=1)
        self.conv4 = nn.Conv2d(dim1, 1, kernel_size=1)
        self.convup1 = nn.Sequential(
            nn.Conv2d(d4, d3, kernel_size=1),
            self.upsample
        )
        self.convup2 = nn.Sequential(
            nn.Conv2d(d3*2, d2, kernel_size=1),
            self.upsample
        )
        self.convup3 = nn.Sequential(
            nn.Conv2d(d2 * 2, d1, kernel_size=1),
            self.upsample
        )
        self.convup4 = nn.Sequential(
            nn.Conv2d(d1 * 2, d1, kernel_size=1),
            self.upsample
        )

    def forward(self, x, f):
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        f1, f2, f3, f4 = f[0], f[1], f[2], f[3]
        f1 = F.interpolate(f1, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=False)
        f2 = F.interpolate(f2, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=(x3.shape[2], x3.shape[3]), mode='bilinear', align_corners=False)
        f4 = F.interpolate(f4, size=(x4.shape[2], x4.shape[3]), mode='bilinear', align_corners=False)

        f4_1 = self.conv1(f4)
        x4_1 = torch.cat([f4, x4], dim=1)
        x4_1 = self.conv1x1_fuse1(x4_1)
        x4_1 = x4_1 * f4_1 + x4_1
        x4_1 = self.convup1(x4_1)

        f3_1 = self.conv2(f3)
        x3_1 = torch.cat([x4_1, f3], dim=1)
        x3_1 = self.conv1x1_fuse21(x3_1)
        x3_2 = torch.cat([x3, f3], dim=1)
        x3_2 = self.conv1x1_fuse22(x3_2)
        x3_1 = x3_1 * f3_1 + x3_1
        x3_2 = x3_2 * f3_1 + x3_2
        x3_3 = torch.cat([x3_1, x3_2], dim=1)
        x3_3 = self.convup2(x3_3)

        f2_1 = self.conv3(f2)
        x2_1 = torch.cat([x3_3, f2], dim=1)
        x2_1 = self.conv1x1_fuse31(x2_1)
        x2_2 = torch.cat([x2, f2], dim=1)
        x2_2 = self.conv1x1_fuse32(x2_2)
        x2_1 = x2_1 * f2_1 + x2_1
        x2_2 = x2_2 * f2_1 + x2_2
        x2_3 = torch.cat([x2_1, x2_2], dim=1)
        x2_3 = self.convup3(x2_3)

        f1_1 = self.conv4(f1)
        x1_1 = torch.cat([x2_3, f1], dim=1)
        x1_1 = self.conv1x1_fuse41(x1_1)
        x1_2 = torch.cat([x1, f1], dim=1)
        x1_2 = self.conv1x1_fuse42(x1_2)
        x1_1 = x1_1 * f1_1 + x1_1
        x1_2 = x1_2 * f1_1 + x1_2
        x1_3 = torch.cat([x1_1, x1_2], dim=1)
        x_out = self.convup4(x1_3)
        return x_out

class Conv_down(nn.Module):
    def __init__(self, dim1, dim2, do_down=False):
        super().__init__()
        if do_down:
            self.conv = nn.Sequential(
                nn.Conv2d(dim1, dim2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, dim2),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim2, dim2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, dim2),
                nn.ReLU(inplace=True),
                DepthWiseConv(dim2, dim2),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(dim1, dim2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, dim2),
                nn.ReLU(inplace=True),
                DepthWiseConv(dim2, dim2),
            )
    def forward(self, x):
        x = self.conv(x)
        return x

class FGDiff(Module):
    def __init__(self, dims, dim_input=1, embedding_dim=256, dim_output=1):
        super(FGDiff, self).__init__()
        dim1, dim2, dim3, dim4 = dims[0], dims[1], dims[2], dims[3]
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.time_embed_dim = embedding_dim

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )
        self.FSDE1 = FSDE(dim1, embedding_dim)
        self.FSDE2 = FSDE(dim2, embedding_dim)
        self.FSDE3 = FSDE(dim3, embedding_dim)
        self.FSDE4 = FSDE(dim4, embedding_dim)
        resnet_block = partial(ResnetBlock, groups=8)
        self.down = nn.Sequential(
            Conv_down(dim_input, 32, do_down=True),
            resnet_block(32, 32, time_emb_dim=self.time_embed_dim),

            Conv_down(32, 64),
            resnet_block(64, 64, time_emb_dim=self.time_embed_dim),

            Conv_down(64, 128),
            resnet_block(128, 128, time_emb_dim=self.time_embed_dim),

            Conv_down(128, 256),
            resnet_block(256, 256, time_emb_dim=self.time_embed_dim),
        )
        self.PFA = PFA(dim1, dim2, dim3, dim4)
        self.decoder = decoder([32, 64, 128, 256], dims)
        self.pred = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.dim_output, kernel_size=1),
        )

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.get_default_dtype()

    def forward(self, input_features, timesteps, xt):
        B, C, H, W = xt.shape
        xt, sal_map = torch.chunk(xt, 2, 1)
        x = xt+sal_map*xt
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
        f = input_features
        f1, f2, f3, f4 = f[0], f[1], f[2], f[3]
        ############################################## 特征处理模块
        r1 = self.FSDE1(f1, t, sal_map)
        r2 = self.FSDE2(f2, t, sal_map)
        r3 = self.FSDE3(f3, t, sal_map)
        r4 = self.FSDE4(f4, t, sal_map)
        r = self.PFA(r1, r2, r3, r4)
        ##############################################
        _x = []
        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)
        x = self.decoder(_x, r)
        out = self.pred(x)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
