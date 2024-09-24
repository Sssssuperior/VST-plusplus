# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class decoderAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim , bias=qkv_bias)
        self.k = nn.Linear(dim, dim , bias=qkv_bias)
        self.v = nn.Linear(dim, dim , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def get_backtokens(self, x, mask):
        # salient = 1, reverse_mask B,1,784
        reverse_mask = mask.masked_fill(mask < 0.5, float(1.0)).masked_fill(mask >= 0.5, float(0.0))
        #B,1,784 * B,784,384
        eps = 1e-10
        back_tokens = (reverse_mask @ x) / (torch.sum(reverse_mask, dim=2) +eps).unsqueeze(2)
        return back_tokens

    def forward(self, x, pos, mask, mask_block):
        if mask != None:
            B, N, C = x.shape
            # x B,786,384 -> B,784,384
            # mask B,1,784
            back_tokens = self.get_backtokens(x[:,1:N-1,:], mask)#B,1,384
            kv_x = torch.cat((x, back_tokens),dim=1) #B,786+1,384
            
            q_x = self.with_pos_embed(x, pos[:,:N,:])
            k_x = self.with_pos_embed(kv_x, pos)

            q = self.q(q_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #B,786,1,384
            k = self.k(k_x).reshape(N+1, C) #787,384
            v = self.v(kv_x).reshape(N+1, C)
            
            mask = mask.squeeze(0).squeeze(0) #1,1,787->787
            mask_block[1:N-1] = mask
            mask_k = torch.nonzero(mask_block >= 0.5).squeeze(1)

            k = k.index_select(0, mask_k)
            v = v.index_select(0, mask_k)

            k = k.reshape(B, self.num_heads, -1, C // self.num_heads)  # 1,6,N,64
            v = v.reshape(B, self.num_heads, -1, C // self.num_heads)
            attn = (q @ k.transpose(-2, -1)) * self.scale  # B,num_heads,786,787
            
        else:
            B, N, C = x.shape
            qk_x = self.with_pos_embed(x,pos[:,:N,:])
            q = self.q(qk_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.k(qk_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale  # B,num_heads,784,784
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MutualAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.rgb_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_proj = nn.Linear(dim, dim)

        self.depth_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, rgb_fea, depth_fea):
        B, N, C = rgb_fea.shape

        rgb_q = self.rgb_q(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_k = self.rgb_k(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_v = self.rgb_v(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q [B, nhead, N, C//nhead]

        depth_q = self.depth_q(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_k = self.depth_k(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_v = self.depth_v(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # rgb branch
        rgb_attn = (rgb_q @ depth_k.transpose(-2, -1)) * self.scale
        rgb_attn = rgb_attn.softmax(dim=-1)
        rgb_attn = self.attn_drop(rgb_attn)

        rgb_fea = (rgb_attn @ depth_v).transpose(1, 2).reshape(B, N, C)
        rgb_fea = self.rgb_proj(rgb_fea)
        rgb_fea = self.proj_drop(rgb_fea)

        # depth branch
        depth_attn = (depth_q @ rgb_k.transpose(-2, -1)) * self.scale
        depth_attn = depth_attn.softmax(dim=-1)
        depth_attn = self.attn_drop(depth_attn)

        depth_fea = (depth_attn @ rgb_v).transpose(1, 2).reshape(B, N, C)
        depth_fea = self.depth_proj(depth_fea)
        depth_fea = self.proj_drop(depth_fea)

        return rgb_fea, depth_fea


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_decoder(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = decoderAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, pos, mask=None, mask_block=None):
        x = x + self.drop_path(self.attn(self.norm1(x),pos,mask, mask_block))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MutualSelfBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        # mutual attention
        self.norm1_rgb_ma = norm_layer(dim)
        self.norm2_depth_ma = norm_layer(dim)
        self.mutualAttn = MutualAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm3_rgb_ma = norm_layer(dim)
        self.norm4_depth_ma = norm_layer(dim)
        self.mlp_rgb_ma = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_depth_ma = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # rgb self attention
        self.norm1_rgb_sa = norm_layer(dim)
        self.selfAttn_rgb = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_rgb_sa = norm_layer(dim)
        self.mlp_rgb_sa = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # depth self attention
        self.norm1_depth_sa = norm_layer(dim)
        self.selfAttn_depth = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_depth_sa = norm_layer(dim)
        self.mlp_depth_sa = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, rgb_fea, depth_fea):

        # mutual attention
        rgb_fea_fuse, depth_fea_fuse = self.drop_path(self.mutualAttn(self.norm1_rgb_ma(rgb_fea), self.norm2_depth_ma(depth_fea)))

        rgb_fea = rgb_fea + rgb_fea_fuse
        depth_fea = depth_fea + depth_fea_fuse

        rgb_fea = rgb_fea + self.drop_path(self.mlp_rgb_ma(self.norm3_rgb_ma(rgb_fea)))
        depth_fea = depth_fea + self.drop_path(self.mlp_depth_ma(self.norm4_depth_ma(depth_fea)))

        # rgb self attention
        rgb_fea = rgb_fea + self.drop_path(self.selfAttn_rgb(self.norm1_rgb_sa(rgb_fea)))
        rgb_fea = rgb_fea + self.drop_path(self.mlp_rgb_sa(self.norm2_rgb_sa(rgb_fea)))

        # depth self attention
        depth_fea = depth_fea + self.drop_path(self.selfAttn_depth(self.norm1_depth_sa(depth_fea)))
        depth_fea = depth_fea + self.drop_path(self.mlp_depth_sa(self.norm2_depth_sa(depth_fea)))

        return rgb_fea, depth_fea


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, depth_fea, zoom, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        B, H, W = not_mask.shape
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        z_max = torch.max(depth_fea, 1)[0].unsqueeze(2)
        z_min = torch.min(depth_fea, 1)[0].unsqueeze(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
            z_embed = (((depth_fea - z_min) / (z_max -z_min + eps)) * self.scale).reshape(B,H,W)
            z_embed = torch.ceil(z_embed*H)/H

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
        pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
        pos_z = (torch.stack(
                (pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()), dim=4
            ).flatten(3))*zoom
        pos = torch.cat((pos_y, pos_x, pos_z), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

