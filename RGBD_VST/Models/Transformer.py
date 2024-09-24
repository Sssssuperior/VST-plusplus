import torch
from torch import nn
from .transformer_block import MutualSelfBlock
from .transformer_block import Block,PositionEmbeddingSine,Block_decoder
from timm.models.layers import trunc_normal_
from einops import rearrange

class TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                 MutualSelfBlock(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])

        self.rgb_norm = norm_layer(embed_dim)
        self.depth_norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, rgb_fea, depth_fea):

        for block in self.blocks:
            rgb_fea, depth_fea = block(rgb_fea, depth_fea)

        rgb_fea = self.rgb_norm(rgb_fea)
        depth_fea = self.depth_norm(depth_fea)

        return rgb_fea, depth_fea


class token_TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, token_dim = 64,mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(token_TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                 Block_decoder(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])
        N_steps = embed_dim // 3
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.norm = norm_layer(embed_dim)
        self.mlp3 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, fea,  saliency_tokens, contour_tokens, sal_PE, con_PE, back_PE, depth, zoom, mask=None):
    
        _, HW, _ = fea.shape # B, 196, 384
        # fea_pos B, 384, 14, 14
        # depth B, 1, 14, 14
        depth = rearrange(depth, 'b c h w -> b (h w) c')
        fea_pos = rearrange(fea,'b (h w) c -> b c h w', h = int(torch.sqrt(torch.tensor(HW).to(torch.double))))
        # patch_PE B, 384, 14, 14 -> B, 196, 384
        patch_PE = self.pe_layer(fea_pos, depth, zoom, None)
        patch_PE = rearrange(patch_PE,'b c h w -> b (h w) c')
        fea = torch.cat((saliency_tokens, fea), dim=1)
        fea = torch.cat((fea, contour_tokens), dim=1)
        fea_pos = torch.cat((sal_PE, patch_PE), dim=1)
        fea_pos = torch.cat((fea_pos, con_PE), dim=1)
        fea_pos = torch.cat((fea_pos, back_PE), dim=1)
        
        if mask != None:
            for block in self.blocks:
                fea = block(fea, fea_pos, mask)
        else:
            for block in self.blocks:
                fea = block(fea, fea_pos)

        saliency_tokens = fea[:, 0, :].unsqueeze(1)
        contour_tokens = fea[:, -1, :].unsqueeze(1)
        fea_output = fea[:,1:-1,:]
        fea_tmp = self.mlp3(self.norm(fea_output))

        return fea_output,saliency_tokens,contour_tokens,fea_tmp,fea


class Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(Transformer, self).__init__()

        self.encoderlayer = TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)

    def forward(self, rgb_fea, depth_fea):

        rgb_memory, depth_memory = self.encoderlayer(rgb_fea, depth_fea)

        return rgb_memory, depth_memory


class saliency_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, 0, :].unsqueeze(1), x[:, 1:-1, :]
        # T_s [B, 1, 384]  F_s [B, 14*14, 384]

        q = self.q(F_s).reshape(B, N-2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-2, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, 1:-1, :]
        return infer_fea


class contour_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, -1, :].unsqueeze(1), x[:, 1:-1, :]
        # T_s [B, 1, 384]  F_s [B, 14*14, 384]

        q = self.q(F_s).reshape(B, N-2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-2, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, 1:-1, :]
        return infer_fea


class token_Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.,in_dim=64):
        super(token_Transformer, self).__init__()

        self.norm = nn.LayerNorm(embed_dim * 2)
        self.mlp_s = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.saliency_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.contour_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.sal_PE = nn.Embedding(1, embed_dim) #1,384
        self.con_PE = nn.Embedding(1, embed_dim)
        self.back_PE = nn.Embedding(1, embed_dim)
        self.zoom_16 = nn.Parameter(torch.ones(1)*0.5)
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,token_dim=in_dim)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)
        self.contour_token_pre = contour_token_inference(dim=embed_dim, num_heads=1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, in_dim),
        )
        self.norm1_c = nn.LayerNorm(embed_dim)
        self.mlp1_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, in_dim),
        )
    def forward(self, rgb_fea, depth_fea, depth):
        B, _, _ = rgb_fea.shape
        fea_1_16 = torch.cat([rgb_fea, depth_fea], dim=2)
        fea_1_16 = self.mlp_s(self.norm(fea_1_16))   # [B, 14*14, 384]
        
        saliency_tokens = self.saliency_token.expand(B, -1, -1)
        contour_tokens = self.contour_token.expand(B, -1, -1)
        
        sal_PE =  self.sal_PE.weight.unsqueeze(1).repeat(B, 1, 1) #B,1,384
        con_PE = self.con_PE.weight.unsqueeze(1).repeat(B, 1, 1)
        back_PE = self.back_PE.weight.unsqueeze(1).repeat(B, 1, 1)
        fea_1_16,saliency_tokens,contour_tokens,fea_16,fea_1_16_s = self.encoderlayer(fea_1_16, saliency_tokens, contour_tokens, sal_PE, con_PE, back_PE,depth ,self.zoom_16)
       
        saliency_tokens_tmp = self.mlp1(self.norm1(saliency_tokens))
        contour_tokens_tmp = self.mlp1_c(self.norm1_c(contour_tokens))
        
        saliency_fea_1_16 = self.saliency_token_pre(fea_1_16_s)
        contour_fea_1_16 = self.contour_token_pre(fea_1_16_s)
        
        return  fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea_1_16, contour_fea_1_16, sal_PE, con_PE, back_PE