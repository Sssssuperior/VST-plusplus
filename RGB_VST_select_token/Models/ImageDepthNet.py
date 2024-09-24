import torch.nn as nn
from .t2t_vit import T2t_vit_t_14
from .swin_transformer import swin_transformer_T, swin_transformer_S, swin_transformer_B
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder,decoder_module


class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()

        # VST Encoder
        # VST Encoder
        # choose backbone version from [swin_transformer_T, swin_transformer_S, swin_transformer_B] and T2t_vit_t_14
        self.rgb_backbone = swin_transformer_T(pretrained =True,args = args)

        self.mlp32 = nn.Sequential(
                nn.Linear(768, 384),
                nn.GELU(),
                nn.Linear(384, 384),)
                
        self.mlp16 = nn.Sequential(
                nn.Linear(384, 64),
                nn.GELU(),
                nn.Linear(64, 64),)
                
        self.norm1 = nn.LayerNorm(64)
        self.mlp1 = nn.Sequential(
            nn.Linear(64, 384),
            nn.GELU(),
            nn.Linear(384, 384),
        )
        self.fuse_32_16 = decoder_module(dim=384, token_dim=64, img_size=args.img_size, ratio=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        
        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)

    def forward(self, image_Input):

        # VST Encoder
        rgb_fea_1_4 , rgb_fea_1_8, rgb_fea_1_16, rgb_fea_1_32 = self.rgb_backbone(image_Input)
        rgb_fea_1_32 = self.mlp32(rgb_fea_1_32)
        rgb_fea_1_16 = self.mlp16(rgb_fea_1_16)
        rgb_fea_1_16 = self.fuse_32_16(rgb_fea_1_32, rgb_fea_1_16)
        rgb_fea_1_16 = self.mlp1(self.norm1(rgb_fea_1_16))

        # VST Convertor
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)

        # VST Decoder
        fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea_1_16, contour_fea_1_16, sal_PE, con_PE, back_PE = self.token_trans(rgb_fea_1_16)
        outputs = self.decoder(fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea_1_16, contour_fea_1_16, rgb_fea_1_8, rgb_fea_1_4,  sal_PE, con_PE, back_PE)

        return outputs
