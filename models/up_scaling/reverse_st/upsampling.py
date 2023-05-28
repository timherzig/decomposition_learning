import torch
import torch.nn as nn
from models.transformer.swin_transformer import SwinTransformer3D, BasicLayer
from einops import rearrange


class PatchSplitting(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm): 
        super().__init__()
        self.dim = dim
        self.expansion = nn.Linear(dim, 2*dim, bias=False)
        self.norm = norm_layer(dim//2)

    def forward(self, x): 

        x = self.expansion(x)
        x0, x1, x2, x3 = torch.split(x, x.shape[-1]//4, dim = -1)
        merged = torch.empty((x0.shape[0], x0.shape[1], x0.shape[2]*2, x0.shape[3]*2, x0.shape[4]))
        merged[:, :, 0::2, 0::2, :] = x0
        merged[:, :, 0::2, 1::2, :] = x1
        merged[:, :, 1::2, 0::2, :] = x2
        merged[:, :, 1::2, 1::2, :] = x3

        merged = self.norm(merged)
        #print(merged.shape)

        return merged

class SwinTransformer3D_up(SwinTransformer3D):
    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4,4,4),
                 in_chans=3,      
                 embed_dim=768,   # changed
                 depths=[1, 1, 1, 1], #changed 
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()
        
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed3D(
        #     patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim / 2**i_layer),   #changed from *2 to /2
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchSplitting if i_layer<self.num_layers-1 else None,  #changed
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim / 2**(self.num_layers-1))   #changed from *2 to /2

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self._freeze_stages()

    def _freeze_stages(self):
        # if self.frozen_stages >= 0:
        #     self.patch_embed.eval()
        #     for param in self.patch_embed.parameters():
        #         param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def forward(self, x):
        """Forward function."""
        #x = self.patch_embed(x)
        #print("after patch parition: ", x.shape)
        #x = self.pos_drop(x)     # needed? 

        for idx, layer in enumerate(self.layers):
            x, _ = layer(x.contiguous())
            print("Layer nr:" ,str(idx), " shape: ", x.shape)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        proj = torch.nn.ConvTranspose3d(x.shape[1], 3, (2,4,4), (2,4,4), dilation=(1,1,1))
        x = proj(x)
        print(x.shape)

        return x

        

# if __name__ == "__main__":
#     x = torch.rand(10,8,8,8,768)
#     for i in range(4):
#         run = PatchSplitting(x.shape[4])
#         x = run.forward(x)
        
        

        
