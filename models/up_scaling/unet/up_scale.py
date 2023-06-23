from torch import nn
from models.up_scaling.unet.unet3d import DoubleConv, Decoder, create_decoders
import torch


class UpSampler(nn.Module):
    """
    Module for upscaling output features from video swin transformer. It firstly applies upsampling with skip connection following U-Net architecture. After reaching size [B, C, T/2, H/4, W/4], we apply same upsampling layers, however without skip connections resulting in [B, C_, T, H, W] dim. Finally, a 1x1 kernel conv. layer is applied to obtain desired number of output channels without changing T, H and W.

    Args:
        f_maps (int): Channel size of feature maps in encoder part
        conv_kernel_size (int or tuple): size of the convolving kernel
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        is3d (Boolean): use 3dConv layers.
        out_channels (int): number of output channels
        skipless_scale_factor (int, tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation from the corresponding encoder

        skipless_size (tuple): List containing size for each layer without skip connections. Each has dimensions [input_num_chan, output_num_chan, T, H W].

    """

    def __init__(
        self,
        f_maps,
        conv_kernel_size,
        conv_padding,
        layer_order,
        num_groups,
        is3d,
        output_dim,
        skipless_scale_factor,
        skipless_size,
        omit_skip_connections,
    ):
        super(UpSampler, self).__init__()
        self.decoders = create_decoders(
            f_maps,
            DoubleConv,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            is3d,
        )
        self.omit_skip_connections = omit_skip_connections
        self.skipless_size = skipless_size
        # Upscaling layers without skip connections (no data available from encoder)

        self.layers_no_skip = []
        for size in skipless_size:
            layer = Decoder(
                size[0],
                size[1],
                basic_module=DoubleConv,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
                is3d=is3d,
                scale_factor=skipless_scale_factor,
            )
            self.layers_no_skip.append(layer)

        self.layers_no_skip = nn.ModuleList(self.layers_no_skip)

        # Final conv layer to reduce number of channels using 1x1 kernel
        self.final_layer = nn.Conv3d(skipless_size[-1][1], output_dim, kernel_size=1)

    def forward(self, encoder_features, x):
        # Upscale first layers with information from encoder
        for decoder, features in zip(self.decoders, encoder_features):
            # replace features with 0 to remove skip connections
            if self.omit_skip_connections:
                features = features**2 * 0
            # pass the output from the corresponding encoder and the output
            # of the previous decoder

            x = decoder(features, x)
        # current dimensionality: Batch x 96 x 5 x 64 x 64

        # Apply final upscaling layers without skip connections
        for idx, layer in enumerate(self.layers_no_skip):
            output_size = self.skipless_size[idx]
            dummy_features = torch.zeros(
                (1, 1, output_size[2], output_size[3], output_size[4])
            )
            dummy_features = dummy_features.to(x.device)
            x = layer(dummy_features, x, skip_joining=True)
        # Dim. after upscaling: Batch x _ x 10 x 256 x 256

        # Apply final conv layer
        x = self.final_layer(x)
        # Final dimensions: Batch x output_dim x 10 x 256 x 256
        return x
