import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.models.model import Decomposer
from src.models.transformer.swin_transformer import SwinTransformer3D


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # Flatten image [B, C, H, W] -> [B, C, H*W]
        # Reshape to [B, H*W, C]
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(
            1, 2
        )  # --- [B, H*W, C]
        # x = x.view(-1, self.channels, x.shape[2] * x.shape[3]).swapaxes(1, 2)
        x_ln = self.ln(x)  # --- [B, H*W, C]
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)  # --- [B, H*W, C]
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(
            -1, self.channels, self.size, self.size
        )  # --- [B, H*W, C] -> [B, C, H*W] -> [B, C, H, W]
        # return attention_value.swapaxes(2, 1).view(
        #     -1, self.channels, x.shape[2], x.shape[2]
        # )


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            x_new = self.double_conv(x)
            return F.gelu(x + x_new)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        """
        x: [B, C, H, W]
        t: [B, time_dim]
        """
        x = self.maxpool_conv(x)  # --- [B, 2*C, H/2, W/2]
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )  # --- [B, 2*C, H/2, W/2]
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(
        self,
        c_in=3,
        c_out=3,
        time_dim=256,
        img_size=256,
        conditioning_config=None,
        device=None,
    ):
        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.conditioning_config = conditioning_config
        self.time_dim = time_dim
        self.img_size = img_size
        # self.inc = DoubleConv(c_in, 64)
        # self.down1 = Down(64, 128)
        # self.sa1 = SelfAttention(128, 32)
        # self.down2 = Down(128, 256)
        # self.sa2 = SelfAttention(256, 16)
        # self.down3 = Down(256, 256)
        # self.sa3 = SelfAttention(256, 8)

        # self.bot1 = DoubleConv(256, 512)
        # self.bot2 = DoubleConv(512, 512)
        # self.bot3 = DoubleConv(512, 256)

        # self.up1 = Up(512, 128)
        # self.sa4 = SelfAttention(128, 16)
        # self.up2 = Up(256, 64)
        # self.sa5 = SelfAttention(64, 32)
        # self.up3 = Up(128, 64)
        # self.sa6 = SelfAttention(64, 64)
        # self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        ######### Atempt 2 #########

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        # self.sa1 = SelfAttention(128, img_size // 2)
        self.down2 = Down(128, 256)
        # self.sa2 = SelfAttention(256, img_size // 4)
        self.down3 = Down(256, 256)
        # self.sa3 = SelfAttention(256, img_size // 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        # self.sa4 = SelfAttention(128, img_size // 4)
        self.up2 = Up(256, 64)
        # self.sa5 = SelfAttention(64, img_size // 2)
        self.up3 = Up(128, 64)
        # self.sa6 = SelfAttention(64, img_size)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        ######### Atempt 3 #########

        # self.inc = DoubleConv(c_in, 256)
        # self.down1 = Down(256, 512, emb_dim=time_dim)
        # self.sa1 = SelfAttention(512, 128)
        # self.down2 = Down(512, 1024, emb_dim=time_dim)
        # self.sa2 = SelfAttention(1024, 64)
        # self.down3 = Down(1024, 1024, emb_dim=time_dim)
        # self.sa3 = SelfAttention(1024, 32)

        # self.bot1 = DoubleConv(1024, 2048)
        # self.bot2 = DoubleConv(2048, 2048)
        # self.bot3 = DoubleConv(2048, 1024)

        # self.up1 = Up(2048, 512, emb_dim=time_dim)
        # self.sa4 = SelfAttention(512, 64)
        # self.up2 = Up(1024, 256, emb_dim=time_dim)
        # self.sa5 = SelfAttention(256, 128)
        # self.up3 = Up(512, 256, emb_dim=time_dim)
        # self.sa6 = SelfAttention(256, 256)
        # self.outc = nn.Conv2d(256, c_out, kernel_size=1)

        # Learnable projection of time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Condition on embeddings generated by the conditioning_model
        if not conditioning_config.model.swin.use_checkpoint:
            self.conditioning_model = SwinTransformer3D(
                patch_size=conditioning_config.model.swin.patch_size
            )
        else:
            self.conditioning_model = SwinTransformer3D(
                pretrained=conditioning_config.model.swin.checkpoint,
                patch_size=conditioning_config.model.swin.patch_size,
                frozen_stages=0,
            )
            print("Loaded SWIN checkpoint")
            print("-----------------")

        # # For 256x256 images
        # self.averagepool_3D = nn.AvgPool3d(kernel_size=(5, 8, 8), stride=1) # --- (B, C, 1, 1, 1)

        # For 128x128 images
        self.averagepool_3D = nn.AvgPool3d(
            kernel_size=(5, 4, 4), stride=1
        )  # --- (B, C, 1, 1, 1)
        self.condition_emb = nn.Linear(768, self.time_dim)

    def conditioning_encoding(self, conditioning_images):
        """Transfrom x from shape [B, 768, 5, 8, 8] to [B, 256] using Linear layers"""
        x = self.conditioning_model(conditioning_images)
        # map embeddings to time_dim
        # average pooling for [B, 768, 5, 8, 8] -> [B, 768, 1, 1, 1]
        x = self.averagepool_3D(x)
        # unsqueeze -> [B, 768]
        x = torch.squeeze(x)
        # linear layer: [B, 768] -> [B, time_dim]
        x = self.condition_emb(x)

        return x

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, conditioning_images):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        t = self.time_emb(t)

        # Condition on embeddings generated by the conditioning_model
        if conditioning_images is not None:
            input_seq_emb = self.conditioning_encoding(conditioning_images)
            t += input_seq_emb

        x1 = self.inc(x)  # --- [B, C, H, W]
        x2 = self.down1(x1, t)  # --- [B, 2*C, H/2, W/2]
        # x2 = self.sa1(x2)  # --- [B, 2*C, H/2, W/2]
        x3 = self.down2(x2, t)  # --- [B, 4*C, H/4, W/4]
        # x3 = self.sa2(x3)  # --- [B, 4*C, H/4, W/4]
        x4 = self.down3(x3, t)  # --- [B, 4*C, H/8, W/8]
        # x4 = self.sa3(x4)  # --- [B, 4*C, H/8, W/8]

        x4 = self.bot1(x4)  # --- [B, 8*C, H/8, W/8]
        x4 = self.bot2(x4)  # --- [B, 8*C, H/8, W/8]
        x4 = self.bot3(x4)  # --- [B, 4*C, H/8, W/8]

        x = self.up1(x4, x3, t)  # --- [B, 2*C, H/4, W/4]
        # x = self.sa4(x)  # --- [B, 2*C, H/4, W/4]
        x = self.up2(x, x2, t)  # --- [B, C, H/2, W/2]
        # x = self.sa5(x)  # --- [B, C, H/2, W/2]
        x = self.up3(x, x1, t)  # --- [B, C, H, W]
        # x = self.sa6(x)
        output = self.outc(x)
        return output


if __name__ == "__main__":
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)
