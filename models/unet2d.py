import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)

class UNet2D(nn.Module):
    """
    2D Residual U-Net for BraTS Multi-modal Slice Segmentation.
    Input: (B, 4, H, W) -> [FLAIR, T1, T1ce, T2]
    Output: (B, 4, H, W) -> [Logits for classes 0, 1, 2, 3]
    """
    def __init__(self, in_channels=4, num_classes=4, base_filters=32):
        super().__init__()
        
        # Encoder
        self.enc1 = ResBlock2D(in_channels, base_filters)         # 32
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResBlock2D(base_filters, base_filters * 2)     # 64
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ResBlock2D(base_filters * 2, base_filters * 4) # 128
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ResBlock2D(base_filters * 4, base_filters * 8) # 256
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResBlock2D(base_filters * 8, base_filters * 16) # 512
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ResBlock2D(base_filters * 16, base_filters * 8)
        
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResBlock2D(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResBlock2D(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResBlock2D(base_filters * 2, base_filters)
        
        # Final classifier head
        self.final_conv = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.final_conv(d1)

if __name__ == "__main__":
    model = UNet2D(in_channels=4, num_classes=4)
    dummy_input = torch.randn(2, 4, 240, 240)
    out = model(dummy_input)
    print("UNet2D Forward Pass Test:")
    print("Input:", dummy_input.shape, "-> Output:", out.shape)
