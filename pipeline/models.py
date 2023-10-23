import torch
import torch.nn as nn
import torch.nn.functional as F

# Autoencoder Architecture
class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Decoder
        self.up5 = self.upconv_block(512, 256)
        self.up4 = self.upconv_block(256, 128)
        self.up3 = self.upconv_block(128, 64)
        self.up2 = self.upconv_block(64, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)

        # Decoder
        up5 = self.up5(pool4)
        up4 = self.up4(up5)
        up3 = self.up3(up4)
        up2 = self.up2(up3)

        # Output
        out = self.out_conv(up2)
        out = torch.sigmoid(out)

        return out

# Instantiate the Autoencoder model
# in_channels = 3  # Assuming 3 input channels
# out_channels = 3  # Assuming 3 output channels
# model = AutoEncoder(in_channels, out_channels)
# print(model)