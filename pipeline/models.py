import torch
import torch.nn as nn
import torch.nn.functional as F

# Autoencoder Architecture
class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up5 = self.upconv_block(512, 256)
        self.up4 = self.upconv_block(256, 128)
        self.up3 = self.upconv_block(128, 64)
        self.up2 = self.upconv_block(64, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        # Encoder
        # print("init", x.shape)
        enc1 = self.enc1(x)
        # print("enc1", enc1.shape)
        pool1 = self.pool1(enc1)
        # print("pool1", pool1.shape)
        enc2 = self.enc2(pool1)
        # print("enc2", enc2.shape)
        pool2 = self.pool2(enc2)
        # print("pool2", pool2.shape)
        enc3 = self.enc3(pool2)
        # print("enc3", enc3.shape)
        pool3 = self.pool3(enc3)
        # print("pool3", pool3.shape)
        enc4 = self.enc4(pool3)
        # print("enc4", enc4.shape)
        pool4 = self.pool4(enc4)
        # print("pool4", pool4.shape)

        # Decoder
        up5 = self.up5(pool4)
        # print("up5", up5.shape)
        up4 = self.up4(up5)
        # print("up4", up4.shape)
        up3 = self.up3(up4)
        # print("up3", up3.shape)
        up2 = self.up2(up3)
        # print("up2", up2.shape)

        # Output
        out = self.out_conv(up2)
        # print("out", out.shape)
        out = torch.sigmoid(out)
        # print("sigmoid", out.shape)

        return out

# Instantiate the Autoencoder model
# in_channels = 3  # Assuming 3 input channels
# out_channels = 3  # Assuming 3 output channels
# model = AutoEncoder(in_channels, out_channels)
# print(model)