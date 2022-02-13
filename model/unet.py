import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LitUNet(pl.LightningModule): # UNet
    def __init__(self, dim, in_channels=3, out_channels=1, init_features=16):
        super(LitUNet, self).__init__()

        if (dim == 3):
            features = init_features
            self.encoder1 = LitUNet._block(in_channels, features)
            self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder2 = LitUNet._block(features, features)
            self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder3 = LitUNet._block(features, features)
            self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder4 = LitUNet._block(features, features)
            self.pool4 = nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder5 = LitUNet._block(features, features)
            self.pool5 = nn.AvgPool3d(kernel_size=2, stride=2)
            
            self.bottleneck = LitUNet._block(features, features)
            
            self.upconv5 = nn.ConvTranspose3d(
                features, features, kernel_size=2, stride=2
            )
            self.decoder5 = LitUNet._block(features * 2, features)
            
            self.upconv4 = nn.ConvTranspose3d(
                features, features, kernel_size=2, stride=2
            )
            self.decoder4 = LitUNet._block(features * 2, features)
            
            self.upconv3 = nn.ConvTranspose3d(
                features, features, kernel_size=2, stride=2
            )
            self.decoder3 = LitUNet._block(features * 2, features)
            
            self.upconv2 = nn.ConvTranspose3d(
                features, features, kernel_size=2, stride=2
            )
            self.decoder2 = LitUNet._block(features * 2, features)
            
            self.upconv1 = nn.ConvTranspose3d(
                features, features, kernel_size=2, stride=2
            )
            self.decoder1 = LitUNet._block(features * 2, features)

            self.conv = nn.Conv3d(
                in_channels=features, out_channels=out_channels, kernel_size=1
            )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        
        bottleneck = self.bottleneck(self.pool5(enc5))

        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
    
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1, 
                bias=False,
                padding_mode='circular'),
            nn.BatchNorm3d(features),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode='circular'),
            nn.BatchNorm3d(features),
            nn.ReLU()
            )
