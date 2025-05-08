
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F


MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))

sys.path.insert(0, MODULE_PATH)

class SpeckleDisplacementNet(nn.Module):
    """ Neural network for predicting speckle pattern displacement (dx, dy) from reference and distorted images."""
    def __init__(self):
        super(SpeckleDisplacementNet, self).__init__()
        # Physics-informed: Sobel filters for computing image gradients (fixed weights)
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)  # horizontal edge-detector kernel
        self.register_buffer('sobel_y', sobel_y)  # vertical edge-detector kernel

        # Encoder: Convolutional layers for feature extraction (with down-sampling)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=7, stride=1, padding=3)     # 7x7 conv
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)    # 5x5 conv, stride 2 (downsample by 2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)   # 3x3 conv, stride 2 (downsample)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # 3x3 conv, stride 2 (downsample)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)  # 3x3 conv, stride 2 (downsample)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # 3x3 conv (bottleneck, no further down-sampling)

        # Decoder: Transposed conv for up-sampling and conv for merging with skip connections
        self.upconv5 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)  # upsample (1/16 -> 1/8)
        self.iconv5  = nn.Conv2d(in_channels=256+256, out_channels=128, kernel_size=3, padding=1)                 # merge with skip from conv4
        self.upconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)  # upsample (1/8 -> 1/4)
        self.iconv4  = nn.Conv2d(in_channels=128+128, out_channels=64, kernel_size=3, padding=1)                  # merge with skip from conv3
        self.upconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)    # upsample (1/4 -> 1/2)
        self.iconv3  = nn.Conv2d(in_channels=64+64, out_channels=32, kernel_size=3, padding=1)                    # merge with skip from conv2
        self.upconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)    # upsample (1/2 -> 1/1)
        self.iconv2  = nn.Conv2d(in_channels=32+32, out_channels=32, kernel_size=3, padding=1)                    # merge with skip from conv1

        # Final output layer: predicts 2-channel displacement (dx, dy) at full resolution
        self.predict_flow = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x):
        """ Forward pass for the network.
                param x: Input tensor of shape [B, 2, H, W] containing reference and distorted images.
                return: Tensor of shape [B, 2, H, W] containing predicted displacement (dx, dy) for each pixel. """
        # Split input into reference and distorted images
        ref       = x[:, 0:1, :, :]  # shape [B,1,H,W]
        distorted = x[:, 1:2, :, :]  # shape [B,1,H,W]

        # Compute image gradients (physics-inspired features)
        gradx_ref  = F.conv2d(ref      , self.sobel_x, padding=1)  # ∂I_ref/∂x
        grady_ref  = F.conv2d(ref      , self.sobel_y, padding=1)  # ∂I_ref/∂y
        gradx_dist = F.conv2d(distorted, self.sobel_x, padding=1)  # ∂I_dist/∂x
        grady_dist = F.conv2d(distorted, self.sobel_y, padding=1)  # ∂I_dist/∂y

        # Concatenate original images and gradients to form input features for the encoder
        encoder_input = torch.cat([ref, distorted, gradx_ref, grady_ref, gradx_dist, grady_dist], dim=1)  # [B,6,H,W]

        # Encoder: convolutional feature extraction with down-sampling
        c1 = F.relu(self.conv1(encoder_input))   # [B,32,H, W]   – features at full resolution
        c2 = F.relu(self.conv2(c1))              # [B,64,H/2,W/2]
        c3 = F.relu(self.conv3(c2))              # [B,128,H/4,W/4]
        c4 = F.relu(self.conv4(c3))              # [B,256,H/8,W/8]
        c5 = F.relu(self.conv5(c4))              # [B,256,H/16,W/16]
        bottleneck = F.relu(self.conv6(c5))      # [B,256,H/16,W/16] – deepest features

        # Decoder: up-sample and refine with skip connections from encoder
        up5    = F.relu(self.upconv5(bottleneck))  # [B,256, H/8, W/8]
        merge5 = torch.cat([up5, c4], dim=1)       # concat skip from conv4: [B,256+256, H/8, W/8]
        iconv5 = F.relu(self.iconv5(merge5))       # [B,128, H/8, W/8]

        up4    = F.relu(self.upconv4(iconv5))      # [B,128, H/4, W/4]
        merge4 = torch.cat([up4, c3], dim=1)       # concat skip from conv3: [B,128+128, H/4, W/4]
        iconv4 = F.relu(self.iconv4(merge4))       # [B,64,  H/4, W/4]

        up3    = F.relu(self.upconv3(iconv4))      # [B,64,  H/2, W/2]
        merge3 = torch.cat([up3, c2], dim=1)       # concat skip from conv2: [B,64+64, H/2, W/2]
        iconv3 = F.relu(self.iconv3(merge3))       # [B,32,  H/2, W/2]

        up2    = F.relu(self.upconv2(iconv3))      # [B,32,  H,   W]
        merge2 = torch.cat([up2, c1], dim=1)       # concat skip from conv1: [B,32+32, H,   W]
        iconv2 = F.relu(self.iconv2(merge2))       # [B,32,  H,   W]

        # Predict displacement field (no activation on output; regression can produce positive or negative values)
        flow = self.predict_flow(iconv2)           # [B,2,   H,   W], two channels for (dx, dy)
        return flow

