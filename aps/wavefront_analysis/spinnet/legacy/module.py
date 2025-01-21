#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    	: 08 / 12 / 2022
@Author  	: Zhi Qiao
@Contact	: z.qiao1989@gmail.com
@File    	: module.py
@Software	: SPINNet
@Desc		: SPINNet neural network module
'''


import torch
import torch.nn.functional as F

def conv(batch_norm,
         in_channels,
         out_channels,
         kernel_size=3,
         stride=1,
         dilation=1,
         activation='LeakyReLu'):
    # the basic structure of the network.
    if batch_norm:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            dilation=dilation,
                            padding=((kernel_size - 1) * dilation) // 2,
                            bias=False), torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            if activation == 'LeakyReLu' else torch.nn.ReLU(inplace=False)
            )
    else:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            dilation=dilation,
                            padding=((kernel_size - 1) * dilation) // 2,
                            bias=True),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            if activation == 'LeakyReLu' else torch.nn.ReLU(inplace=False))

def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(
        1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(
        1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)

    return grid


class FeatureExtractor(torch.nn.Module):
    # extract image feature with pyramid
    #  return pramid level: [6, 5, 4, 3, 2, 1]
    def __init__(self, argv):
        super(FeatureExtractor, self).__init__()
        self.argv = argv

        self.NetFeature = []
        for l, (ch_in,
                ch_out) in enumerate(zip(argv.lv_chs[:-1], argv.lv_chs[1:])):
            layer = torch.nn.Sequential(
                conv(argv.batch_norm, ch_in, ch_out, stride=2),
                conv(argv.batch_norm, ch_out, ch_out))
            self.add_module(f'Feature(Lv{l})', layer)
            self.NetFeature.append(layer)

    # end

    def forward(self, img):
        feature_pyramid = []
        for net in self.NetFeature:
            img = net(img)
            feature_pyramid.append(img)

        return feature_pyramid[::-1]


class costvol_layer(torch.nn.Module):
    # the layer to calculate the corrlation volume between two images
    def __init__(self, argv) -> None:
        super(costvol_layer, self).__init__()
        self.search_range = argv.search_range
        self.argv = argv

    def forward(self, first, second):
        """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
        Args:
            first: Level of the feature pyramid of Image1
            second: Warped level of the feature pyramid of image22
            search_range: Search range (maximum displacement)
        """
        padded_lvl = torch.nn.functional.pad(
            second, (self.search_range, self.search_range, self.search_range,
                     self.search_range)).to(self.argv.device)
        _, _, h, w = first.shape
        max_offset = self.search_range * 2 + 1

        cost_vol = []
        for y in range(0, max_offset):
            for x in range(0, max_offset):
                second_slice = padded_lvl[:, :, y:y + h, x:x + w]
                cost = torch.mean(first * second_slice, dim=1, keepdim=True)
                cost_vol.append(cost)
        cost_vol = torch.cat(cost_vol, dim=1).to(self.argv.device)

        return cost_vol
        

class Warping_layer(torch.nn.Module):
    # the warping layer to wrap image with the flow
    def __init__(self, argv) -> None:
        super(Warping_layer, self).__init__()
        self.argv = argv

    def forward(self, x, flow):
        argv = self.argv
        # WarpingLayer uses F.grid_sample, which expects normalized grid
        flow_for_grip = torch.zeros_like(flow)
        flow_for_grip[:, 0, :, :] = flow[:, 0, :, :] / (
            (flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:, 1, :, :] = flow[:, 1, :, :] / (
            (flow.size(2) - 1.0) / 2.0)

        x_shape = x.size()

        tenHorizontal = torch.linspace(-1.0, 1.0, x_shape[3]).view(1, 1, 1, x_shape[3]).expand(x_shape[0], -1, x_shape[2], -1)

        tenVertical = torch.linspace(-1.0, 1.0, x_shape[2]).view(1, 1, x_shape[2], 1).expand(x_shape[0], -1, -1, x_shape[3])

        if self.argv.device == 'cuda':
            grid_x = torch.cat([tenHorizontal, tenVertical], 1).type(x.type()).cuda()

        else:
            grid_x = torch.cat([tenHorizontal, tenVertical], 1)

        grid = (grid_x - flow_for_grip).permute(0, 2, 3, 1)
        x_warp = torch.nn.functional.grid_sample(x,
                                                 grid,
                                                 mode='bilinear',
                                                 padding_mode='zeros',
                                                 align_corners=True)

        return x_warp


class phaseC_layer(torch.nn.Module):
    # the calculate phase induced intensity changing
    def __init__(self, argv) -> None:
        super(phaseC_layer, self).__init__()
        self.argv = argv

    def forward(self, flow):

        if True:
            return 1.0
        else:
            argv = self.argv
            if self.argv.device == 'cuda':

                kernel_x = torch.tensor([[0., 0., 0.],
                                [-0.5, 0., 0.5],
                                [0., 0., 0.]]).type(flow.type()).cuda()
                kernel_y = torch.tensor([[0., -0.5, 0.],
                                    [0., 0., 0.],
                                    [0., 0.5, 0.]]).type(flow.type()).cuda()
                lower_limit = torch.tensor(1e-1).type(flow.type()).cuda()
            else:
                kernel_x = torch.tensor([[0., 0., 0.],
                                        [-0.5, 0., 0.5],
                                        [0., 0., 0.]])
                kernel_y = torch.tensor([[0., -0.5, 0.],
                                        [0., 0., 0.],
                                        [0., 0.5, 0.]])
                lower_limit = torch.tensor(1e-1)

            kernel_x = kernel_x.view(1, 1, 3, 3)
            kernel_y = kernel_y.view(1, 1, 3, 3)

            phase_c = 1.0 + F.conv2d(input=flow[:, 0:1, :, :], weight=kernel_x, padding=1) +\
                        F.conv2d(input=flow[:, 1:2, :, :], weight=kernel_y, padding=1)
            
            return torch.maximum(phase_c, lower_limit)


class FlowEstimator(torch.nn.Module):
    # Estimator: combine the costvol, flow, T, ref to get estimated flow
    def __init__(self, argv, in_ch):
        # in_ch: the input channels
        super(FlowEstimator, self).__init__()
        self.argv = argv

        self.NetMain_flow = torch.nn.Sequential(
            conv(argv.batch_norm, in_ch, 128), conv(argv.batch_norm, 128, 128),
            conv(argv.batch_norm, 128, 96), conv(argv.batch_norm, 96, 64),
            conv(argv.batch_norm, 64, 32),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=2,
                            kernel_size=3,
                            stride=1,
                            padding=1))

    def forward(self, x):
        return self.NetMain_flow(x)


class Refiner_flow(torch.nn.Module):
    # refiner: refine the flow with subpixel resolution
    def __init__(self, argv, ch_feature, upsampling=False):
        #
        # ch_feature: the feature pyramid's channel
        # upsampling: if true, the feature pyramid are upsampled to get finer flow and T
        super(Refiner_flow, self).__init__()
        self.argv = argv
        self.upsampling = upsampling
        self.ch_feature = ch_feature
        if upsampling:
            self.netFeat = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=ch_feature, out_channels=ch_feature*2, kernel_size=1, stride=1, padding=0),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                        )
            ch_num = 4 * ch_feature + 2
        else:
            self.netFeat = torch.nn.Sequential()
            ch_num = 2 * ch_feature + 2

        self.warping = Warping_layer(self.argv)
        self.phaseC = phaseC_layer(self.argv)
        
        self.NetMain_refiner = torch.nn.Sequential(
            conv(argv.batch_norm, ch_num, 128), conv(argv.batch_norm, 128, 128, 3),
            conv(argv.batch_norm, 128, 96, 3), conv(argv.batch_norm, 96, 64, 3),
            conv(argv.batch_norm, 64, 32, 3),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=2,
                            kernel_size=1,
                            stride=1,
                            padding=0)             
                            )

    def forward(self, imgFeature_first, imgFeature_second, flow):
        if self.upsampling:
            # upsample to double channel's number
            imgFeature_first = self.netFeat(imgFeature_first)
            imgFeature_second = self.netFeat(imgFeature_second)
        
        with torch.cuda.amp.autocast(enabled=False):
            imgFeature_first = self.warping(imgFeature_first.float(), flow.float()) / self.phaseC(flow.float())
        return flow + self.NetMain_refiner(torch.cat([imgFeature_first, imgFeature_second, flow], 1))

