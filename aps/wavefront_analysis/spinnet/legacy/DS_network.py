#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    	: 08 / 12 / 2022
@Author  	: Zhi Qiao
@Contact	: z.qiao1989@gmail.com
@File    	: DS_network.py
@Software	: SPINNet
@Desc		: network structure of SPINNet
            model of deep speckle. including the follow parts:
            1. feature extractor
            2. flow estimator
            3. refiner for flow
'''
import os
import torch
from aps.wavefront_analysis.spinnet.legacy.module import FeatureExtractor, costvol_layer, Warping_layer, FlowEstimator, Refiner_flow, phaseC_layer

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class Network(torch.nn.Module):
    def __init__(self, argv):
        super(Network, self).__init__()
        self.argv = argv

        self.netFeatures = FeatureExtractor(self.argv)

        self.warping = Warping_layer(self.argv)
        self.phaseC = phaseC_layer(self.argv)

        if argv.corr == 'costvol_layer':
            self.corr = costvol_layer(argv)

        self.flow_estimators = torch.nn.ModuleList()
        # flow estimator, the input is pyramid feature + cost volume + previous flow
        # so the input channel here is: ch + (argv.search_range*2+1)**2 + 2
        for l, ch in enumerate(argv.lv_chs[::-1]):
            layer = FlowEstimator(argv, (argv.search_range*2+1)**2)
            self.flow_estimators.append(layer)

        # refiner for flow, if pyramid level is lower than 2, upsample the pyramid channels to subpixel resolution
        self.flow_refiner = torch.nn.ModuleList()
        # refiner for level 4, instead of level 5
        layer = Refiner_flow(argv, argv.lv_chs[1], False)
        self.flow_refiner.append(layer)

        # init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m.bias is not None: torch.nn.init.uniform_(m.bias)
                torch.nn.init.xavier_uniform_(m.weight)

            if isinstance(m, torch.nn.ConvTranspose2d):
                if m.bias is not None: torch.nn.init.uniform_(m.bias)
                torch.nn.init.xavier_uniform_(m.weight)
        

    def forward(self, imgs):
        # the input data should be [batch, 2, ch, Height, Width], [:, 0, :, :] is ref, [:, 0, :, :] is img
        img_first = imgs[:, 0:1, :, :, ]
        img_second = imgs[:, 1:2, :, :]

        # obtain pyramid features (6, 5, 4, 3, 2, 1) + original image
        imgFeature_first = self.netFeatures(img_first) + [img_first]
        imgFeature_second = self.netFeatures(img_second) + [img_second]

        # get down sampling pure images
        tenFirst = [img_first]
        tenSecond = [img_second]
        for intLevel in range(self.argv.num_levels-1):
            tenFirst.append(torch.nn.functional.interpolate(input=tenFirst[-1], size=(imgFeature_first[-(intLevel+2)].shape[2], imgFeature_first[-(intLevel+2)].shape[3]), mode='bilinear', align_corners=True))
            tenSecond.append(torch.nn.functional.interpolate(input=tenSecond[-1], size=(imgFeature_second[-(intLevel+2)].shape[2], imgFeature_second[-(intLevel+2)].shape[3]), mode='bilinear', align_corners=True))
        tenFirst = tenFirst[::-1]
        tenSecond = tenSecond[::-1]

        # record flow and T from each pyramid level
        flow_list = []

        for lv, (f1, f2, img1, img2) in enumerate(zip(imgFeature_first[:self.argv.output_level+1], imgFeature_second[:self.argv.output_level+1], tenFirst[:self.argv.output_level+1], tenSecond[:self.argv.output_level+1])):

            # upsample flow and scale the displacement
            if lv == 0:

                shape = list(f1.size()); shape[1] = 2
                if self.argv.device == 'cuda':
                    flow = torch.zeros(shape).cuda()
                else:
                    flow = torch.zeros(shape)
            else:
                # scale factor of 2 for flow/displacement
                flow = torch.nn.functional.interpolate(flow, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2
                
            with torch.cuda.amp.autocast(enabled=False):
                    f1_warp = self.warping(f1.float(), flow.float())

            # correlation
            corr = self.corr(f2, f1_warp)

            if self.argv.corr_activation: 
                torch.nn.functional.leaky_relu_(corr)

            # concat and estimate flow
            # ATTENTION: `+ flow` makes flow estimator learn to estimate residual flow
            if self.argv.residual:
                flow_coarse = self.flow_estimators[lv](torch.cat([corr], dim = 1)) + flow
            else:
                flow_coarse = self.flow_estimators[lv](torch.cat([corr], dim = 1))
            
            if self.argv.with_refiner and lv == self.argv.output_level:
                flow = self.flow_refiner[0](f1, f2, flow)

            else:
                flow = flow_coarse

            # record each flow in the list
            flow_list.append(flow)

        flow = torch.nn.functional.interpolate(flow, scale_factor = 2 ** (self.argv.num_levels - self.argv.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.argv.num_levels - self.argv.output_level - 1)

        return flow

