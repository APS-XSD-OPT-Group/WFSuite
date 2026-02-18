#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    	: 08 / 19 / 2022
@Author  	: Zhi Qiao
@Contact	: z.qiao1989@gmail.com
@File    	: SPINNet_estimate.py
@Software	: AbsolutePhase
@Desc		: use SPINNet to get the phase based on the simulated reference and measured sample images.
'''

import torch
import math
import numpy as np
import os
import json
import h5py
import sys

from aps.wf_suite.spinnet.legacy.DS_network import Network, MODULE_PATH
# Insert the absolute path to the directory containing the DS_network module to the system path
sys.path.insert(0, MODULE_PATH)

##########################################################
def write_h5(result_path, file_name, data_dict):
    ''' this function is used to save the variables in *args to hdf5 file
        args are in format: {'name': data}
    '''
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with h5py.File(os.path.join(result_path, file_name+'.hdf5'), 'w') as f:
        for key_name in data_dict:
            f.create_dataset(key_name, data=data_dict[key_name], compression="gzip", compression_opts=9)
    print('result hdf5 file : {} saved'.format(file_name+'.hdf5'))

class args_setting:
    def __init__(self, dict_para):

        self.device = dict_para['device']
        self.OutputFolder = dict_para['OutputFolder']
        self.DataFolder = dict_para['DataFolder']
        self.TestFolder = dict_para['TestFolder']
        self.num_workers = dict_para['num_workers']
        self.batch_size = dict_para['batch_size']
        self.epoch = dict_para['epoch']
        self.lr = dict_para['lr']
        self.lv_chs = dict_para['lv_chs']
        self.output_level = dict_para['output_level']
        self.batch_norm = dict_para['batch_norm']
        self.corr = dict_para['corr']
        self.corr_activation = dict_para['corr_activation']
        self.search_range = dict_para['search_range']
        self.residual = dict_para['residual']
        self.fp16 = dict_para['fp16']
        self.with_refiner = dict_para['with_refiner']
        self.load_all_data = dict_para['load_all_data']
        self.load_check_point = dict_para['load_check_point']
        self.device_type = torch.device(self.device)
        self.num_levels = len(self.lv_chs)


def estimate(img_stack, model, args):
    """
    inference function of SPINNet using the trained model

    Args:
        img_stack (ndarray): input ref and sample image stack
        model (torch.model): loaded trained torch model
        args (dict): parameters of the trained model

    Returns:
        _type_: _description_
    """
    intWidth = img_stack[0].shape[-1]
    intHeight = img_stack[0].shape[-2]

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    img_stack = np.array(img_stack)
    img_stack = torch.FloatTensor(
        np.ascontiguousarray(
            img_stack[:, :, :].astype(np.float32)))

    img_stack = torch.nn.functional.interpolate(
                            input=img_stack,
                            size=(intPreprocessedHeight, intPreprocessedWidth),
                            mode='bilinear',
                            align_corners=True)


    with torch.no_grad():
        # Transfer to GPU
        img_stack = img_stack.to(args.device_type)
        flow_predict = model(img_stack)
        
        tenFlow = torch.nn.functional.interpolate(
                        input=flow_predict,
                        size=(intHeight, intWidth),
                        mode='bilinear',
                        align_corners=True)
        
        tenFlow = tenFlow.cpu().detach().numpy()
        
        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow[0, :, :, :]


def SPINNet_estimate(ref, img, model_path, setting_path, device):
    """
    SPINNet_estimate to predict the phase according to ref and img

    Args:
        ref (ndarray): ref image
        img (ndarray): sample image
        model_path (str): saved model path
        setting_path (str): saved setting path of the trained model
        device (srt): use cuda or cpu
    """

    # load the setting from the trained model
    with open(setting_path) as f:
        setting_dict = json.load(f)

    setting_dict['device'] = device

    args = args_setting(setting_dict)

    # print(args.__dict__)
    
    torch.manual_seed(42)
    model = Network(args).to(args.device_type)
    checkpoint = torch.load(model_path, map_location=args.device_type)
    print('load checkpoint from file: {}'.format(model_path))
    
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # load data
    
    f_max = np.amax([ref, img])
    img = img / f_max
    ref = ref / f_max

    img_stack = []
    img_stack.append(np.array([ref, img]))

    tenFlow = estimate(img_stack, model, args)
    # print(tenFlow.shape)
    displace_x = tenFlow[0, :, :] - np.mean(tenFlow[0, 0:50, 0:50])
    displace_y = tenFlow[1, :, ] - np.mean(tenFlow[1, 0:50, 0:50])

    return displace_y, displace_x
        

        
    
