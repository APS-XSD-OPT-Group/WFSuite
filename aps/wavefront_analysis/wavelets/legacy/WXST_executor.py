#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2025, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2025. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# ----------------------------------------------------------------------- #

################################################################
#
# REPLACE MAIN IN ZHI QIAO's LEGACY CODE
#
################################################################
import os
import numpy as np
import scipy.constants as sc
import scipy.ndimage as snd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib

from aps.wavefront_analysis.common.arguments import Args
from aps.wavefront_analysis.common.legacy.func import prColor
from aps.wavefront_analysis.wavelets.legacy.func import load_image, auto_crop, image_align
from aps.wavefront_analysis.wavelets.legacy.WXST import WXST
from aps.common.driver.beamline.generic_camera import get_image_data

class WXSTResult:
    def __init__(self, displace, DPC_x, DPC_y, phase):
        self.__displace = displace
        self.__DPC_x = DPC_x
        self.__DPC_y = DPC_y
        self.__phase = phase

    @property
    def displace(self): return self.__displace
    @displace.setter
    def displace(self, value): self.__displace = value

    @property
    def DPC_y(self): return self.__DPC_y
    @DPC_y.setter
    def DPC_y(self, value): self.__DPC_y = value

    @property
    def DPC_x(self): return self.__DPC_x
    @DPC_x.setter
    def DPC_x(self, value): self.__DPC_x = value

    @property
    def phase(self): return self.__phase
    @phase.setter
    def phase(self, value): self.__phase = value

    def to_dict(self):
        return {
            'displace': self.__displace,
            'DPC_x': self.__DPC_x,
            'DPC_y': self.__DPC_y,
            'phase': self.__phase
        }


def execute_process_image(**arguments):
    arguments["img"]              = arguments.get("img", "../testdata/single-shot/sample_001.tif") # path to sample image
    arguments["ref"]              = arguments.get("ref", "../testdata/single-shot/ref_001.tif") # path to reference image'
    arguments["dark"]             = arguments.get("dark", None) # dark image for image correction
    arguments["flat"]             = arguments.get("flat", None) # flat image for image correction
    arguments["result_folder"]    = arguments.get("result_folder", "../testdata/single-shot/WXST_results") # saving folder
    arguments["crop"]             = arguments.get("crop", [450, 1000, 500, 1000]) # image crop, if is [256], central crop. if len()==4, boundary crop, if is 0, use gui crop, if is -1, use auto-crop
    arguments["p_x"]              = arguments.get("p_x", 0.65e-6) # pixel size
    arguments["scaling_x"]        = arguments.get("scaling_x", 1.0) # x pixel scaling from detector to sample
    arguments["scaling_y"]        = arguments.get("scaling_y", 1.0) # y pixel scaling from detector to sample
    arguments["energy"]           = arguments.get("energy", 14e3) # X-ray energy
    arguments["distance"]         = arguments.get("distance", 500e-3) # detector to mask distance
    arguments["down_sampling"]    = arguments.get("down_sampling", 1) # down-sample images to reduce memory cost and accelerate speed.
    arguments["GPU"]              = arguments.get("GPU", False) # Use GPU or not. GPU can be 2 times faster. But multi-resolution process is disabled.
    arguments["use_wavelet"]      = arguments.get("use_wavelet", False) # use wavelet transform or not.
    arguments["wavelet_lv_cut"]   = arguments.get("wavelet_lv_cut", 2) # wavelet cutting level
    arguments["pyramid_level"]    = arguments.get("pyramid_level", 1) # pyramid level used for speckle tracking.
    arguments["n_iter"]           = arguments.get("n_iter", 1) # number of iteration for speckle tracking. 1 is good.
    arguments["template_size"]    = arguments.get("template_size", 5) # template size in the WXST
    arguments["cal_half_window"]  = arguments.get("cal_half_window", 10) # searching window of speckle tracking. Means the largest displacement can be calculated.
    arguments["nCores"]           = arguments.get("nCores", 4) # number of CPU cores used for calculation.
    arguments["nGroup"]           = arguments.get("nGroup", 1) # number of groups that parallel calculation is splitted into.

    arguments["verbose"]          = arguments.get("verbose", True)
    arguments["save_images"]      = arguments.get("save_images", True)

    args = Args(arguments)

    for key, value in args.__dict__.items():
        prColor('{}: {}'.format(key, value), 'cyan')

    import cv2 # import here prevents conflict with system openCV

    File_ref          = args.ref
    File_img          = args.img
    flat              = args.flat
    dark              = args.dark
    folder_result     = args.result_folder
    N_s               = args.template_size
    cal_half_window   = args.cal_half_window
    N_s_extend        = 4 # the calculation window for high order pyramid
    n_cores           = args.nCores
    n_group           = args.nGroup
    energy            = args.energy
    p_x               = args.p_x
    scaling_x         = args.scaling_x
    scaling_y         = args.scaling_y
    z                 = args.distance
    pyramid_level     = args.pyramid_level
    n_iter            = args.n_iter
    use_GPU           = args.GPU
    down_sample       = args.down_sampling
    use_wavelet       = args.use_wavelet
    wavelet_level_cut = args.wavelet_lv_cut


    def _load_image(file_img):
        extension = os.path.splitext(file_img.lower())[1]
        if   extension == ".tif":  return load_image(file_img)
        elif extension == ".hdf5":
            image, _, _ = get_image_data(file_img)
            return image

    ref_data = _load_image(File_ref)
    img_data = _load_image(File_img)

    ref_data = ref_data.astype(np.float32)
    img_data = img_data.astype(np.float32)
    if args.dark == 'None': dark = np.zeros_like(img_data, dtype=np.float32)
    else:                   dark = _load_image(dark).astype(np.float32)

    if args.flat == 'None': flat = np.ones_like(img_data, dtype=np.float32)
    else:                   flat = _load_image(flat).astype(np.float32)

    zero_mask = (flat - dark) != 0
    img_data[zero_mask] = (img_data[zero_mask] - dark[zero_mask]) / (flat[zero_mask] - dark[zero_mask])
    ref_data[zero_mask] = (ref_data[zero_mask] - dark[zero_mask]) / (flat[zero_mask] - dark[zero_mask])
    # do image alignment
    if True:
        pos_shift, ref_data = image_align(img_data, ref_data)
        max_shift = int(np.amax(np.abs(pos_shift)) + 1)
        crop_area = lambda img: img[max_shift:-max_shift, max_shift:-max_shift]
        img_data = crop_area(img_data)
        ref_data = crop_area(ref_data)

    if len(args.crop) == 4:
        pass
    elif len(args.crop) == 1:
        if args.crop[0] == -1:
            prColor('auto crop------------------------------------------------', 'green')
            # use auto-crop according to the DPC_x boundary. rectangular shapess
            pattern_size = 5e-6  # assume 5um mask pattern
            flat = snd.uniform_filter(img_data, size=10 * (pattern_size / p_x))
            args.crop = auto_crop(flat, shrink=0.85)
        else:
            # Central crop with the provided width
            crop_width = args.crop[0]
            # Calculate image center
            center_y = img_data.shape[0] // 2
            center_x = img_data.shape[1] // 2
            # Calculate half-width of the crop
            half_width = crop_width // 2
            # Calculate four corners for the cropping region
            y0 = max(0, center_y - half_width)
            y1 = min(img_data.shape[0], center_y + half_width)
            x0 = max(0, center_x - half_width)
            x1 = min(img_data.shape[1], center_x + half_width)
            # Set args.crop to the calculated boundary
            args.crop = [y0, y1, x0, x1]
    else:
        raise Exception('Error: wrong crop option. -1 for autocrop, [256] for central crop; [y0, y1, x0, x1] for boundary crop')
    
    print(args.crop)

    boundary_crop = lambda img: img[int(args.crop[0]):int(args.crop[1]), int(args.crop[2]):int(args.crop[3])]
    
    ref_data = boundary_crop(ref_data)
    img_data = boundary_crop(img_data)

    M_image = ref_data.shape[0]

    size_origin = ref_data.shape

    ref_data = ref_data.astype(np.float32)
    img_data = img_data.astype(np.float32)

    # down-sample or not
    if down_sample != 1:
        prColor('down-sample image: {}'.format(down_sample), 'cyan')
        d_size = (int(ref_data.shape[1] * down_sample), int(ref_data.shape[0] * down_sample))

        img_data = cv2.resize(img_data, d_size)
        ref_data = cv2.resize(ref_data, d_size)

    WXST_solver = WXST(img_data,
                       ref_data,
                       M_image=M_image,
                       N_s=N_s,
                       cal_half_window=cal_half_window,
                       N_s_extend=N_s_extend,
                       n_cores=n_cores,
                       n_group=n_group,
                       energy=energy,
                       p_x=p_x,
                       z=z,
                       wavelet_level_cut=wavelet_level_cut,
                       pyramid_level=pyramid_level,
                       n_iter=n_iter,
                       use_wavelet=use_wavelet,
                       use_GPU=use_GPU,
                       scaling_x=scaling_x,
                       scaling_y=scaling_y)

    if not os.path.exists(folder_result): os.makedirs(folder_result)
    
    sample_transmission = img_data / (np.abs(ref_data) + 1)
    
    if args.save_images: plt.imsave(os.path.join(folder_result, 'transmission.png'), sample_transmission)

    WXST_solver.run(result_path=folder_result)

    displace = WXST_solver.displace
    DPC_x    = WXST_solver.DPC[1]
    DPC_y    = WXST_solver.DPC[0]
    phase    = WXST_solver.phase

    # down-sample or not
    if down_sample != 1:
        prColor('scale back', 'green')
        displace_x = cv2.resize(displace[1], (size_origin[1], size_origin[0])) * (1 / down_sample)
        displace_y = cv2.resize(displace[0], (size_origin[1], size_origin[0])) * (1 / down_sample)
        displace = [displace_y, displace_x]
        DPC_x = cv2.resize(DPC_x, (size_origin[1], size_origin[0])) * (1 / down_sample)
        DPC_y = cv2.resize(DPC_y, (size_origin[1], size_origin[0])) * (1 / down_sample)
        phase = cv2.resize(phase, (size_origin[1], size_origin[0])) * (1 / down_sample)

    result = WXSTResult(displace=displace,
                        DPC_x=DPC_x,
                        DPC_y=DPC_y,
                        phase=phase)

    if args.save_images: 
        plt.imsave(os.path.join(folder_result, 'displace_x.png'), displace[1])
        plt.imsave(os.path.join(folder_result, 'displace_y.png'), displace[0])
        plt.imsave(os.path.join(folder_result, 'dpc_x.png'), DPC_x)
        plt.imsave(os.path.join(folder_result, 'dpc_y.png'), DPC_y)
        plt.imsave(os.path.join(folder_result, 'phase.png'), phase)
    
        matplotlib.use('Agg')  # Non-GUI backend
        plt.figure()
        plt.imshow(displace[0])
        cbar = plt.colorbar()
        cbar.set_label('[pixels]', rotation=90)
        plt.savefig(os.path.join(folder_result, 'displace_y_colorbar.png'), dpi=150)
        plt.close()  # Close the figure after saving
    
        plt.figure()
        plt.imshow(displace[1])
        cbar = plt.colorbar()
        cbar.set_label('[pixels]', rotation=90)
        plt.savefig(os.path.join(folder_result, 'displace_x_colorbar.png'), dpi=150)
        plt.close()  # Close the figure after saving
    
        plt.figure()
        plt.imshow(phase)
        cbar = plt.colorbar()
        cbar.set_label('[rad]', rotation=90)
        plt.savefig(os.path.join(folder_result, 'phase_colorbar.png'), dpi=150)
        plt.close()  # Close the figure after saving
    
        # plt.show()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        XX, YY = np.meshgrid(
            np.arange(phase.shape[1]) * p_x * scaling_x * 1e6,
            np.arange(phase.shape[0]) * p_x * scaling_y * 1e6)
        ax1.plot_surface(XX, YY, phase, cmap=cm.get_cmap('hot'))
        ax1.set_xlabel('x [μm]')
        ax1.set_ylabel('y [μm]')
        ax1.set_zlabel('phase [rad]')
        plt.savefig(os.path.join(folder_result, 'Phase_3d.png'), dpi=150)
        plt.close()

    return result.to_dict()
