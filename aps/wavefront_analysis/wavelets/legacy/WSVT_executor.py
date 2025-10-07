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

import os
import numpy as np
import scipy.constants as sc
import scipy.ndimage as snd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib

from aps.wavefront_analysis.common.arguments import Args
from aps.wavefront_analysis.common.legacy.func import prColor
from aps.wavefront_analysis.wavelets.legacy.func import load_images, auto_crop
from aps.wavefront_analysis.wavelets.legacy.WSVT import WSVT

class WSVTResult:
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

def execute_process_images(**arguments):
    arguments["crop"]            = arguments.get("crop", [450, 1000, 500, 1000]) # image crop, if is [256], central crop. if len()==4, boundary crop, if is 0, use gui crop, if is -1, use auto-crop
    arguments["folder_img"]      = arguments.get("folder_img") # Path to image folder
    arguments["folder_ref"]      = arguments.get("folder_ref") # Path to reference folder
    arguments["folder_result"]   = arguments.get("folder_result") # Path to result folder
    arguments["cal_half_window"] = arguments.get("cal_half_window", 20) # Number of pixels for each calculation area (half window size)
    arguments["n_cores"]         = arguments.get("n_cores", 4) # Number of cores for parallel processing
    arguments["n_group"]         = arguments.get("n_group", 1) # Number to reduce memory usage per group
    arguments["energy"]          = arguments.get("energy", 8.9e3) # Beam energy in eV"
    arguments["pixel_size"]      = arguments.get("p_x", 0.65e-6) # Pixel size in meters
    arguments["distance"]        = arguments.get("distance", 300e-3) # Distance in meters
    arguments["use_wavelet"]     = arguments.get("use_wavelet", False) # Whether to use wavelet transform (0 or 1)
    arguments["wavelet_ct"]      = arguments.get("wavelet_ct", 2) # Wavelet level cut
    arguments["pyramid_level"]   = arguments.get("pyramid_level", 1) # Pyramid level to wrap images
    arguments["n_iteration"]     = arguments.get("n_iteration", 1) # Number of iterations for calculation
    arguments["n_scan"]          = arguments.get("n_scan", 1) # Number of scans for calculation
    arguments["use_GPU"]         = arguments.get("use_GPU", False) # Whether to use GPU (0 or 1)
    arguments["scaling_x"]       = arguments.get("scaling_x", 1.0) # x pixel scaling from detector to sample
    arguments["scaling_y"]       = arguments.get("scaling_y", 1.0) # y pixel scaling from detector to sample

    arguments["verbose"]         = arguments.get("verbose", True)
    arguments["save_images"]     = arguments.get("save_images", True)

    args = Args(arguments)

    for key, value in args.__dict__.items(): prColor('{}: {}'.format(key, value), 'cyan')

    # Assign arguments to variables
    folder_img        = args.folder_img
    folder_ref        = args.folder_ref
    folder_result     = args.folder_result
    cal_half_window   = args.cal_half_window # the number of the area to calculate for each pixel, 2*cal_half_window X 2*cal_half_window
    n_s_extend        = 4 # the calculation window for high order pyramid (still hardcoded)
    n_cores           = args.n_cores
    n_group           = args.n_group
    energy            = args.energy
    p_x               = args.pixel_size
    z                 = args.distance
    use_wavelet       = args.use_wavelet
    wavelet_level_cut = args.wavelet_ct
    pyramid_level     = args.pyramid_level
    n_iter            = args.n_iteration
    n_scan            = args.n_scan
    use_GPU           = args.use_GPU
    scaling_x         = args.scaling_x
    scaling_y         = args.scaling_y
    use_estimate      = False

    ref_data = load_images(folder_ref, '*.tif')
    img_data = load_images(folder_img, '*.tif')

    # crop image to roi
    if len(args.crop) == 4:
        # boundary crop, use the corner index [y0, y1, x0, x1]
        # boundary_crop = lambda img: img[int(args.crop[0]):int(args.crop[1]),
        #                                 int(args.crop[2]):int(args.crop[3])]
        pass
    elif len(args.crop) == 1:
        if args.crop[0] == -1:
            prColor('auto crop------------------------------------------------', 'green')
            # use auto-crop according to the intensity boundary. rectangular shapess
            pattern_size = 5e-6  # assume 5um mask pattern
            flat = snd.uniform_filter(img_data[0], size=10 * (pattern_size / p_x))
            args.crop = auto_crop(flat, shrink=0.85)
        else:
            # Central crop with the provided width
            crop_width = args.crop[0]
            # Calculate image center
            center_y = img_data.shape[1] // 2
            center_x = img_data.shape[2] // 2
            # Calculate half-width of the crop
            half_width = crop_width // 2
            # Calculate four corners for the cropping region
            y0 = max(0, center_y - half_width)
            y1 = min(img_data.shape[1], center_y + half_width)
            x0 = max(0, center_x - half_width)
            x1 = min(img_data.shape[2], center_x + half_width)
            # Set args.crop to the calculated boundary
            args.crop = [y0, y1, x0, x1]
    else:
        raise Exception('Error: wrong crop option. -1 for autocrop, [256] for central crop; [y0, y1, x0, x1] for boundary crop')

    print(args.crop)

    boundary_crop = lambda img: img[int(args.crop[0]):int(args.crop[1]),
                                int(args.crop[2]):int(args.crop[3])]
    # Apply cropping to all images in ref_data and img_data
    ref_data_cropped = np.array([boundary_crop(img) for img in ref_data])
    img_data_cropped = np.array([boundary_crop(img) for img in img_data])

    # Update ref_data and img_data with cropped data
    ref_data = ref_data_cropped
    img_data = img_data_cropped
    M_image = ref_data.shape[1]

    ref_data = ref_data.astype(np.float32)
    img_data = img_data.astype(np.float32)

    if n_scan <= ref_data.shape[0]:
        ref_data = ref_data[0:n_scan, :, :]
        img_data = img_data[0:n_scan, :, :]

    print("use {} scan position for calculation".format(ref_data.shape[0]))

    WSVT_solver = WSVT(img_data,
                       ref_data,
                       M_image=M_image,
                       N_s_extend=n_s_extend,
                       cal_half_window=cal_half_window,
                       n_cores=n_cores,
                       n_group=n_group,
                       energy=energy,
                       p_x=p_x,
                       z=z,
                       wavelet_level_cut=wavelet_level_cut,
                       pyramid_level=pyramid_level,
                       n_iter=n_iter,
                       use_estimate=use_estimate,
                       use_wavelet=use_wavelet,
                       use_GPU=use_GPU,
                       scaling_x=scaling_x,
                       scaling_y=scaling_y,
                       crop=args.crop)

    if not os.path.exists(folder_result): os.makedirs(folder_result)

    sample_transmission = img_data[0] / ref_data[0]

    if args.save_images: plt.imsave(os.path.join(folder_result, 'transmission.png'), sample_transmission)

    WSVT_solver.run(result_path=folder_result)

    displace = WSVT_solver.displace
    DPC_x    = WSVT_solver.DPC[1]
    DPC_y    = WSVT_solver.DPC[0]
    phase    = WSVT_solver.phase

    result = WSVTResult(displace=displace,
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