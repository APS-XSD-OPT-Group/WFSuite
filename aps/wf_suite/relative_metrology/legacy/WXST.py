#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    	: 08 / 08 / 2022
@Author  	: Zhi Qiao
@Contact	: z.qiao1989@gmail.com
@File    	: WSVT.py
@Software	: WaveletSBI
@Desc		: single-shot mode wavelet speckle tracking (WaveletSBI) script

1. Qiao, Zhi, Xianbo Shi, and Lahsen Assoufid. “Single-Shot Speckle Tracking Method Based on Wavelet Transform and Multi-Resolution Analysis.” In Advances in Metrology for X-Ray and EUV Optics IX, edited by Lahsen Assoufid, Haruhiko Ohashi, and Anand Asundi, 22. Online Only, United States: SPIE, 2020. https://doi.org/10.1117/12.2569135.
2. Qiao, Zhi, Xianbo Shi, Rafael Celestre, and Lahsen Assoufid. “Wavelet-Transform-Based Speckle Vector Tracking Method for X-Ray Phase Imaging.” Optics Express 28, no. 22 (October 26, 2020): 33053. https://doi.org/10.1364/OE.404606.

'''

import numpy as np
import pywt
import os
import time

import torch
import scipy.constants as sc
import multiprocessing as ms
import concurrent.futures
import scipy.interpolate as sfit

from aps.wf_suite.relative_metrology.legacy.func import prColor, wavelet_transform_multiprocess, write_h5, write_json, find_disp, frankotchellappa, cost_volume, slope_tracking
from aps.wf_suite.common.legacy.euclidean_dist import dist_numba

class WXST:
    def __init__(self,
                 img,
                 ref,
                 M_image=512,
                 N_s=5,
                 cal_half_window=20,
                 N_s_extend=4,
                 n_cores=4,
                 n_group=4,
                 energy=14e3,
                 p_x=0.65e-6,
                 z=500e-3,
                 wavelet_level_cut=2,
                 pyramid_level=2,
                 n_iter=1,
                 use_estimate=False,
                 use_wavelet=True,
                 use_GPU=0,
                 scaling_x=1.0,
                 scaling_y=1.0):
        self.img_data = img
        self.ref_data = ref
        # roi of the images
        self.M_image = M_image
        # template window, the N_s nearby pixels used to represent the local pixel, 2*N_s+1
        self.N_s = N_s
        # the number of the area to calculate for each pixel, 2*cal_half_window X 2*cal_half_window
        self.cal_half_window = cal_half_window
        # the calculation window for high order pyramid
        self.N_s_extend = N_s_extend

        # process number for parallel
        self.n_cores = n_cores
        # number to reduce the each memory use
        self.n_group = n_group

        # energy, 10kev
        self.energy = energy
        self.wavelength = sc.value(
            'inverse meter-electron volt relationship') / energy
        # pixel size [m]
        self.p_x = p_x
        # scaling factor before phase integration
        self.scaling_x = scaling_x
        self.scaling_y = scaling_y
        # distance [m]
        self.z = z
        self.wavelet_level_cut = wavelet_level_cut
        # pyramid level to wrap the images
        self.pyramid_level = pyramid_level
        # iterations for the calculation
        self.n_iter = n_iter
        # if use the estimated displace as initial guess
        self.use_estimate = use_estimate
        # if use wavelet transform or not
        self.use_wavelet = use_wavelet

        # use GPU or not
        if torch.cuda.is_available() and use_GPU == 1:
            prColor('Use GPU found. Disable multi-resolution', 'cyan')
            self.use_GPU = True
            # for GPU, there's no multi-resolution process
            self.pyramid_level = 0
        else:
            prColor('No gpu found. Use CPU instead.', 'cyan')
            self.use_GPU = False

        if self.use_estimate:
            # get initial estimation from the cv2 flow tracking
            displace_estimate = slope_tracking(self.ref_data,
                                               self.img_data,
                                               n_window=self.cal_half_window)
            self.displace_estimate = [
                displace_estimate[0], displace_estimate[1]
            ]
        else:
            m, n = self.img_data.shape
            self.displace_estimate = [np.zeros((m, n)), np.zeros((m, n))]

    def template_stack(self, img):
        '''
            stack the nearby pixels in 2*N_s+1
        '''
        img_stack = []
        axis_Nw = np.arange(-self.N_s, self.N_s + 1)
        for x in axis_Nw:
            for y in axis_Nw:
                img_stack.append(np.roll(np.roll(img, x, axis=0), y, axis=1))

        return np.array(img_stack)

    def pyramid_data(self):
        """
        generate pyramid data for multi-resolution

        """
        ref_pyramid = []
        img_pyramid = []
        prColor(
            'obtain pyramid image and stack the window with pyramid level: {}'.
            format(self.pyramid_level), 'green')
        ref_pyramid.append(self.ref_data)
        img_pyramid.append(self.img_data)

        for kk in range(self.pyramid_level):
            ref_pyramid.append(
                pywt.dwtn(ref_pyramid[kk], 'db3', mode='zero',
                          axes=(-2, -1))['aa'])
            img_pyramid.append(
                pywt.dwtn(img_pyramid[kk], 'db3', mode='zero',
                          axes=(-2, -1))['aa'])

        normlize_std = lambda img: (
            (img - np.ndarray.mean(img, axis=0)) / np.ndarray.std(img, axis=0))

        ref_pyramid = [
            normlize_std(self.template_stack(img_data))
            for img_data in ref_pyramid
        ]
        img_pyramid = [
            normlize_std(self.template_stack(img_data))
            for img_data in img_pyramid
        ]

        return ref_pyramid, img_pyramid

    def resampling_spline(self, img, s):
        # img: original
        # s: size of the sampling, (row, col)
        m, n = img.shape
        x_axis = np.arange(n)
        y_axis = np.arange(m)
        fit = sfit.RectBivariateSpline(y_axis, x_axis, img)

        x_new = np.linspace(0, n - 1, s[1])
        y_new = np.linspace(0, m - 1, s[0])

        return fit(y_new, x_new)

    def wavelet_data(self):
        """
        generate wavelet data from the image data

        """
        # process the data to get the wavelet transform
        ref_pyramid, img_pyramid = self.pyramid_data()
        if self.use_wavelet:
            prColor('obtain wavelet data...', 'green')
            wavelet_method = 'db2'
            # wavelet_method = 'bior1.3'
            # wavelet wrapping level. 2 is half, 3 is 1/3 of the size
            max_wavelet_level = pywt.dwt_max_level(ref_pyramid[0].shape[0],
                                                   wavelet_method)
            prColor('max wavelet level: {}'.format(max_wavelet_level), 'green')
            self.wavelet_level = max_wavelet_level
            coefs_level = self.wavelet_level + 1 - self.wavelet_level_cut

            if ref_pyramid[0].shape[0] > 150:
                self.wavelet_add_list = [0, 0, 0, 0, 0, 0]
            elif ref_pyramid[0].shape[0] > 50:
                self.wavelet_add_list = [0, 0, 1, 2, 2, 2]
            else:
                self.wavelet_add_list = [2, 2, 2, 2, 2, 2]

            # wavelet transform and cut for the pyramid images
            start_time = time.time()
            for p_level in range(len(img_pyramid)):
                if p_level > len(self.wavelet_add_list):
                    wavelevel_add = 2
                else:
                    wavelevel_add = self.wavelet_add_list[p_level]

                img_wa, level_name = wavelet_transform_multiprocess(
                    img_pyramid[p_level],
                    8,
                    wavelet_method,
                    w_level=self.wavelet_level,
                    return_level=coefs_level + wavelevel_add)

                img_pyramid[p_level] = img_wa

                ref_wa, level_name = wavelet_transform_multiprocess(
                    ref_pyramid[p_level],
                    8,
                    wavelet_method,
                    w_level=self.wavelet_level,
                    return_level=coefs_level + wavelevel_add)

                ref_pyramid[p_level] = ref_wa

                prColor(
                    'pyramid level: {}\nvector length: {}\nUse wavelet coef: {}'
                    .format(p_level, ref_wa.shape[2], level_name), 'green')

            end_time = time.time()
            print('wavelet time: {}'.format(end_time - start_time))
        else:
            img_pyramid = [
                np.moveaxis(img_data, 0, -1) for img_data in img_pyramid
            ]
            ref_pyramid = [
                np.moveaxis(img_data, 0, -1) for img_data in ref_pyramid
            ]
            self.wavelet_level = None
            self.wavelet_add_list = None
            self.wavelet_level_cut = None

        return ref_pyramid, img_pyramid

    def displace_wavelet(self, y_list, img_wa_stack, ref_wa_stack,
                         displace_pyramid, cal_half_window, n_pad):
        """
        calculate displacement using CPU

        Args:
            y_list (list): y axis of data
            img_wa_stack (ndarray): sample wavelet data
            ref_wa_stack (ndarray): reference wavelet data
            displace_pyramid (ndarray): estimated displacement for multi-resolution
            cal_half_window (int): half size of the searching window
            n_pad (int): padding width

        """
        dim = img_wa_stack.shape
        disp_x = np.zeros((dim[0], dim[1]))
        disp_y = np.zeros((dim[0], dim[1]))

        # the axis for the peak position finding
        window_size = 2 * cal_half_window + 1
        y_axis = np.arange(window_size) - cal_half_window
        x_axis = np.arange(window_size) - cal_half_window
        XX, YY = np.meshgrid(x_axis, y_axis)

        for yy in range(dim[0]):
            for xx in range(dim[1]):
                img_wa_line = img_wa_stack[yy, xx, :]
                ref_wa_data = ref_wa_stack[
                    n_pad + yy + int(displace_pyramid[0][yy, xx]):n_pad + yy +
                    int(displace_pyramid[0][yy, xx]) + window_size,
                    n_pad + xx + int(displace_pyramid[1][yy, xx]):n_pad + xx +
                    int(displace_pyramid[1][yy, xx]) + window_size, :]

                Corr_img = dist_numba(img_wa_line, ref_wa_data)

                disp_y[yy,
                       xx], disp_x[yy,
                                   xx], _, _ = find_disp(Corr_img,
                                                         XX,
                                                         YY,
                                                         sub_resolution=True)

        disp_add_y = displace_pyramid[0] + disp_y
        disp_add_x = displace_pyramid[1] + disp_x
        return disp_add_y, disp_add_x, y_list

    def displace_torch(self, img_wa_stack, ref_wa_stack, cal_half_window):
        """
        calculate displacement using GPU

        Args:
            img_wa_stack (ndarray): sample wavelet data
            ref_wa_stack (ndarray): reference wavelet data
            displace_pyramid (ndarray): estimated displacement for multi-resolution
            cal_half_window (int): half size of the searching window
        """
        dim = img_wa_stack.shape
        disp_x = np.zeros((dim[0], dim[1]))
        disp_y = np.zeros((dim[0], dim[1]))

        start_time = time.time()
        img_stack_cuda = torch.from_numpy(np.moveaxis(img_wa_stack, -1,
                                                      0)).cuda()
        ref_stack_cuda = torch.from_numpy(np.moveaxis(ref_wa_stack, -1,
                                                      0)).cuda()

        cost_vol = cost_volume(img_stack_cuda,
                               ref_stack_cuda,
                               search_range=cal_half_window)
        cost_vol = cost_vol.cpu().numpy()

        end_time = time.time()
        prColor('time cost for cost vol: {}'.format(end_time - start_time),
                'cyan')

        # the axis for the peak position finding
        window_size = 2 * cal_half_window + 1
        y_axis = np.arange(window_size) - cal_half_window
        x_axis = np.arange(window_size) - cal_half_window
        XX, YY = np.meshgrid(x_axis, y_axis)

        cores = ms.cpu_count()
        prColor('Computer available cores: {}'.format(cores), 'green')

        if cores > self.n_cores:
            cores = self.n_cores
        else:
            cores = ms.cpu_count()
        prColor('Use {} cores'.format(cores), 'light_purple')
        prColor('Process group number: {}'.format(self.n_group),
                'light_purple')

        if cores * self.n_group > self.M_image:
            n_tasks = 4
        else:
            n_tasks = cores * self.n_group

        y_axis = np.arange(dim[0])
        chunks_idx_y = np.array_split(y_axis, n_tasks)
        # use CPU parallel to calculate
        result_list = []
        '''
            find the peak position
        '''
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=cores) as executor:

            futures = []
            for y_list in chunks_idx_y:
                Corr_img = cost_vol[:, y_list, :]
                futures.append(
                    executor.submit(self.find_disp_parallel, Corr_img, XX, YY,
                                    y_list))

            for future in concurrent.futures.as_completed(futures):
                result_list.append(future.result())
                # display the status of the program
                Total_iter = cores * self.n_group
                Current_iter = len(result_list)
                percent_iter = Current_iter / Total_iter * 100
                str_bar = '>' * (int(np.ceil(percent_iter / 2))) + ' ' * (int(
                    (100 - percent_iter) // 2))
                prColor(
                    '\r' + str_bar + 'processing: [%3.1f%%] ' % (percent_iter),
                    'purple')

        disp_y_list = [item[0] for item in result_list]
        disp_x_list = [item[1] for item in result_list]
        y_list = [item[2] for item in result_list]

        for y, dx, dy in zip(y_list, disp_x_list, disp_y_list):
            disp_x[y, :] = dx
            disp_y[y, :] = dy

        return disp_y, disp_x

    def find_disp_parallel(self, Corr_img, XX, YY, y_list):
        """
        find the displacement using multi-processing

        Args:
            Corr_img (ndarray): correlation array
            XX (ndarray): x axis
            YY (ndarray): y axis
            y_list (list): y position for multi-processing
        """
        dim = Corr_img.shape
        window_size = int(np.sqrt(dim[0]))
        disp_y = np.zeros((dim[1], dim[2]))
        disp_x = np.zeros((dim[1], dim[2]))
        for yy in range(dim[1]):
            for xx in range(dim[2]):

                temp = Corr_img[:, yy, xx].reshape(window_size, window_size)
                disp_y[yy,
                       xx], disp_x[yy,
                                   xx], _, _ = find_disp(temp,
                                                         XX,
                                                         YY,
                                                         sub_resolution=True)

        return disp_y, disp_x, y_list

    def solver_cuda(self):
        """
        speckle tracking solver using GPU
        """
        ref_wavelet, img_wavelet = self.wavelet_data()

        transmission = self.img_data / (np.abs(self.ref_data + 1))
        for attr in ('img_data', 'ref_data'):
            self.__dict__.pop(attr, None)

        start_time = time.time()

        displace_y, displace_x = self.displace_torch(
            img_wavelet[0],
            ref_wavelet[0],
            cal_half_window=self.cal_half_window)

        end_time = time.time()
        prColor(
            '\r' + 'Processing time: {:0.3f} s'.format(end_time - start_time),
            'light_purple')

        displace = [
            np.fmax(np.fmin(displace_y, self.cal_half_window),
                    -self.cal_half_window),
            np.fmax(np.fmin(displace_x, self.cal_half_window),
                    -self.cal_half_window)
        ]
        prColor('displace map wrapping: {}'.format(displace[0].shape), 'green')
        print('max of displace: {}, min of displace: {}'.format(
            np.amax(displace[0]), np.amin(displace[1])))

        displace[0] = -displace[0][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]
        displace[1] = -displace[1][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]

        DPC_y = (displace[0] - np.mean(displace[0])) * self.p_x / self.z
        DPC_x = (displace[1] - np.mean(displace[1])) * self.p_x / self.z

        #Oct 2024, XSHI added pixel scaling
        phase = frankotchellappa(DPC_x, DPC_y, self.p_x*self.scaling_x, self.p_x*self.scaling_y) * 2 * np.pi / self.wavelength
        
        self.time_cost = end_time - start_time

        return displace, [DPC_y, DPC_x], phase, transmission

    def solver(self):
        """
        speckle tracking solver using CPU
        """
        ref_pyramid, img_pyramid = self.wavelet_data()

        transmission = self.img_data / (np.abs(self.ref_data + 1))
        for attr in ('img_data', 'ref_data'):
            self.__dict__.pop(attr, None)

        cores = ms.cpu_count()
        prColor('Computer available cores: {}'.format(cores), 'green')

        if cores > self.n_cores:
            cores = self.n_cores
        else:
            cores = ms.cpu_count()
        prColor('Use {} cores'.format(cores), 'light_purple')
        prColor('Process group number: {}'.format(self.n_group),
                'light_purple')

        if cores * self.n_group > self.M_image:
            n_tasks = 4
        else:
            n_tasks = cores * self.n_group

        start_time = time.time()
        # use pyramid wrapping
        max_pyramid_searching_window = int(
            np.ceil(self.cal_half_window / 2**self.pyramid_level))
        searching_window_pyramid_list = [self.N_s_extend
                                         ] * self.pyramid_level + [
                                             int(max_pyramid_searching_window)
                                         ]

        displace = self.displace_estimate

        for k_iter in range(self.n_iter):
            # iteration to approximating the results
            displace = [img / 2**self.pyramid_level for img in displace]

            m, n, c = img_pyramid[-1].shape
            displace[0] = self.resampling_spline(displace[0], (m, n))
            displace[1] = self.resampling_spline(displace[1], (m, n))

            prColor(
                'down sampling the dispalce to size: {}'.format(
                    displace[0].shape), 'green')

            displace = [
                np.fmax(
                    np.fmin(displace[0],
                            self.cal_half_window / 2**self.pyramid_level),
                    -self.cal_half_window / 2**self.pyramid_level),
                np.fmax(
                    np.fmin(displace[1],
                            self.cal_half_window / 2**self.pyramid_level),
                    -self.cal_half_window / 2**self.pyramid_level)
            ]

            for p_level in range(self.pyramid_level, -1, -1):
                # first pyramid, searching the window. Then search nearby
                if p_level == self.pyramid_level:
                    pyramid_seaching_window = searching_window_pyramid_list[
                        p_level]
                    m, n, c = img_pyramid[p_level].shape
                    displace_pyramid = [np.round(img) for img in displace]
                    n_pad = int(np.ceil(self.cal_half_window / 2**p_level))

                else:
                    pyramid_seaching_window = searching_window_pyramid_list[
                        p_level]
                    # extend displace_pyramid with upsampling of 2 and also displace value is 2 times larger
                    m, n, c = img_pyramid[p_level].shape
                    displace_pyramid = [
                        np.round(self.resampling_spline(img * 2, (m, n)))
                        for img in displace
                    ]

                    displace_pyramid = [
                        np.fmax(
                            np.fmin(displace_pyramid[0],
                                    self.cal_half_window / 2**p_level),
                            -self.cal_half_window / 2**p_level),
                        np.fmax(
                            np.fmin(displace_pyramid[1],
                                    self.cal_half_window / 2**p_level),
                            -self.cal_half_window / 2**p_level)
                    ]

                    n_pad = int(np.ceil(self.cal_half_window / 2**p_level))

                prColor(
                    'pyramid level: {}\nImage size: {}\nsearching window:{}'.
                    format(p_level, ref_pyramid[p_level].shape,
                           pyramid_seaching_window), 'cyan')
                # split the y axis into small groups, all splitted in vertical direction
                y_axis = np.arange(ref_pyramid[p_level].shape[0])
                chunks_idx_y = np.array_split(y_axis, n_tasks)

                dim = img_pyramid[p_level].shape

                ref_wa_pad = np.pad(ref_pyramid[p_level],
                                    ((n_pad + pyramid_seaching_window,
                                      n_pad + pyramid_seaching_window),
                                     (n_pad + pyramid_seaching_window,
                                      n_pad + pyramid_seaching_window),
                                     (0, 0)),
                                    'constant',
                                    constant_values=(0, 0))

                # use CPU parallel to calculate
                result_list = []
                '''
                    calculate the pixel displacement for the pyramid images
                '''
                with concurrent.futures.ProcessPoolExecutor(
                        max_workers=cores) as executor:

                    futures = []
                    for y_list in chunks_idx_y:
                        # get the stack data
                        img_wa_stack = img_pyramid[p_level][y_list, :, :]
                        ref_wa_stack = ref_wa_pad[
                            y_list[0]:y_list[-1] + 2 *
                            (n_pad + pyramid_seaching_window) + 1, :, :]

                        # start the jobs
                        futures.append(
                            executor.submit(self.displace_wavelet, y_list,
                                            img_wa_stack, ref_wa_stack,
                                            (displace_pyramid[0][y_list, :],
                                             displace_pyramid[1][y_list, :]),
                                            pyramid_seaching_window, n_pad))

                    for future in concurrent.futures.as_completed(futures):

                        try:
                            result_list.append(future.result())
                            # display the status of the program
                            Total_iter = cores * self.n_group
                            Current_iter = len(result_list)
                            percent_iter = Current_iter / Total_iter * 100
                            str_bar = '>' * (int(np.ceil(
                                percent_iter / 2))) + ' ' * (int(
                                    (100 - percent_iter) // 2))
                            prColor(
                                '\r' + str_bar + 'processing: [%3.1f%%] ' %
                                (percent_iter), 'purple')

                        except:
                            prColor('Error in the parallel calculation', 'red')

                disp_y_list = [item[0] for item in result_list]
                disp_x_list = [item[1] for item in result_list]
                y_list = [item[2] for item in result_list]

                displace_y = np.zeros((dim[0], dim[1]))
                displace_x = np.zeros((dim[0], dim[1]))

                for y, disp_x, disp_y in zip(y_list, disp_x_list, disp_y_list):
                    displace_x[y, :] = disp_x
                    displace_y[y, :] = disp_y

                displace = [
                    np.fmax(
                        np.fmin(displace_y, self.cal_half_window / 2**p_level),
                        -self.cal_half_window / 2**p_level),
                    np.fmax(
                        np.fmin(displace_x, self.cal_half_window / 2**p_level),
                        -self.cal_half_window / 2**p_level)
                ]
                prColor('displace map wrapping: {}'.format(displace[0].shape),
                        'green')
                print('max of displace: {}, min of displace: {}'.format(
                    np.amax(displace[0]), np.amin(displace[1])))

        end_time = time.time()
        prColor(
            '\r' + 'Processing time: {:0.3f} s'.format(end_time - start_time),
            'light_purple')

        # remove the padding boundary of the displacement
        displace[0] = -displace[0][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]
        displace[1] = -displace[1][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]

        DPC_y = (displace[0] - np.mean(displace[0])) * self.p_x / self.z
        DPC_x = (displace[1] - np.mean(displace[1])) * self.p_x / self.z
        
        #Oct 2024, XSHI added pixel scaling
        phase = frankotchellappa(DPC_x, DPC_y, self.p_x*self.scaling_x, self.p_x*self.scaling_y) * 2 * np.pi / self.wavelength
        
        self.time_cost = end_time - start_time

        return displace, [DPC_y, DPC_x], phase, transmission

    def run(self, result_path=None):
        if self.use_GPU:

            self.displace, self.DPC, self.phase, self.transmission = self.solver_cuda(
            )
        else:
            self.displace, self.DPC, self.phase, self.transmission = self.solver(
            )

        if result_path is not None:
            '''
            save the calculation results
            '''
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            self.result_filename = 'WXST_result'

            write_h5(
                result_path, self.result_filename, {
                    'displace_x': self.displace[1],
                    'displace_y': self.displace[0],
                    'DPC_x': self.DPC[1],
                    'DPC_y': self.DPC[0],
                    'phase': self.phase,
                    'transmission_image': self.transmission
                })

            parameter_dict = {
                'M_image': self.M_image,
                'template_window': self.N_s,
                'N_s extend': self.N_s_extend,
                'half_window': self.cal_half_window,
                'energy': self.energy,
                'wavelength': self.wavelength,
                'p_x': self.p_x*self.scaling_x,
                'p_y': self.p_x*self.scaling_y,
                'd': self.z,
                'cpu_cores': self.n_cores,
                'n_group': self.n_group,
                'wavelet_level': self.wavelet_level,
                'pyramid_level': self.pyramid_level,
                'n_iter': self.n_iter,
                'time_cost': self.time_cost,
                'use_wavelet': self.use_wavelet,
                'use_GPU': self.use_GPU,
                'wavelet_level_cut': self.wavelet_level_cut,
                'wavelet_add': self.wavelet_add_list
            }

            write_json(result_path, self.result_filename, parameter_dict)


'''
if __name__ == "__main__":
    # paremater settings
    parser = argparse.ArgumentParser(
        description='experimental data analysis for absolute phase measurement',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # shared args
    # ============================================================
    parser.add_argument('--img', type=str, default='../testdata/single-shot/sample_001.tif', help='path to sample image')
    parser.add_argument('--ref', type=str, default='../testdata/single-shot/ref_001.tif', help='path to sample image')
    parser.add_argument('--dark', type=str, default = 'None', help='dark image for image correction')
    parser.add_argument('--flat', type=str, default = 'None', help='flat image for image correction')
    parser.add_argument('--result_folder', type=str, default='../testdata/single-shot/WXST_results', help='saving folder')
    parser.add_argument(
        "--crop",
        nargs="+",
        type=int,
        default=[450, 1000, 500, 1000],
        help=
        'image crop, if is [256], central crop. if len()==4, boundary crop, if is 0, use gui crop, if is -1, use auto-crop'
    )
    parser.add_argument('--p_x', default=0.65e-6, type=float, help='pixel size')
    parser.add_argument('--scaling_x', default=1.0, type=float, help='x pixel scaling from detector to sample')
    parser.add_argument('--scaling_y', default=1.0, type=float, help='y pixel scaling from detector to sample')
    parser.add_argument('--energy', default=14e3, type=float, help='X-ray energy')
    parser.add_argument('--distance', default=500e-3, type=float, help='detector to mask distance')
    parser.add_argument('--down_sampling', type=float, default=1, help='down-sample images to reduce memory cost and accelerate speed.')
    parser.add_argument('--GPU', default=False, action='store_true', help='Use GPU or not. GPU can be 2 times faster. But multi-resolution process is disabled.')
    parser.add_argument('--use_wavelet', default=False, action='store_true', help='use wavelet transform or not.')
    parser.add_argument('--wavelet_lv_cut', default=2, type=int, help='wavelet cutting level')
    parser.add_argument('--pyramid_level', default=1, type=int, help='pyramid level used for speckle tracking.')
    parser.add_argument('--n_iter', default=1, type=int, help='number of iteration for speckle tracking. 1 is good.')
    parser.add_argument('--template_size', default=5, type=int, help='template size in the WXST')
    parser.add_argument('--window_searching', default=10, type=int, help='searching window of speckle tracking. Means the largest displacement can be calculated.')
    parser.add_argument('--nCores', default=4, type=int, help='number of CPU cores used for calculation.')
    parser.add_argument('--nGroup', default=1, type=int, help='number of groups that parallel calculation is splitted into.')

    args = parser.parse_args()

    for key, value in args.__dict__.items():
        prColor('{}: {}'.format(key, value), 'cyan')
    
    File_ref = args.ref
    File_img = args.img
    flat = args.flat
    dark = args.dark
    Folder_result = args.result_folder
    N_s = args.template_size
    cal_half_window = args.window_searching
    # the calculation window for high order pyramid
    N_s_extend = 4
    n_cores = args.nCores
    n_group = args.nGroup
    energy = args.energy
    wavelength = sc.value('inverse meter-electron volt relationship') / energy
    p_x = args.p_x
    scaling_x = args.scaling_x
    scaling_y = args.scaling_y
    z = args.distance
    pyramid_level = args.pyramid_level
    n_iter = args.n_iter
    use_GPU = args.GPU
    down_sample = args.down_sampling
    use_wavelet = args.use_wavelet
    wavelet_level_cut = args.wavelet_lv_cut

    # # roi of the images
    # M_image = int(parameter_wavelet[0])
    # # template window, the N_s nearby pixels used to represent the local pixel, 2*N_s+1
    # N_s = int(parameter_wavelet[1])
    # # the number of the area to calculate for each pixel, 2*cal_half_window X 2*cal_half_window
    # cal_half_window = int(parameter_wavelet[2])
    
    # # process number for parallel
    # n_cores = int(parameter_wavelet[3])
    # # number to reduce the each memory use
    # n_group = int(parameter_wavelet[4])

    # # energy, 10kev
    # energy = float(parameter_wavelet[5])
    
    # p_x = float(parameter_wavelet[6])
    # z = float(parameter_wavelet[7])
    # use_wavelet = int(parameter_wavelet[8])
    # wavelet_level_cut = int(parameter_wavelet[9])
    # # pyramid level to wrap the images
    # pyramid_level = int(parameter_wavelet[10])
    # n_iter = int(parameter_wavelet[11])
    # use_GPU = int(parameter_wavelet[12])
    # down_sample = float(parameter_wavelet[13])

    ref_data = load_image(File_ref)
    img_data = load_image(File_img)

    ref_data = ref_data.astype(np.float32)
    img_data = img_data.astype(np.float32)
    if args.dark == 'None':
        dark = np.zeros_like(img_data, dtype=np.float32)
    else:
        dark = load_image(dark).astype(np.float32)

    if args.flat == 'None':
        flat = np.ones_like(img_data, dtype=np.float32)
    else:
        flat = load_image(flat).astype(np.float32)

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
        # boundary crop, use the corner index [y0, y1, x0, x1]
        # boundary_crop = lambda img: img[int(args.crop[0]):int(args.crop[1]),
        #                                 int(args.crop[2]):int(args.crop[3])]
        pass
    elif len(args.crop) == 1:
        if args.crop[0] == 0:
            # use gui crop
            print("before crop------------------------------------------------")
            _, corner = crop_gui(img_data)
            print("after crop------------------------------------------------")
            
            args.crop = [
                int(corner[0][0]),
                int(corner[1][0]),
                int(corner[0][1]),
                int(corner[1][1])
            ]
        elif args.crop[0] == -1:
            prColor('auto crop------------------------------------------------', 'green')
            # use auto-crop according to the intensity boundary. rectangular shapess
            pattern_size = 5e-6 # assume 5um mask pattern
            flat = snd.uniform_filter(img_data, size=10*(pattern_size/p_x))
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
        # error input
        prColor(
            'error: wrong crop option. 0 for gui crop; [256] for central crop; [y0, y1, x0, x1] for bournday crop',
            'red')
        sys.exit()
    print(args.crop)
    
    boundary_crop = lambda img: img[int(args.crop[0]):int(args.crop[1]),
                                        int(args.crop[2]):int(args.crop[3])]
    ref_data = boundary_crop(ref_data)
    img_data = boundary_crop(img_data)

    M_image = ref_data.shape[0]
    # ref_data = image_roi(ref_data, M_image)
    # img_data = image_roi(img_data, M_image)

    size_origin = ref_data.shape

    ref_data = ref_data.astype(np.float32)
    img_data = img_data.astype(np.float32)

    # down-sample or not
    if down_sample != 1:
        prColor('down-sample image: {}'.format(down_sample),
                'cyan')
        d_size = (int(ref_data.shape[1]*down_sample), int(ref_data.shape[0]*down_sample))

        # from func import binning2
        # img = binning2(img)
        # ref = binning2(ref)
        # print(img.shape)

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

    if not os.path.exists(Folder_result):
        os.makedirs(Folder_result)
    sample_transmission = img_data / (np.abs(ref_data) + 1)
    plt.imsave(os.path.join(Folder_result, 'transmission.png'),
               sample_transmission)

    WXST_solver.run(result_path=Folder_result)
    
    displace = WXST_solver.displace
    DPC_x = WXST_solver.DPC[1]
    DPC_y = WXST_solver.DPC[0]
    phase = WXST_solver.phase
    result_filename = WXST_solver.result_filename

    # down-sample or not
    if down_sample != 1:
        prColor('scale back', 'green')
        displace_x = cv2.resize(displace[1], (size_origin[1], size_origin[0])) * (1 / down_sample)
        displace_y = cv2.resize(displace[0], (size_origin[1], size_origin[0])) * (1 / down_sample)
        displace = [displace_y, displace_x]
        DPC_x = cv2.resize(DPC_x, (size_origin[1], size_origin[0])) * (1 / down_sample)
        DPC_y = cv2.resize(DPC_y, (size_origin[1], size_origin[0])) * (1 / down_sample)
        phase = cv2.resize(phase, (size_origin[1], size_origin[0])) * (1 / down_sample)

    # save phase directly in the result folder
    img_filename = os.path.basename(File_img).split('.')[0]

    plt.imsave(os.path.join(Folder_result, 'displace_x.png'), displace[1])
    plt.imsave(os.path.join(Folder_result, 'displace_y.png'), displace[0])
    plt.imsave(os.path.join(Folder_result, 'dpc_x.png'), DPC_x)
    plt.imsave(os.path.join(Folder_result, 'dpc_y.png'), DPC_y)
    plt.imsave(os.path.join(Folder_result, 'phase.png'), phase)

    matplotlib.use('Agg')  # Non-GUI backend
    plt.figure()
    plt.imshow(displace[0])
    cbar = plt.colorbar()
    cbar.set_label('[pixels]', rotation=90)
    plt.savefig(os.path.join(Folder_result, 'displace_y_colorbar.png'), dpi=150)
    plt.close()  # Close the figure after saving

    plt.figure()
    plt.imshow(displace[1])
    cbar = plt.colorbar()
    cbar.set_label('[pixels]', rotation=90)
    plt.savefig(os.path.join(Folder_result, 'displace_x_colorbar.png'), dpi=150)
    plt.close()  # Close the figure after saving

    plt.figure()
    plt.imshow(phase)
    cbar = plt.colorbar()
    cbar.set_label('[rad]', rotation=90)
    plt.savefig(os.path.join(Folder_result, 'phase_colorbar.png'), dpi=150)
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
    plt.savefig(os.path.join(Folder_result, 'Phase_3d.png'), dpi=150)
    plt.close()    
'''