#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    	: 07 / 16 / 2022
@Author  	: Zhi Qiao
@Contact	: z.qiao1989@gmail.com
@File    	: WXST.py
@Software	: AbsolutePhase
@Desc		: Wavelet-transform-based X-ray speckle tracking (WXST) method. Use for generating initial guess of CMMI.
            Qiao, Zhi, Xianbo Shi, Rafael Celestre, and Lahsen Assoufid. “Wavelet-Transform-Based Speckle Vector Tracking Method for X-Ray Phase Imaging.” Optics Express 28, no. 22 (October 26, 2020): 33053. https://doi.org/10.1364/OE.404606.
            Qiao, Zhi, Xianbo Shi, and Lahsen Assoufid. “Single-Shot Speckle Tracking Method Based on Wavelet Transform and Multi-Resolution Analysis.” In Advances in Metrology for X-Ray and EUV Optics IX, edited by Lahsen Assoufid, Haruhiko Ohashi, and Anand Asundi, 22. Online Only, United States: SPIE, 2020. https://doi.org/10.1117/12.2569135.

'''

import numpy as np
import pywt
import os
import time
import torch
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt
import multiprocessing as ms
import concurrent.futures
import scipy.interpolate as sfit
import scipy.ndimage.filters

from aps.wavefront_analysis.common.legacy.func import prColor, slop_tracking, write_h5
from aps.wavefront_analysis.common.legacy.euclidean_dist import dist_numba

def save_figure(image_pair, path='./', p_x=1, extention='.tif'):
    """
        save the data in visible figure
    Args:
        extention (str): image format, png, bmp. Defaults to '.tif'.
    """
    if extention == '.tif':
        for each_pair in image_pair:
            if each_pair[1] is None:
                continue
            im = Image.fromarray(each_pair[1])
            im.save(os.path.join(path, each_pair[0] + extention))
    else:
        for each_pair in image_pair:
            if each_pair[1] is None:
                continue
            extent_data = np.array([
                -each_pair[1].shape[1] / 2 * p_x * 1e6,
                each_pair[1].shape[1] / 2 * p_x * 1e6,
                -each_pair[1].shape[0] / 2 * p_x * 1e6,
                each_pair[1].shape[0] / 2 * p_x *1e6
            ])
            plt.figure()
            plt.imshow(each_pair[1],
                        interpolation='bilinear',
                        extent=extent_data)
            if p_x == 1:
                plt.xlabel('x (pixel)', fontsize=22)
                plt.ylabel('y (pixel)', fontsize=22)
            else:
                plt.xlabel('x ($\mu$m)', fontsize=22)
                plt.ylabel('y ($\mu$m)', fontsize=22)
            cbar = plt.colorbar()
            cbar.set_label(each_pair[2], rotation=90, fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(path, each_pair[0] + extention), dpi=150)
            plt.close()

def save_figure_1D(image_pair, path='./', p_x=1):
    """
        save the data in visible figure
    Args:
        extention (str): image format, png, bmp. Defaults to '.tif'.
    """
    for each_pair in image_pair:
        if each_pair[1] is None:
            continue
        
        x_axis = (np.arange(len(each_pair[1])) - len(each_pair[1])/2) * p_x
        plt.figure()
        plt.plot(x_axis*1e6, each_pair[1], 'k')
        if p_x == 1:
            plt.xlabel('x (pixel)', fontsize=22)
        else:
            plt.xlabel('x ($\mu$m)', fontsize=22)
        plt.ylabel(each_pair[2], fontsize=22)
        plt.tight_layout()
        plt.savefig(os.path.join(path, each_pair[0] + '.png'), dpi=150)
        plt.close()

def save_data(data, path_folder='./', overwrite=True):
    """ save data in the path folder
        Args:
            path_folder (str, optional): result path folder. Defaults to './'.
    """
    if path_folder is not None:
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        result_filename = 'single_shot'
        kk = 1
        result_filename =  result_filename + '_{}'.format(kk)
        if not overwrite:
            while os.path.exists(
                    os.path.join(
                        path_folder,
                        result_filename + '.hdf5')) or os.path.exists(
                            os.path.join(path_folder,
                                        result_filename + '.json')):
                kk += 1
                result_filename =  result_filename + '_{}'.format(kk)
        write_h5(path_folder, result_filename, data)

def filter_erosion(image, val_thresh, filt_sz=2):
    
    """ Function to apply erosion filter in order to remove some of the 
    impulse noise 
    It returns filtered image 
    val_thresh = the value above which the pixel value must be changed
    varMargin = optional field, an integer value for the size of the 
    median filter, default value is 5

    Based on a Matlab function by S. Berujon
    """

    # Input image size
    [m, n] = image.shape
    
    # GIVEN THAT MIRROR IS USED, CONSIDER TO REMOVE EXTRA CORNER TREATMENT!
    # Indices for 'corners' of input image
    pts2 = ((0, 0, 0, 0, 1, 1, m-2, m-2, m-1, m-1, m-1, m-1 ), 
            (0, 1, n-2, n-1, 0, n-1, 0, n-1, 0, 1, n-2 , n-1))
    # Values in these 'corners'
    val2 = image[pts2]
    # Apply 2D median filter - using option mode 'mirror' equivalent to 
    # 'symmetric' in Matlab
    image_filt = scipy.ndimage.filters.median_filter(image,filt_sz, 
                                                     mode='mirror')
    # Check difference between initial and filtered images element by element    
    diff_img = abs(image - image_filt)
    # Consider this difference null for 'corners'
    diff_img[pts2] = 0
    # Check which elements have a difference above the treshold value
    pts1 = np.where(diff_img > val_thresh)
    # Calculate the percentage
    # logging.info('Percentage of points modified by my_erosion: ' 
    #              + str(float(len(pts1)) / float(n * m) * 100.0) + str(' %'))
    # Replace these elements with median value, excluding 'corners'    
    image[pts1] = image_filt[pts1]
    
    return image

def find_disp(Corr_img, XX_axis, YY_axis, sub_resolution=True):
    '''
        find the peak value in the Corr_img
        the XX_axis, YY_axis is the corresponding position for the Corr_img
    '''

    # find the maximal value and postion
    
    pos = np.unravel_index(np.argmax(Corr_img, axis=None), 
                            Corr_img.shape)
    SN_ratio, Corr_max = 0, 0
    # Compute displacement on both axes
    Corr_img_pad = np.pad(Corr_img, ((1,1), (1,1)), 'edge')
    max_pos_y = pos[0] + 1
    max_pos_x = pos[1] + 1

    dy = (Corr_img_pad[max_pos_y + 1, max_pos_x] - Corr_img_pad[max_pos_y - 1, max_pos_x]) / 2.0
    dyy = (Corr_img_pad[max_pos_y + 1, max_pos_x] + Corr_img_pad[max_pos_y - 1, max_pos_x] 
           - 2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dx = (Corr_img_pad[max_pos_y, max_pos_x + 1] - Corr_img_pad[max_pos_y, max_pos_x - 1]) / 2.0
    dxx = (Corr_img_pad[max_pos_y, max_pos_x + 1] + Corr_img_pad[max_pos_y, max_pos_x - 1] 
           - 2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dxy = (Corr_img_pad[max_pos_y + 1, max_pos_x + 1] - Corr_img_pad[max_pos_y + 1, max_pos_x - 1] 
           - Corr_img_pad[max_pos_y - 1, max_pos_x + 1] + Corr_img_pad[max_pos_y - 1, max_pos_x - 1]) / 4.0
    
    if ((dxx * dyy - dxy * dxy) != 0.0):
        det = 1.0 / (dxx * dyy - dxy * dxy)
    else:
        det = 0.0
    # the XX, YY axis resolution
    pixel_res_x = XX_axis[0, 1] - XX_axis[0,0]
    pixel_res_y = YY_axis[1, 0] - YY_axis[0,0]
    Minor_disp_x = (- (dyy * dx - dxy * dy) * det) * pixel_res_x
    Minor_disp_y = (- (dxx * dy - dxy * dx) * det) * pixel_res_y

    if sub_resolution:
        disp_x = Minor_disp_x + XX_axis[pos[0], pos[1]]  
        disp_y = Minor_disp_y + YY_axis[pos[0], pos[1]]
    else:
        disp_x = XX_axis[pos[0], pos[1]]  
        disp_y = YY_axis[pos[0], pos[1]]
    
    max_x = XX_axis[0, -1]
    min_x = XX_axis[0, 0]
    max_y = YY_axis[-1, 0]
    min_y = YY_axis[0, 0]

    if disp_x > max_x:
        disp_x = max_y
    elif disp_x < min_x:
        disp_x = min_x

    if disp_y > max_y:
        disp_y = max_y
    elif disp_y < min_y:
        disp_y = min_y

    return disp_y, disp_x, SN_ratio, Corr_max

def Wavelet_transform(img, wavelet_method='db6', w_level=1, return_level=1):
    '''
        do the wavelet transfrom for the 3D image data 
    '''
    coeffs = pywt.wavedec(img, wavelet_method, level=w_level, mode='zero', axis=0)

    coeffs_filter = np.concatenate(coeffs[0:return_level], axis=0)
    coeffs_filter = np.moveaxis(coeffs_filter, 0, -1)

    level_name = []
    for kk in range(w_level):
        level_name.append('D{:d}'.format(kk+1))
    level_name.append('A{:d}'.format(w_level))
    level_name = level_name[-return_level:]
    
    return coeffs_filter, level_name

# define a function to apply to each image
def wavedec_func(img, y_list, wavelet_method='db6', w_level=5, return_level=4):

    coeffs = pywt.wavedec(img, wavelet_method, level=w_level, mode='zero', axis=0)
    coeffs_filter = np.concatenate(coeffs[0:return_level], axis=0)
    coeffs_filter = np.moveaxis(coeffs_filter, 0, -1)
    # coeffs_filter = img
    level_name = []
    for kk in range(w_level):
        level_name.append('D{:d}'.format(kk+1))
    level_name.append('A{:d}'.format(w_level))
    level_name = level_name[-return_level:]
    
    return coeffs_filter, level_name, y_list

def Wavelet_transform_multiprocess(img, n_cores, wavelet_method='db6', w_level=1, return_level=1):
    '''
        use multi-process to accelerate wavelet transform
        img is in a shape of [ch, H, W]
    '''
    cores = ms.cpu_count()
    prColor('Computer available cores: {}'.format(cores), 'green')

    if cores > n_cores:
        cores = n_cores
    else:
        cores = ms.cpu_count()
    prColor('Use {} cores'.format(cores), 'light_purple')
    n_tasks = cores

    # split the y axis into small groups, all splitted in vertical direction
    y_axis = np.arange(img.shape[1])
    chunks_idx_y = np.array_split(y_axis, n_tasks)

    dim = img.shape

    # use CPU parallel to calculate
    result_list = []
    '''
        calculate the pixel displacement for the pyramid images
    '''

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=cores) as executor:

        futures = []
        img_list = []
        for y_list in chunks_idx_y:
            # get the stack data
            img_split = img[:, y_list, :]
            img_list.append(img_split)
        
        result_list = []
        for result in executor.map(wavedec_func, img_list, chunks_idx_y, [wavelet_method]*len(chunks_idx_y), [w_level]*len(chunks_idx_y), [return_level]*len(chunks_idx_y)):
            result_list.append(result)

    img_wavelet_list = [item[0] for item in result_list]
    level_name = [item[1] for item in result_list]
    y_list = [item[2] for item in result_list]

    img_wavelet = np.zeros((dim[1], dim[2], img_wavelet_list[0].shape[-1]), dtype=img.dtype)

    for y, img_w in zip(y_list, img_wavelet_list):
        img_wavelet[y, :, :] = img_w

    return img_wavelet, level_name[0]

def image_preprocess(image, have_dark, dark_img, have_flat, flat_img):
    '''
        do the flat or dark correction for the images
        img:            image to be corrected
        have_dark:      if there is dark
        dark_img:           dark image
        have_flat:      if there is flat
        flat_img:           flat image        
    '''
    if (have_flat != 0 and have_dark != 0):
        numerator = (flat_img - dark_img).clip(0.00000001)
        # numerator = numerator / np.amax(numerator)
        image = ((image - dark_img) / numerator) * np.amax(image)
    elif (have_dark != 0):
        image = (image - dark_img).clip(0.00000001)
    elif (have_flat != 0):
        flat_img[flat_img == 0] = 0.00000001
        # flat_img = flat_img / np.amax(flat_img)
        image = (image / flat_img) * np.amax(image)

    return image

def cost_volume(first, second, search_range):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        first: Level of the feature pyramid of Image1
        second: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = nn.functional.pad(second, (search_range, search_range, search_range, search_range))
    _, h, w = first.shape
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            second_slice = padded_lvl[:, y:y+h, x:x+w]
            cost = torch.mean(first * second_slice, dim=0, keepdim=True)
            cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, dim=0)

    return cost_vol
    
class WXST:
    def __init__(self,
                 img,
                 ref,
                 N_s=5,
                 cal_half_window=20,
                 N_s_extend=4,
                 n_cores=4,
                 n_group=4,
                 wavelet_level_cut=2,
                 pyramid_level=2,
                 n_iter=1,
                 use_estimate=False,
                 use_wavelet=True,
                 use_GPU=False):
        self.img_data = img
        self.ref_data = ref
        # roi of the images
        self.M_image = np.amin(img.shape)
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
        if torch.cuda.is_available() and use_GPU:
            prColor('Use GPU found. Disable multi-resolution', 'cyan')
            self.use_GPU = True
            # for GPU, there's no multi-resolution process
            self.pyramid_level = 0
        else:
            prColor('No gpu found. Use CPU instead.', 'cyan')
            self.use_GPU = False


        if self.use_estimate:
            # get initial estimation from the cv2 flow tracking
            displace_estimate = slop_tracking(self.ref_data,
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
        # get the pyramid data
        # method 1, pyramid then stack the image, which means the template window size is increasing for different pyramid level
        # get the pyramid wrapping images
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

                # use mutli-threading to speed up wavelet transform
                img_wa, level_name = Wavelet_transform_multiprocess(
                    img_pyramid[p_level],
                    16,
                    wavelet_method,
                    w_level=self.wavelet_level,
                    return_level=coefs_level + wavelevel_add)
                    
                img_pyramid[p_level] = img_wa
                
                ref_wa, level_name = Wavelet_transform_multiprocess(
                    ref_pyramid[p_level],
                    16,
                    wavelet_method,
                    w_level=self.wavelet_level,
                    return_level=coefs_level + wavelevel_add)
                # print(ref_wa.dtype, ref_wa.shape)
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
        '''
            calculate the coefficient of each pixel
        '''
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

                # get the correlation matrix
                '''
                    euclidean distance
                '''
                Corr_img = dist_numba(img_wa_line, ref_wa_data)
                '''
                    use gradient to find the peak
                '''
                disp_y[yy, xx], disp_x[yy, xx], SN_ratio, max_corr = find_disp(
                    Corr_img, XX, YY, sub_resolution=True)

        disp_add_y = displace_pyramid[0] + disp_y
        disp_add_x = displace_pyramid[1] + disp_x
        return disp_add_y, disp_add_x, y_list
    

    def displace_torch(self, img_wa_stack, ref_wa_stack,
                         cal_half_window):
        '''
            calculate the coefficient of each pixel
        '''
        dim = img_wa_stack.shape
        disp_x = np.zeros((dim[0], dim[1]))
        disp_y = np.zeros((dim[0], dim[1]))

        start_time = time.time()
        img_stack_cuda = torch.from_numpy(np.moveaxis(img_wa_stack, -1, 0)).cuda()
        ref_stack_cuda = torch.from_numpy(np.moveaxis(ref_wa_stack, -1, 0)).cuda()

        cost_vol = cost_volume(img_stack_cuda, ref_stack_cuda, search_range=cal_half_window)
        cost_vol = cost_vol.cpu().numpy()

        end_time = time.time()
        prColor('time cost for cost vol: {}'.format(end_time - start_time), 'cyan')

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
                    executor.submit(self.find_disp_parallel, Corr_img, XX, YY, y_list)
                )

            for future in concurrent.futures.as_completed(futures):
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

        disp_y_list = [item[0] for item in result_list]
        disp_x_list = [item[1] for item in result_list]
        y_list = [item[2] for item in result_list]

        for y, dx, dy in zip(y_list, disp_x_list, disp_y_list):
            disp_x[y, :] = dx
            disp_y[y, :] = dy

        return disp_y, disp_x


    def find_disp_parallel(self, Corr_img, XX, YY, y_list):
        dim = Corr_img.shape
        window_size = int(np.sqrt(dim[0]))
        disp_y = np.zeros((dim[1], dim[2]))
        disp_x = np.zeros((dim[1], dim[2]))
        for yy in range(dim[1]):
            for xx in range(dim[2]):

                temp = Corr_img[:, yy, xx].reshape(window_size, window_size)
                disp_y[yy, xx], disp_x[yy, xx], SN_ratio, max_corr = find_disp(
                        temp, XX, YY, sub_resolution=True)

        return disp_y, disp_x, y_list

    def solver_cuda(self):

        ref_wavelet, img_wavelet = self.wavelet_data()
        transmission = self.img_data / (np.abs(self.ref_data+1))
        for attr in ('img_data', 'ref_data'):
            self.__dict__.pop(attr, None)

        start_time = time.time()

        displace_y, displace_x = self.displace_torch(img_wavelet[0], ref_wavelet[0], cal_half_window=self.cal_half_window)

        end_time = time.time()
        prColor(
            '\r' + 'Processing time: {:0.3f} s'.format(end_time - start_time),
            'light_purple')
        
        displace = [
                    np.fmax(
                        np.fmin(displace_y, self.cal_half_window),
                        -self.cal_half_window),
                    np.fmax(
                        np.fmin(displace_x, self.cal_half_window),
                        -self.cal_half_window)
                ]
        prColor('displace map wrapping: {}'.format(displace[0].shape),
                'green')
        print('max of displace: {}, min of displace: {}'.format(
            np.amax(displace[0]), np.amin(displace[1])))
        displace[0] = -displace[0]
        displace[1] = -displace[1]

        return displace, transmission
        

    def solver(self):

        ref_pyramid, img_pyramid = self.wavelet_data()
        transmission = self.img_data / (np.abs(self.ref_data+1))
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

                    # n_pad = int(np.ceil(cal_half_window / 2**p_level) * 2**(pyramid_level-p_level))
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

                        futures.append(
                            executor.submit(self.displace_wavelet, y_list,
                                            img_wa_stack, ref_wa_stack,
                                            (displace_pyramid[0][y_list, :],
                                            displace_pyramid[1][y_list, :]),
                                            pyramid_seaching_window, n_pad))

                    for future in concurrent.futures.as_completed(futures):
                        
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
        displace[0] = -displace[0]
        displace[1] = -displace[1]

        self.time_cost = end_time - start_time

        return displace, transmission

    def run(self):
        if self.use_GPU:
            self.displace, self.transmission = self.solver_cuda()
        else:
            self.displace, self.transmission = self.solver()



''' 
# LEGACY CODE THAT MUST NOT BE CALLED
if __name__ == "__main__":
    if len(sys.argv) == 1:
        import cv2  # here to avoid conflict with PyQt5

        Folder_path = '../testdata/single-shot/'
        File_ref = os.path.join(Folder_path, 'ref_001.tif')
        File_img = os.path.join(Folder_path, 'sample_001.tif')

        Folder_result = os.path.join(Folder_path, 'XST_result')
        # [image_size, template_window, cal_half_window, n_group, n_cores, energy, pixel_size, distance, use_wavelet, wavelet_ct, pyramid level, n_iteration]
        parameter_wavelet = [
            1024, 5, 20, 4, 4, 20e3, 0.65e-6, 200e-3, 0, 1, 0, 1
        ]

    elif len(sys.argv) == 4:
        File_img = sys.argv[1]
        File_ref = sys.argv[2]
        Folder_result = sys.argv[3]
        # [image_size, template_window, cal_half_window, n_group, n_cores, energy, pixel_size, distance, wavelet_ct, pyramid level, n_iteration]
        parameter_wavelet = [
            512, 5, 20, 4, 4, 14e3, 0.65e-6, 500e-3, 1, 2, 2, 1
        ]
    elif len(sys.argv) == 16:
        File_img = sys.argv[1]
        File_ref = sys.argv[2]
        Folder_result = sys.argv[3]
        parameter_wavelet = sys.argv[4:]
    else:
        prColor('Wrong parameters! should be: sample, ref, result', 'red')

    prColor('folder: {}'.format(Folder_result), 'green')
    # roi of the images
    M_image = int(parameter_wavelet[0])
    # template window, the N_s nearby pixels used to represent the local pixel, 2*N_s+1
    N_s = int(parameter_wavelet[1])
    # the number of the area to calculate for each pixel, 2*cal_half_window X 2*cal_half_window
    cal_half_window = int(parameter_wavelet[2])
    # the calculation window for high order pyramid
    N_s_extend = 4

    # process number for parallel
    n_cores = int(parameter_wavelet[3])
    # number to reduce the each memory use
    n_group = int(parameter_wavelet[4])

    # energy, 10kev
    energy = float(parameter_wavelet[5])
    wavelength = sc.value('inverse meter-electron volt relationship') / energy
    p_x = float(parameter_wavelet[6])
    z = float(parameter_wavelet[7])
    use_wavelet = int(parameter_wavelet[8])
    wavelet_level_cut = int(parameter_wavelet[9])
    # pyramid level to wrap the images
    pyramid_level = int(parameter_wavelet[10])
    n_iter = int(parameter_wavelet[11])
    down_sample = 0.5


    ref_data = load_image(File_ref)
    img_data = load_image(File_img)

    # do image alignment
    if True:
        pos_shift, ref_data = image_align(img_data, ref_data)
        max_shift = int(np.amax(np.abs(pos_shift))+1)
        crop_area = lambda img: img[max_shift:-max_shift, max_shift:-max_shift]
        img_data = crop_area(img_data)
        ref_data = crop_area(ref_data)

    # radio_flat = load_images(Folder_radio, 'flat*.tif')
    # radio_dark = load_images(Folder_radio, 'dark*.tif')

    # ref_data = image_preprocess(ref_data, have_dark=1, dark_img=radio_dark, have_flat=1, flat_img=radio_flat)
    # img_data = image_preprocess(img_data, have_dark=1, dark_img=radio_dark, have_flat=1, flat_img=radio_flat)

    # take out the roi
    ref_data = image_roi(ref_data, M_image)
    img_data = image_roi(img_data, M_image)

    size_origin = ref_data.shape
    # down-sample or not
    if down_sample != 1:
        prColor('down-sample image: {}'.format(down_sample), 'cyan')
        d_size = (int(ref_data.shape[1]*down_sample), int(ref_data.shape[0]*down_sample))

        # from func import binning2
        # img = binning2(img)
        # ref = binning2(ref)
        # print(img.shape)

        img_data = cv2.resize(img_data, d_size)
        ref_data = cv2.resize(ref_data, d_size)

    ref_data = ref_data.astype(np.float32)
    img_data = img_data.astype(np.float32)

    # Nx = [1110, 1132+2000]
    # Ny = [370, 420+2300]

    # ref_data = ref_data[Ny[0]:Ny[1], Nx[0]:Nx[1]]
    # img_data = img_data[Ny[0]:Ny[1], Nx[0]:Nx[1]]

    WXST_solver = WXST(img_data.astype(np.float32),
                       ref_data.astype(np.float32),
                       N_s=N_s,
                       cal_half_window=cal_half_window,
                       N_s_extend=N_s_extend,
                       n_cores=n_cores,
                       n_group=n_group,
                       wavelet_level_cut=wavelet_level_cut,
                       pyramid_level=pyramid_level,
                       n_iter=n_iter,
                       use_wavelet=use_wavelet,
                       use_estimate=False)

    # # get initial estimation from the cv2 flow tracking
    # displace_estimate = slope_tracking(ref_data,
    #                                    img_data,
    #                                    N_window=cal_half_window)

    if not os.path.exists(Folder_result): os.makedirs(Folder_result)

    sample_transmission = img_data / (np.abs(ref_data)+1)

    plt.imsave(os.path.join(Folder_result, 'transmission.png'), sample_transmission)

    WXST_solver.run()
    displace_y, displace_x = WXST_solver.displace

    if down_sample != 1:
        displace_x = cv2.resize(displace_x, (size_origin[1], size_origin[0])) * (1 / down_sample)
        displace_y = cv2.resize(displace_y, (size_origin[1], size_origin[0])) * (1 / down_sample)

    crop_boundary = cal_half_window + N_s*int(1/down_sample)

    displace_y = displace_y[crop_boundary:-crop_boundary, crop_boundary:-crop_boundary]
    displace_x = displace_x[crop_boundary:-crop_boundary, crop_boundary:-crop_boundary]

    displace = [displace_y, displace_x]
    DPC_y = (displace_y - np.mean(displace_y)) * p_x / z
    DPC_x = (displace_x - np.mean(displace_x)) * p_x / z

    phase = frankotchellappa(DPC_x, DPC_y) * p_x * 2 * np.pi / (1.24/energy*1e-6)

    curve_y = np.gradient(displace_y, axis=0)/z
    curve_x = np.gradient(displace_x, axis=1)/z
    prColor('mean curvature: {}y    {}x'.format(1/np.mean(curve_y), 1/np.mean(curve_x)), 'cyan')

    block_width = int(10*5e-6 / p_x) + 2 * cal_half_window
    print(block_width)

    line_displace_y = displace_y[:, int(displace_y.shape[0] // 2 - block_width // 2):int(displace_y.shape[0] // 2 - block_width // 2 + block_width)]
    line_displace_x = displace_x[int(displace_y.shape[0] // 2 - block_width // 2):int(displace_y.shape[0] // 2 - block_width // 2 + block_width), :]

    line_displace = [np.mean(line_displace_y, axis=1), np.mean(line_displace_x, axis=0)]
    line_displace = [line_displace[0] - np.mean(line_displace[0]), line_displace[1] - np.mean(line_displace[1])]

    line_dpc = [line_displace[0] * p_x / z, 
                line_displace[1] * p_x / z]
    line_phase = [np.cumsum(line_dpc[0])*p_x * 2 * np.pi / (1.24/energy*1e-6),
                    np.cumsum(line_dpc[1])*p_x * 2 * np.pi / (1.24/energy*1e-6)]
    line_curve = [np.gradient(line_displace[0])/z,
                        np.gradient(line_displace[1])/z]

    prColor('mean curvature: {}y    {}x'.format(1/np.mean(curve_y), 1/np.mean(curve_x)), 'cyan')

    line_curve_filter = [snd.gaussian_filter(line_curve[0], 21), snd.gaussian_filter(line_curve[1], 21)]

    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(line_curve[0], 'k')
    plt.plot(line_curve_filter[0], 'r')
    plt.xlabel('[px]')
    plt.ylabel('[1/m]')
    plt.grid()
    plt.title('vertical curvature')
    plt.subplot(122)
    plt.plot(line_curve[1], 'k')
    plt.plot(line_curve_filter[1], 'r')
    plt.xlabel('[px]')
    plt.ylabel('[1/m]')
    plt.grid()
    plt.title('horizontal curvature')
    plt.savefig(os.path.join(Folder_result, 'linecurve_filter.png'), dpi=150)
    plt.close()

    # To do: saving data and figures. Get 1D line profile and curvature profile.
    save_figure(image_pair=[['displace_x', displace_x, '[px]'],
                            ['displace_y', displace_y, '[px]'],
                            ['curve_y', curve_y, '[1/m]'],
                            ['curve_x', curve_x, '[1/m]'],
                            ['phase', phase, '[rad]']], path=Folder_result, p_x=p_x, extention='.png')

    save_figure_1D(image_pair=[['line_displace_x', line_displace[1], '[px]'],
                            ['line_phase_x', line_phase[1], '[rad]'],
                            ['line_displace_y', line_displace[0], '[px]'],
                            ['line_phase_y', line_phase[0], '[rad]'],
                            ['line_curve_y', line_curve_filter[0], '[1/m]'],
                            ['line_curve_x', line_curve_filter[1], '[1/m]']], path=Folder_result, p_x=p_x)

    save_data({'displace_x': displace_x,
               'displace_y': displace_y,
               'phase': phase,
               'line_phase_y': line_phase[0],
               'line_displace_y': line_displace[0],
               'line_curve_y': line_curve_filter[0],
               'line_phase_x': line_phase[1],
               'line_displace_x': line_displace[1],
               'line_curve_x': line_curve_filter[1]},
              Folder_result, p_x)
'''