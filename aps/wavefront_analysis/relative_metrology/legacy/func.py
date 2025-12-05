#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    	: 08 / 08 / 2022
@Author  	: Zhi Qiao
@Contact	: z.qiao1989@gmail.com
@File    	: func.py
@Software	: WaveletSBI
@Desc		: collection of functions used for wavelet SBI
'''

import numpy as np
import sys
import pywt
import os
from PIL import Image
import multiprocessing as ms
import concurrent.futures
import glob
import torch
from torch import nn
import scipy.ndimage as snd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib

def load_images(folder_path, filename_format='*.tif'):
    """
    load_images to load images

    Args:
        folder_path (str): image folder
        filename_format (str, optional): image filename and extension. Defaults to '*.tif'.

    Returns:
        image data
    """
    img = [np.array(Image.open(f_single)) for f_single in sorted(glob.glob(os.path.join(folder_path, filename_format)))]
    if len(img) == 0: raise IOError('Error: wrong data path. No data is loaded. {}'.format(folder_path))

    return np.array(img)


def load_image(file_path):
    """
    load_images to load images

    Args:
        Folder_path (str): image folder
        filename_format (str, optional): image filename and extension. Defaults to '*.tif'.

    Returns:
        image data
    """
    if os.path.exists(file_path): img = np.array(Image.open(file_path))
    else: raise IOError('Error: wrong data path. No data is loaded.')

    return np.array(img)

def slope_tracking(img, ref, n_window=15):
    """
    slope_tracking to use opencv to track the displacement movement roughly

    Args:
        img (ndarray): sample image
        ref (ndarray): ref image
        n_window (int, optional): window size. Defaults to 15.

    Returns:
        the displacement of the pixels in the images  [dips_H, disp_V]
    """
    # the pyramid scale, make the undersampling image
    pyramid_scal = 0.5
    # the pyramid levels
    levels = 2
    # window size of the displacement calculation
    winsize = n_window
    # iteration for the calculation
    n_iter = 10
    # neighborhood pixel size to calculate the polynomial expansion, which makes the results smooth but blurred
    n_poly = 3
    # standard deviation of the Gaussian that is used to smooth derivatives used as a basis for
    # the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
    sigma_poly = 1.2

    flags = 1

    import cv2 # prevents system library conflict
    flow = cv2.calcOpticalFlowFarneback(ref, img, None, pyramid_scal, levels,
                                        winsize, n_iter, n_poly, sigma_poly,
                                        flags)

    displace = np.array([flow[..., 1], flow[..., 0]])

    return displace

def cost_volume(first, second, search_range):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        first: Level of the feature pyramid of Image1
        second: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = nn.functional.pad(
        second, (search_range, search_range, search_range, search_range))
    _, h, w = first.shape
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            # slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            second_slice = padded_lvl[:, y:y + h, x:x + w]
            # cost = tf.reduce_mean(first * second_slice, axis=3, keepdims=True)
            cost = torch.mean(first * second_slice, dim=0, keepdim=True)
            cost_vol.append(cost)
    # cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = torch.cat(cost_vol, dim=0)

    return cost_vol

def image_roi(img, M):
    '''
        take out the interested area of the all data.
        input:
            img:            image data, 2D or 3D array
            M:              the interested array size
                            if M = 0, use the whole size of the data
        output:
            img_data:       the area of the data
    '''
    img_size = img.shape
    if M == 0:
        return img
    elif len(img_size) == 2:
        if M > min(img_size):
            return img
        else:
            pos_0 = np.arange(M) - np.round(M / 2) + np.round(img_size[0] / 2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M / 2) + np.round(img_size[1] / 2)
            pos_1 = pos_1.astype('int')
            img_data = img[pos_0[0]:pos_0[-1] + 1, pos_1[0]:pos_1[-1] + 1]
    elif len(img_size) == 3:
        if M > min(img_size[1:]):
            return img
        else:
            pos_0 = np.arange(M) - np.round(M / 2) + np.round(img_size[1] / 2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M / 2) + np.round(img_size[2] / 2)
            pos_1 = pos_1.astype('int')
            img_data = np.zeros((img_size[0], M, M))
            for kk, pp in enumerate(img):
                img_data[kk] = pp[pos_0[0]:pos_0[-1] + 1,
                                  pos_1[0]:pos_1[-1] + 1]

    return img_data

def prColor(word, color_type):
    ''' function to print color text in terminal
        input:
            word:           word to print
            color_type:     which color
                            'red', 'green', 'yellow'
                            'light_purple', 'purple'
                            'cyan', 'light_gray'
                            'black'
    '''
    end_c = '\033[00m'
    if color_type == 'red':
        start_c = '\033[91m'
    elif color_type == 'green':
        start_c = '\033[92m'
    elif color_type == 'yellow':
        start_c = '\033[93m'
    elif color_type == 'light_purple':
        start_c = '\033[94m'
    elif color_type == 'purple':
        start_c = '\033[95m'
    elif color_type == 'cyan':
        start_c = '\033[96m'
    elif color_type == 'light_gray':
        start_c = '\033[97m'
    elif color_type == 'black':
        start_c = '\033[98m'
    else:
        print('color not right')
        sys.exit()

    print(start_c + str(word) + end_c)

def frankotchellappa(dpc_x, dpc_y, p_x=1.0, p_y=1.0):
    '''
        Frankt-Chellappa Algorithm
        input:
            dpc_x:              the differential phase along x
            dpc_y:              the differential phase along y
            p_x:                pixel size along x (physical units)
            p_y:                pixel size along y (physical units)      
        output:
            phi:                phase calculated from the dpc
    '''
    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    fftshift = lambda x: np.fft.fftshift(x)

    NN, MM = dpc_x.shape

    # Adjust the spatial frequencies to incorporate the different pixel sizes
    wx, wy = np.meshgrid(np.fft.fftfreq(MM, d=p_x) * 2 * np.pi,
                         np.fft.fftfreq(NN, d=p_y) * 2 * np.pi,
                         indexing='xy')
    wx = fftshift(wx)
    wy = fftshift(wy)
    
    numerator = -1j * wx * fft2(dpc_x) - 1j * wy * fft2(dpc_y)
    # here use the numpy.fmax method to eliminate the zero point of the division
    denominator = np.fmax((wx)**2 + (wy)**2, np.finfo(float).eps)

    div = numerator / denominator

    phi = np.real(ifft2(div))

    phi -= np.mean(np.real(phi))

    return phi

# calculate the displacement from the correlation map
def find_disp(Corr_img, XX_axis, YY_axis, sub_resolution=True):
    """
    find the peak value in the Corr_img
        the XX_axis, YY_axis is the corresponding position for the Corr_img
    """

    # find the maximal value and postion
    Corr_max = np.amax(Corr_img)
    pos = np.unravel_index(np.argmax(Corr_img, axis=None), Corr_img.shape)

    # Calculate the average to which to compare the signal
    avg = (np.sum(np.abs(Corr_img)) - np.abs(Corr_max)) \
           / (Corr_img.shape[0] * Corr_img.shape[1] -1) if (Corr_img.shape[0] * Corr_img.shape[1] -1) != 0 else 0

    # Assign the signal-to-noise
    SN_ratio = Corr_max / avg if avg != 0 else 0
    if (avg):
        SN_ratio = Corr_max / float(avg)
    else:
        SN_ratio = 0.0

    # Compute displacement on both axes
    Corr_img_pad = np.pad(Corr_img, ((1, 1), (1, 1)), 'edge')
    max_pos_y = pos[0] + 1
    max_pos_x = pos[1] + 1

    dy = (Corr_img_pad[max_pos_y + 1, max_pos_x] -
          Corr_img_pad[max_pos_y - 1, max_pos_x]) / 2.0
    dyy = (Corr_img_pad[max_pos_y + 1, max_pos_x] +
           Corr_img_pad[max_pos_y - 1, max_pos_x] -
           2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dx = (Corr_img_pad[max_pos_y, max_pos_x + 1] -
          Corr_img_pad[max_pos_y, max_pos_x - 1]) / 2.0
    dxx = (Corr_img_pad[max_pos_y, max_pos_x + 1] +
           Corr_img_pad[max_pos_y, max_pos_x - 1] -
           2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dxy = (Corr_img_pad[max_pos_y + 1, max_pos_x + 1] -
           Corr_img_pad[max_pos_y + 1, max_pos_x - 1] -
           Corr_img_pad[max_pos_y - 1, max_pos_x + 1] +
           Corr_img_pad[max_pos_y - 1, max_pos_x - 1]) / 4.0

    if ((dxx * dyy - dxy * dxy) != 0.0):
        det = 1.0 / (dxx * dyy - dxy * dxy)
    else:
        det = 0.0
    # the XX, YY axis resolution
    pixel_res_x = XX_axis[0, 1] - XX_axis[0, 0]
    pixel_res_y = YY_axis[1, 0] - YY_axis[0, 0]
    Minor_disp_x = (-(dyy * dx - dxy * dy) * det) * pixel_res_x
    Minor_disp_y = (-(dxx * dy - dxy * dx) * det) * pixel_res_y

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
    """
    Wavelet_transform for the 3D image data
    
    img:                image data
    wavelet_method:     relative_metrology to be used
    w_level:            wavelet level
    return_level:       returned wavelet level

    Returns:
        wavelet coefficients, level name
    """
    coeffs = pywt.wavedec(img,
                          wavelet_method,
                          level=w_level,
                          mode='zero',
                          axis=0)

    coeffs_filter = np.concatenate(coeffs[0:return_level], axis=0)
    coeffs_filter = np.moveaxis(coeffs_filter, 0, -1)

    level_name = []
    for kk in range(w_level):
        level_name.append('D{:d}'.format(kk + 1))
    level_name.append('A{:d}'.format(w_level))
    level_name = level_name[-return_level:]

    return coeffs_filter, level_name

# define a function to apply to each image
def wavedec_func(img, y_list, wavelet_method='db6', w_level=5, return_level=4):
    """
    decompose the data into wavelet data

    Args:
        img (ndarray): image data
        y_list (list): y axis position
        wavelet_method (str, optional): relative_metrology to be used. Defaults to 'db6'.
        w_level (int, optional): wavelet level. Defaults to 5.
        return_level (int, optional): wavelet level of returned data. Defaults to 4.

    Returns:
        wavelet coefficients, level name, y axis
    """
    coeffs = pywt.wavedec(img,
                          wavelet_method,
                          level=w_level,
                          mode='zero',
                          axis=0)
    coeffs_filter = np.concatenate(coeffs[0:return_level], axis=0)
    coeffs_filter = np.moveaxis(coeffs_filter, 0, -1)

    level_name = []
    for kk in range(w_level):
        level_name.append('D{:d}'.format(kk + 1))
    level_name.append('A{:d}'.format(w_level))
    level_name = level_name[-return_level:]

    return coeffs_filter, level_name, y_list

def wavelet_transform_multiprocess(img,
                                   n_cores,
                                   wavelet_method='db6',
                                   w_level=1,
                                   return_level=1):
    """
    use multi-process to accelerate wavelet transform
    img is in a shape of [ch, H, W]

    Args:
        img (ndarray): image data
        n_cores (int): number of cpu cores used
        wavelet_method (str, optional): relative_metrology used. Defaults to 'db6'.
        w_level (int, optional): wavelet level. Defaults to 1.
        return_level (int, optional): return wavelet level. Defaults to 1.

    Returns:
        wavelet coefficients, level name
    """

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:

        futures = []
        img_list = []
        for y_list in chunks_idx_y:
            # get the stack data
            img_split = img[:, y_list, :]
            img_list.append(img_split)

        result_list = []
        for result in executor.map(wavedec_func, img_list, chunks_idx_y,
                                   [wavelet_method] * len(chunks_idx_y),
                                   [w_level] * len(chunks_idx_y),
                                   [return_level] * len(chunks_idx_y)):
            result_list.append(result)

    img_wavelet_list = [item[0] for item in result_list]
    level_name = [item[1] for item in result_list]
    y_list = [item[2] for item in result_list]

    img_wavelet = np.zeros((dim[1], dim[2], img_wavelet_list[0].shape[-1]),
                           dtype=img.dtype)

    for y, img_w in zip(y_list, img_wavelet_list):
        img_wavelet[y, :, :] = img_w

    return img_wavelet, level_name[0]

def filter_erosion(image, val_thresh, filt_sz=2):
    import scipy.ndimage.filters
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
    pts2 = ((0, 0, 0, 0, 1, 1, m - 2, m - 2, m - 1, m - 1, m - 1, m - 1),
            (0, 1, n - 2, n - 1, 0, n - 1, 0, n - 1, 0, 1, n - 2, n - 1))
    # Values in these 'corners'
    val2 = image[pts2]
    # Apply 2D median filter - using option mode 'mirror' equivalent to
    # 'symmetric' in Matlab
    image_filt = scipy.ndimage.filters.median_filter(image,
                                                     filt_sz,
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

def write_h5(result_path, file_name, data_dict):
    """
    write_h5 to save data in hdf5 file

    Args:
        result_path (str): result folder
        file_name (str): saved file name
        data_dict (dict): data dict to be saved
    """
    import h5py
    import os
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with h5py.File(os.path.join(result_path, file_name + '.hdf5'), 'w') as f:
        for key_name in data_dict:
            f.create_dataset(key_name,
                             data=data_dict[key_name],
                             compression="gzip",
                             compression_opts=9)
    prColor('result hdf5 file : {} saved'.format(file_name + '.hdf5'), 'green')

def read_h5(file_path, key_name, print_key=False):
    """
    read_h5 to load data from hdf5 file

    Args:
        file_path (str): file path to the hdf5 file
        key_name (str): data key of the hdf5 file
        print_key (bool, optional): print the data key or not. Defaults to False.

    Returns:
        loaded data
    """
    import h5py
    import os
    if not os.path.exists(file_path):
        prColor('Wrong file path', 'red')
        sys.exit()

    with h5py.File(file_path, 'r') as f:
        # List all groups
        if print_key:
            prColor("Keys: {}".format(list(f.keys())), 'green')
        data = f[key_name][:]
    return data

def write_json(result_path, file_name, data_dict):
    """
    write_json to save data into json file

    Args:
        result_path (str): result folder
        file_name (str): file name
        data_dict (dict): dict to be saved
    """
    import os
    import json
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_name_para = os.path.join(result_path, file_name + '.json')
    with open(file_name_para, 'w') as fp:
        json.dump(data_dict, fp, indent=0)

    prColor('result json file : {} saved'.format(file_name + '.json'), 'green')

def read_json(filepath, print_para=False):
    """
    read_json to load data from json file

    Args:
        filepath (str): file path
        print_para (bool, optional): print data key or not. Defaults to False.

    Returns:
        loaded data
    """
    import os
    import json
    if not os.path.exists(filepath):
        prColor('Wrong file path', 'red')
        sys.exit()
    # file_name_para = os.path.join(result_path, file_name+'.json')
    with open(filepath, 'r') as fp:
        data = json.load(fp)
        if print_para:
            prColor('parameters: {}'.format(data), 'green')

    return data

def save_img(img, filename):
    """
    save_img to save data into tif image

    Args:
        img (ndarray): image data
        filename (str): file path to be saved
    """
    im = Image.fromarray(img)
    im.save(filename)

def image_align(image, offset_image):
    '''
        here's a function to do the alignment of two images.
        the offset_image is shifted relatively to image to find the best position
        and return the shifted back offset_image
        input:
            image:              the first image
            offset_image:       the second image
        output:
            pos:                best shift postion to maximize the correlation
            image_back:         the image after alignment
    '''
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import fourier_shift

    shift, error, diffphase = phase_cross_correlation(image,
                                                      offset_image,
                                                      upsample_factor=100)

    print(
        'shift dist: {}, alignment error: {} and phase difference: {}'.format(
            shift, error, diffphase))
    image_back = fourier_shift(np.fft.fftn(offset_image), shift)
    image_back = np.real(np.fft.ifftn(image_back))
    return shift, image_back

def auto_crop(img, shrink=0.9, count=None):
    # auto-crop to find the rectangular area from the image intensity
    if count is None:
        img_seg = np.ones(img.shape) * (img > np.mean(img))
        
        #img_filter = snd.uniform_filter(img, 15)
        #img_seg = np.ones(img.shape) * (img_filter > np.mean(img_filter))
        #from matplotlib import pyplot as plt
        #plt.figure()
        #plt.imshow(img_filter)
        #plt.colorbar()
        #plt.figure()
        #plt.imshow(img_seg)
        #plt.colorbar()
        
        #plt.show()
        
    else:
        img_seg = np.ones(img.shape) * (img > count)
    cen = snd.measurements.center_of_mass(img_seg)
    cen_x, cen_y = int(cen[0]), int(cen[1])

    # find the boundary
    n_width = 50
    pos = np.array(np.where(img_seg[cen_y-n_width:cen_y+n_width, 0:cen_x]==0))
    left_x = np.amax(pos[1, :])

    pos = np.array(np.where(img_seg[cen_y-n_width:cen_y+n_width, cen_x:]==0))
    right_x = np.amin(pos[1, :]) + cen_x

    pos = np.array(np.where(img_seg[0:cen_y, cen_x-n_width:cen_x+n_width]==0))
    up_y = np.amax(pos[0, :])

    pos = np.array(np.where(img_seg[cen_y:, cen_x-n_width:cen_x+n_width]==0))
    down_y = np.amin(pos[0, :]) + cen_y

    x_width = int(shrink * (right_x - left_x)/2)
    y_width = int(shrink * (down_y - up_y)/2)
    x_cen = int((right_x + left_x)/2)
    y_cen = int((down_y + up_y)/2)

    prColor('auto-crop. center: {} y {} x, width: {} y {} x'.format(y_cen, x_cen, y_width, x_width), 'green')

    return y_cen - y_width, y_cen + y_width, x_cen - x_width, x_cen + x_width