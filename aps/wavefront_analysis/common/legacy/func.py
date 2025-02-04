from calendar import prcal
import numpy as np
import os
import glob
import sys
import h5py
import json
from PIL import Image
from scipy import ndimage as ndi
from scipy import signal

import scipy.ndimage as snd

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

def load_image(file_path):
    if os.path.exists(file_path):
        img = np.array(Image.open(file_path))
    else:
        prColor('Error: wrong data path. No data is loaded.', 'red')
        sys.exit()
    return np.array(img).astype(np.float32)

def img_save(folder_path, filename, img):
    if not os.path.exists(os.path.join(folder_path)): os.makedirs(folder_path)

    im = Image.fromarray(img)
    im.save(os.path.join(folder_path, filename+'.tiff'), save_all=True)
    
    prColor('image saved: {}'.format(filename), 'green')

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
            pos_0 = np.arange(M) - np.round(M/2) + np.round(img_size[0]/2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M/2) + np.round(img_size[1]/2)
            pos_1 = pos_1.astype('int')
            img_data = img[pos_0[0]:pos_0[-1]+1, pos_1[0]:pos_1[-1]+1]
    elif len(img_size) == 3:
        if M > min(img_size[1:]):
            return img
        else:
            pos_0 = np.arange(M) - np.round(M/2) + np.round(img_size[1]/2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M/2) + np.round(img_size[2]/2)
            pos_1 = pos_1.astype('int')
            img_data = np.zeros((img_size[0], M, M))
            for kk, pp in enumerate(img):
                img_data[kk] = pp[pos_0[0]:pos_0[-1]+1, pos_1[0]:pos_1[-1]+1]

    return img_data

def write_h5(result_path, file_name, data_dict):
    ''' this function is used to save the variables in *args to hdf5 file
        args are in format: {'name': data}
    '''
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with h5py.File(os.path.join(result_path, file_name+'.hdf5'), 'w') as f:
        for key_name in data_dict:
            f.create_dataset(key_name, data=data_dict[key_name], compression="gzip", compression_opts=9)
    prColor('result hdf5 file : {} saved'.format(file_name+'.hdf5'), 'green')

def read_h5(file_path, key_name, print_key=False):
    '''
        read the data with the key_name in the h5 file
    '''
    if not os.path.exists(file_path):
        prColor('Wrong file path', 'red')
        sys.exit()

    with h5py.File(file_path, 'r') as f:
        # List all groups
        if print_key:
            prColor("Keys: {}".format(list(f.keys())), 'green')

        # Get the data
        data = f[key_name][:]
    return data

def subpixel_shift(obj, Sprow, Spcol, method='FFT'):
    """
        sub pixel shift of the image
    Args:
        obj ([type]): input array
        Sprow ([type]): shifted pixel along row axis
        Spcol ([type]): shifted pixel along col axis
        method (int, optional): use FFT or interpolation

    Returns:
        [type]: [description]
    """
    if method == 'FFT':
                    
        N_row = len(obj)
        N_col = len(obj[0])
        if N_row % 2 != 0 or N_col % 2 != 0:
            print('The size of data is better to be 2^N for higher accuracy!')

        dk_row = 1/N_row
        dk_col = 1/N_col

        k_row = dk_row * np.fft.ifftshift(np.arange(-N_row/2, N_row/2) if N_row % 2 == 0 else np.arange(-(N_row-1)/2, (N_row+1)/2))
        k_col = dk_col * np.fft.ifftshift(np.arange(-N_col/2, N_col/2) if N_col % 2 == 0 else np.arange(-(N_col-1)/2, (N_col+1)/2))

        KK_row, KK_col = np.meshgrid(k_row, k_col)

        Fr = np.fft.fft2(np.fft.ifftshift(obj))

        Fr = Fr * np.exp(-2j * np.pi * Spcol * KK_col) * np.exp(-2j * np.pi * Sprow * KK_row)
        output = np.fft.fftshift(np.fft.ifft2(Fr))

        return np.real(output)
    
    elif method == 'interp':
        x_axis = range(obj.shape[1])
        y_axis = range(obj.shape[0])
        YY_axis_o, XX_axis_o = np.meshgrid(y_axis, x_axis, indexing='ij')
        YY_axis = YY_axis_o + Sprow
        XX_axis = XX_axis_o + Spcol

        # use ndimage to fit the image
        output = ndi.map_coordinates(obj, [YY_axis, XX_axis], order=2, cval=np.mean(obj))

        return output

    else:
        prColor('wrong method for subpixel shift, should be FFT or interp', 'red')
        sys.exit()

def Data_loader(data_folder,pos_path, ext='tiff', show_detail=True):
    """
        load the scanning data from the data folder
    Args:
        data_folder ([type]): images folder
        pos_path ([type]): file path to the scanning position data
        ext (str, optional): extension of the images. Defaults to 'tiff'.
    """
    if ext == 'tiff' or ext == 'tif':
        img_list = sorted(glob.glob(os.path.join(data_folder, '*.'+ext)))
        img = []
        for im in img_list:
            img.append(load_image(im))
            if show_detail:
                prColor('load image: {}'.format(im), 'cyan')
        img = np.array(img).astype(np.float32)
    else:
        prColor('not support yet for this file', 'red')
        sys.exit()
    
    pos_ext = pos_path.split('.')[-1]
    if pos_ext == 'json':
        if not os.path.exists(pos_path):
            prColor('Wrong file path', 'red')
            sys.exit()
    
        with open(pos_path, 'r') as fp:
            pos = json.load(fp)
            
    return img, pos

def random_phase(n_shape, s_filter):

    phase = ndi.gaussian_filter(
        np.random.normal(loc=0.5, scale=1, size=n_shape), s_filter)
    return (phase - np.amin(phase)) / (np.amax(phase) - np.amin(phase))

def displace_prop(ref, dx, dy, method='no_intensity'):
    """
        propagate the ref image with the phase
    Args:
        ref ([type]): [description]
        dx: displacement dx
        dx: displacement dy
        method (str, optional): intensity or no_intensity. Defaults to 'no_intensity'.

    Returns:
        [type]: [description]
    """
    # lam = .5e-10  # wavelength
    # z = 310e-3      # propagation distance
    # psize = 0.65e-6  # pixel size

    nabla2_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    nablax_kernel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / 2
    nablay_kernel = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2

    # dx = signal.convolve2d(phase * lam * z / (2 * np.pi * psize**2),
    #                        nablax_kernel,
    #                        boundary='symm',
    #                        mode='same')
    # dy = signal.convolve2d(phase * lam * z / (2 * np.pi * psize**2),
    #                        nablay_kernel,
    #                        boundary='symm',
    #                        mode='same')

    # use fitting, for large amount pixel distortion
    x_axis = range(ref.shape[1])
    y_axis = range(ref.shape[0])
    YY_axis_o, XX_axis_o = np.meshgrid(y_axis, x_axis, indexing='ij')
    YY_axis = YY_axis_o - dy
    XX_axis = XX_axis_o - dx

    # use regular fitting
    # f_amp = sfit.RegularGridInterpolator((y_axis, x_axis),
    #                                      ref,
    #                                      bounds_error=False,
    #                                      method='linear',
    #                                      fill_value=np.mean(ref))

    # pts = (np.array([np.ndarray.flatten(YY_axis),
    #                  np.ndarray.flatten(XX_axis)]).transpose())

    # I_re = np.reshape(f_amp(pts), ref.shape)

    #  use griddata fitting

    # pts = (np.array([np.ndarray.flatten(XX_axis_o),
    #                  np.ndarray.flatten(YY_axis_o)]).transpose())
    # I_re = sfit.griddata(pts, np.ndarray.flatten(ref), (XX_axis, YY_axis), method='cubic', fill_value=np.mean(ref))

    # use ndimage to fit the image

    I_re = ndi.map_coordinates(ref, [YY_axis, XX_axis], order=1, cval=np.mean(ref))

    if method == 'intensity':
        I_re = I_re / (1 + signal.convolve2d(dx,
                           nablax_kernel,
                           boundary='symm',
                           mode='same') +
                           signal.convolve2d(dy,
                           nablay_kernel,
                           boundary='symm',
                           mode='same'))

    # pixel distortion method for small amount
    # dI_x, dI_y, dI_xx, dI_yy, dI_xy = partial_dev(ref)

    # I_re = ref / (1+signal.convolve2d(phase*lam*z/(2*np.pi*psize**2), nabla2_kernel, boundary='symm', mode='same')) \
    #         - (dI_x*dx + dI_y*dy + 1/2*(dx**2 * dI_xx  + dy**2 * dI_yy + 2*dx*dy * dI_xy))

    # I_re = np.fmax(np.fmin(I_re, 65535), 0)
    return I_re

def slop_tracking(img, ref, n_window=10):
    '''
        use opencv optical flow function to calculate the moving of the pixels.
        input:
            img:            the sample image
            ref:            the reference image
            p_x:            the pixel size
            z:              the distance
        output:
            phase:          the phase of the wavefront (in meter unit, need to devide the k wavenumber)
            displace:       the displacement of the pixels in the images
                            [dips_H, disp_V]
    '''
    import cv2 # here to avoid conflict with PyQt5

    # the pyramid scale, make the undersampling image
    pyramid_scal = 0.5
    # the pyramid levels
    levels = 4
    # window size of the displacement calculation
    winsize = n_window
    # iteration for the calculation
    n_iter = 10
    # neighborhood pixel size to calculate the polynomial expansion, which makes the results smooth but blurred
    n_poly = 5
    # standard deviation of the Gaussian that is used to smooth derivatives used as a basis for 
    # the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
    sigma_poly = 1.1
    '''
        operation flags that can be a combination of the following:

        OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
        OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian \texttt{winsize}\times\texttt{winsize} filter instead of 
        a box filter of the same size for optical flow estimation; usually, this option gives z more accurate 
        flow than with a box filter, at the cost of lower speed; normally, winsize for a Gaussian window 
        should be set to a larger value to achieve the same level of robustness.
    '''
    flags = 1

    flow = cv2.calcOpticalFlowFarneback(ref,img,None, pyramid_scal, levels, winsize, n_iter, n_poly, sigma_poly, flags)

    return flow[...,0], flow[...,1]

def write_json(result_path, file_name, data_dict):

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_name_para = os.path.join(result_path, file_name+'.json')
    with open(file_name_para, 'w') as fp:
        json.dump(data_dict, fp, indent=4)
    
    prColor('result json file : {} saved'.format(file_name+'.json'), 'green')

def read_json(filepath, print_para=False):

    if not os.path.exists(filepath):
        prColor('Wrong file path: {}'.format(filepath), 'red')
        sys.exit()
    # file_name_para = os.path.join(result_path, file_name+'.json')
    with open(filepath, 'r') as fp:
        data = json.load(fp)
        if print_para:
            prColor('parameters: {}'.format(data), 'green')
    
    return data

def save_img(img, filename):
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
    # from skimage.feature import register_translation
    # from skimage.feature.register_translation import _upsampled_dft
    from skimage.registration import phase_cross_correlation
    # from skimage.registration._phase_cross_correlation import _upsampled_dft
    from scipy.ndimage import fourier_shift
    # roi = lambda x: x[0:100][0:100]
    # shift, error, diffphase = register_translation(image, offset_image, 10)
    shift, error, diffphase = phase_cross_correlation(image, offset_image, upsample_factor=10)
    # shift, error, diffphase = register_translation(roi(image), roi(offset_image), 100)

    print('shift dist: {}, alignment error: {} and phase difference: {}'.format(shift, error, diffphase))
    # image_back = image_shift(offset_image, shift[0], shift[1])
    image_back = fourier_shift(np.fft.fftn(offset_image), shift)
    image_back = np.real(np.fft.ifftn(image_back))
    return shift, image_back

def split_image(img, N, overlap_percent=0):
    """
    split image into multiple patch, with overlapping

    Args:
        img (_type_): 2D array
        overlap_percent (float, optional): overlapping percentage. Defaults to 0.
        N (int):    number of patches
    """
    img_row, img_col = img.shape
    print(img_row, img_col)
    N_extend_r = int(img_row/N[0]*overlap_percent)
    N_extend_c = int(img_col/N[1]*overlap_percent)

    pos_row = np.array_split(np.arange(img_row), N[0])
    pos_col = np.array_split(np.arange(img_col), N[1])

    p_r_list = []
    p_c_list = []
    patches = []
    for p_r in pos_row:
        for p_c in pos_col:
            pr = np.arange(max(p_r[0] - N_extend_r, 0), min(p_r[-1] + 1 + N_extend_r, img_row))
            pc = np.arange(max(p_c[0] - N_extend_c, 0), min(p_c[-1] + 1 + N_extend_c, img_col))

            p_r_list.append(pr)
            p_c_list.append(pc)
            patches.append(img[pr[0]:pr[-1]+1, pc[0]:pc[-1]+1])
    
    return [img_row, img_col], p_r_list, p_c_list, patches

def combine_patches(patches, p_r_list, p_c_list, raw_size):
    """
    combine_patches: combine the splitted patches into one image

    Args:
        patches (list of 2D ndarray): splitted patches
        p_r_list (list of array): row index of each patch
        p_c_list (list of array): col index of each patch
        raw_size (list): raw image size, [row, col]
    """
    img = np.zeros(raw_size)
    N_accum = np.zeros(raw_size)

    for patch, p_r, p_c in zip(patches, p_r_list, p_c_list):
        img[p_r[0]:p_r[-1]+1, p_c[0]:p_c[-1]+1] += patch
        N_accum[p_r[0]:p_r[-1]+1, p_c[0]:p_c[-1]+1] += 1
    
    return img / N_accum

def binning2(img):
    return (img[::2, ::2] + img[1::2, 1::2])/2

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    """

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        prColor('Wrong choice. Should be flat, hanning, hamming, bartlett, blackman', 'red')
        sys.exit()

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def auto_crop(img, shrink=0.9, count=None):
    # auto-crop to find the rectangular area from the image intensity
    if count is None:
        img_seg = np.ones(img.shape) * (img > np.mean(img))
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

def crop_or_pad_to_shape(arr, target_shape):
    """
    Crop or pad a 1D or 2D array to match the target shape using edge values for padding.

    Parameters:
    arr (numpy array): The input array to be cropped or padded.
    target_shape (tuple or int): The target shape (length for 1D or (rows, columns) for 2D).

    Returns:
    numpy array: The cropped or padded array.
    """
    arr_shape = arr.shape

    if isinstance(target_shape, int):
        target_shape = (target_shape,)
    if len(target_shape) != len(arr_shape):
        raise ValueError("Input array and target shape must have the same number of dimensions.")

    crop_or_pad_width = []
    for dim_size, target_dim_size in zip(arr_shape, target_shape):
        crop_size = max(0, dim_size - target_dim_size)
        pad_size = max(0, target_dim_size - dim_size)

        start_crop = crop_size // 2
        end_crop = start_crop + target_dim_size

        pad_width = (pad_size // 2, pad_size - pad_size // 2)
        crop_or_pad_width.append((start_crop, end_crop, pad_width))

    if arr.ndim == 1:
        start_crop, end_crop, pad_width = crop_or_pad_width[0]
        cropped_arr = arr[start_crop:end_crop]
        padded_arr = np.pad(cropped_arr, pad_width, mode='edge')
    elif arr.ndim == 2:
        (start_crop_row, end_crop_row, pad_width_row), (start_crop_col, end_crop_col, pad_width_col) = crop_or_pad_width
        cropped_arr = arr[start_crop_row:end_crop_row, start_crop_col:end_crop_col]
        padded_arr = np.pad(cropped_arr, (pad_width_row, pad_width_col), mode='edge')

    return padded_arr