#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2024, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2024. UChicago Argonne, LLC. This software was produced       #
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
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import json

from skimage.restoration import unwrap_phase
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofryimpl.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D
from wofryimpl.propagator.propagators1D.fresnel_zoom import FresnelZoom1D


from aps.wavefront_analysis.common.arguments import Args

def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def fit_gaussian_and_find_fwhm(x, y):
    p0 = [max(y), np.mean(x), np.std(x)]  # Initial guess: amplitude, mean, standard deviation
    try:
        popt, _ = curve_fit(gaussian, x, y, p0=p0)
        A, x0, sigma = popt
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        return True, sigma, fwhm
    except RuntimeError:
        print("Fit failed.")
        return False, None, None


def find_fwhm(x, y):
    """
    Find the FWHM directly from the y values by identifying the points where
    the intensity drops to half its maximum value.

    Parameters:
    - x: 1D array of x values.
    - y: 1D array of y values corresponding to the intensities.

    Returns:
    - fwhm: The Full Width at Half Maximum.
    """
    half_max = np.max(y) / 2.0
    # Find where the data crosses the half maximum
    cross_half_max_indices = np.where(np.diff(y > half_max))[0]

    if len(cross_half_max_indices) >= 2:
        # Assuming the curve is unimodal and the first and last crossings are the FWHM
        fwhm_x_values = x[cross_half_max_indices[0]], x[cross_half_max_indices[-1]]
        fwhm = fwhm_x_values[1] - fwhm_x_values[0]
        return True, fwhm
    else:
        print("FWHM calculation failed.")
        return False, None


def find_rms(x, intensity, x_range=None):
    if x_range is None or x_range[0] >= x_range[1]:
        x_min, x_max = x.min(), x.max()
    else:
        x_min, x_max = x_range

    # Filter the data within the specified range
    mask = (x >= x_min) & (x <= x_max)
    x_filtered = x[mask]
    intensity_filtered = intensity[mask]

    # Calculate the weighted mean and weighted mean of squares
    mean_x = np.average(x_filtered, weights=intensity_filtered)
    mean_x2 = np.average(x_filtered ** 2, weights=intensity_filtered)
    rms_size = np.sqrt(mean_x2 - mean_x ** 2)

    return True, rms_size


def load_datasets1D(file_path, name_int_x, name_int_y, name_phase_x, name_phase_y):
    with h5py.File(file_path, 'r') as file:
        int_x = np.array(file[name_int_x])
        int_y = np.array(file[name_int_y])
        phase_x = np.array(file[name_phase_x])
        phase_y = np.array(file[name_phase_y])
    return int_x, int_y, phase_x, phase_y


def load_datasets(file_path, dataset_name_int, dataset_name_phase):
    """
    Load specified datasets from an HDF5 file.

    Parameters:
    - file_path: Path to the HDF5 file.
    - dataset_name_int: Name of the dataset for the intensity.
    - dataset_name_phase: Name of the dataset for the phase.

    Returns:
    - A tuple containing two numpy arrays: (A, phase).
    """
    with h5py.File(file_path, 'r') as file:
        intensity = np.array(file[dataset_name_int])  # Loading the specified 'A' dataset
        phase = np.array(file[dataset_name_phase])  # Loading the specified 'phase' dataset
    return intensity, phase


def load_parameters(json_file_path):
    """Load simulation parameters from a JSON file."""
    with open(json_file_path, 'r') as file:
        parameters = json.load(file)
    return parameters


def interpolate_wavefront(wavefront, new_shape):
    """
    Interpolates the wavefront to a new shape using RegularGridInterpolator.

    Parameters:
    - wavefront: 2D numpy array representing the complex wavefront.
    - new_shape: Tuple (new_height, new_width) representing the desired shape.

    Returns:
    - Interpolated wavefront as a 2D numpy array.
    """
    original_shape = wavefront.shape
    # Original grid
    y = np.linspace(0, original_shape[0] - 1, original_shape[0])
    x = np.linspace(0, original_shape[1] - 1, original_shape[1])
    # New grid
    y_new = np.linspace(0, original_shape[0] - 1, new_shape[0])
    x_new = np.linspace(0, original_shape[1] - 1, new_shape[1])
    X_new, Y_new = np.meshgrid(x_new, y_new)

    # Interpolators for the real and imaginary parts
    real_interpolator = RegularGridInterpolator((y, x), wavefront.real)
    imag_interpolator = RegularGridInterpolator((y, x), wavefront.imag)

    # Perform the interpolation
    interpolated_wavefront_real = real_interpolator(np.array([Y_new.ravel(), X_new.ravel()]).T)
    interpolated_wavefront_imag = imag_interpolator(np.array([Y_new.ravel(), X_new.ravel()]).T)

    # Reshape back to the new grid shape and combine the real and imaginary parts
    interpolated_wavefront = interpolated_wavefront_real.reshape(new_shape) + 1j * interpolated_wavefront_imag.reshape(new_shape)

    return interpolated_wavefront


def interpolate_wavefront_with_spline(wavefront, new_shape):
    """
    Interpolates the wavefront to a new shape using RectBivariateSpline
    for bicubic spline interpolation.

    Parameters:
    - wavefront: 2D numpy array representing the complex wavefront.
    - new_shape: Tuple (new_height, new_width) representing the desired shape.

    Returns:
    - Interpolated wavefront as a 2D numpy array.
    """
    original_shape = wavefront.shape
    # Original grid coordinates
    y = np.linspace(0, original_shape[0] - 1, original_shape[0])
    x = np.linspace(0, original_shape[1] - 1, original_shape[1])
    # New grid coordinates
    y_new = np.linspace(0, original_shape[0] - 1, new_shape[0])
    x_new = np.linspace(0, original_shape[1] - 1, new_shape[1])

    # Separate the real and imaginary parts for spline interpolation
    wavefront_real = wavefront.real
    wavefront_imag = wavefront.imag

    # Bicubic spline interpolation for both real and imaginary parts
    spline_real = RectBivariateSpline(y, x, wavefront_real)
    spline_imag = RectBivariateSpline(y, x, wavefront_imag)

    interpolated_wavefront_real = spline_real(y_new, x_new)
    interpolated_wavefront_imag = spline_imag(y_new, x_new)

    # Combine the real and imaginary parts back into a complex array
    interpolated_wavefront = interpolated_wavefront_real + 1j * interpolated_wavefront_imag

    return interpolated_wavefront


def fraunhofer_propagation(wavefront, distance, wavelength, pixel_size):
    """
    Propagates a wavefront using the Fraunhofer far-field approximation.

    Parameters:
    - wavefront: numpy array representing the complex wavefront.
    - distance: Distance to propagate (in meters).
    - wavelength: Wavelength of the wave (in meters).
    - pixel_size: Size of each pixel in the array (in meters).

    Returns:
    - The propagated wavefront as a numpy array.
    """
    # Number of pixels in each dimension
    N_x, N_y = wavefront.shape

    # Spatial frequency domain sampling intervals
    dx = dy = pixel_size
    L_x = N_x * dx
    L_y = N_y * dy

    # Wavenumber
    k = 2 * np.pi / wavelength

    # Spatial frequencies
    fx = np.fft.fftfreq(N_x, d=dx)
    fy = np.fft.fftfreq(N_y, d=dy)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    # Scale factor accounts for propagation distance and wavelength
    scale_factor = np.exp(1j * k * distance) / (1j * wavelength * distance)

    # Fraunhofer diffraction pattern calculation using FFT
    fraunhofer_diffraction_pattern = scale_factor * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wavefront))) * dx * dy

    # The Fraunhofer pattern is typically observed on a plane perpendicular to the direction of propagation,
    # at a distance where the far-field approximation is valid. The field distribution on this plane
    # is proportional to the Fourier transform of the wavefront exiting the aperture.

    return fraunhofer_diffraction_pattern


def fresnel_propagation(wavefront, distance, wavelength, pixel_size):
    """
    Propagates a wavefront using the near-field Fresnel approximation.

    Parameters:
    - wavefront: numpy array representing the complex wavefront.
    - distance: Distance to propagate (in meters).
    - wavelength: Wavelength of the wave (in meters).
    - pixel_size: Size of each pixel in the array (in meters).

    Returns:
    - The propagated wavefront as a numpy array.
    """
    # Number of pixels in each dimension
    N_x, N_y = wavefront.shape

    # Create spatial frequency grids
    dx = pixel_size
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(N_x, d=dx)
    fy = np.fft.fftfreq(N_y, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    # Quadratic phase factor for Fresnel propagation in the frequency domain
    H = np.exp(-1j * k * distance) * np.exp(-1j * (np.pi * wavelength * distance) * (FX ** 2 + FY ** 2))

    # Apply Fourier transform to the wavefront, multiply by the propagation kernel, and inverse Fourier transform
    propagated_wavefront = np.fft.ifft2(np.fft.fft2(wavefront) * H)

    return propagated_wavefront


def fresnel_propagation_with_padding(wavefront, distance, wavelength, pixel_size, pad_size):
    """
    Propagates a wavefront using the near-field Fresnel approximation, with padding,
    utilizing numpy for FFT operations.

    Parameters:
    - wavefront: numpy array representing the complex wavefront.
    - distance: Distance to propagate (in meters).
    - wavelength: Wavelength of the wave (in meters).
    - pixel_size: Size of each pixel in the array (in meters).
    - pad_size: Number of pixels to pad around the original wavefront.

    Returns:
    - The propagated and cropped wavefront as a numpy array.
    """
    # Pad the wavefront array
    padded_wavefront = np.pad(wavefront, pad_width=pad_size, mode='constant', constant_values=0)

    # Perform Fresnel propagation on the padded wavefront
    N_x, N_y = padded_wavefront.shape
    k = 2 * np.pi / wavelength
    dx = pixel_size
    fx = np.fft.fftfreq(N_x, d=dx)
    fy = np.fft.fftfreq(N_y, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    # Calculate the propagation kernel
    H = np.exp(-1j * k * distance) * np.exp(-1j * (np.pi * wavelength * distance) * (FX ** 2 + FY ** 2))

    # Apply the Fourier transform, multiply by the propagation kernel, and then apply the inverse Fourier transform
    propagated_padded_wavefront = np.fft.ifft2(np.fft.fft2(padded_wavefront) * H)

    # Crop the wavefront back to the original size after propagation
    start_x, start_y = pad_size, pad_size
    end_x, end_y = start_x + wavefront.shape[0], start_y + wavefront.shape[1]
    cropped_wavefront = propagated_padded_wavefront[start_x:end_x, start_y:end_y]

    return cropped_wavefront


def fresnel_propagation_with_fourier_padding(wavefront, distance, wavelength, pixel_size, pad_size):
    """
    Propagates a wavefront using the near-field Fresnel approximation,
    with padding applied in the reciprocal (Fourier) space to improve spatial resolution.

    Parameters:
    - wavefront: numpy array representing the complex wavefront.
    - distance: Distance to propagate (in meters).
    - wavelength: Wavelength of the wave (in meters).
    - pixel_size: Size of each pixel in the array (in meters).
    - pad_size: Number of pixels to add as padding in the Fourier domain.

    Returns:
    - The propagated wavefront as a numpy array with improved spatial resolution.
    """
    N_x, N_y = wavefront.shape
    k = 2 * np.pi / wavelength

    # Step 1: Perform the initial FFT
    fft_wavefront = np.fft.fft2(wavefront)

    # Step 2: Pad the Fourier-transformed wavefront
    padded_fft_wavefront = np.pad(fft_wavefront, pad_size, mode='constant', constant_values=0)

    # Recalculate N_x, N_y after padding for the propagation kernel calculation
    N_x_padded, N_y_padded = padded_fft_wavefront.shape

    dx = pixel_size
    # Adjust spatial frequencies for padded size
    fx = np.fft.fftfreq(N_x_padded, d=dx)
    fy = np.fft.fftfreq(N_y_padded, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    # Step 3: Apply the propagation kernel in Fourier domain
    H = np.exp(-1j * k * distance) * np.exp(-1j * (np.pi * wavelength * distance) * (FX ** 2 + FY ** 2))
    propagated_padded_fft_wavefront = padded_fft_wavefront * H

    # Step 4: Perform the inverse FFT
    propagated_wavefront = np.fft.ifft2(propagated_padded_fft_wavefront)

    return propagated_wavefront


def crop_center(wavefront, num_points):
    """
    Crop the center part of the wavefront array to a specified number of points.

    Parameters:
    - wavefront: 2D numpy array of the complex wavefront.
    - num_points: The size of the square side to be cropped from the center.

    Returns:
    - Cropped 2D numpy array of the complex wavefront.
    """
    # Calculate the center index of the array
    center_y, center_x = np.array(wavefront.shape) // 2

    # Calculate the start and end indices of the crop
    start_x = max(center_x - num_points // 2, 0)
    end_x = start_x + num_points
    start_y = max(center_y - num_points // 2, 0)
    end_y = start_y + num_points

    # Crop the wavefront
    cropped_wavefront = wavefront[start_y:end_y, start_x:end_x]

    return cropped_wavefront


def energy_to_wavelength(energy_eV):
    """Convert energy in eV to wavelength in meters."""
    h = 6.62607015e-34  # Planck's constant, m^2 kg / s
    c = 3.0e8  # Speed of light, m / s
    e = 1.602176634e-19  # Elementary charge, C
    energy_joules = energy_eV * e  # Convert energy to joules
    wavelength = h * c / energy_joules
    return wavelength


def unwrap_phase_image(phase_image):
    """
    Unwraps a 2D phase image to remove discontinuities.

    Parameters:
    - phase_image: A 2D numpy array containing the wrapped phase values.

    Returns:
    - A 2D numpy array containing the unwrapped phase values.
    """
    unwrapped_phase = unwrap_phase(phase_image)
    return unwrapped_phase


def decompose_and_propagate_wavefront(wavefront, distance, wavelength, pixel_size, Rx, Ry):
    """
    Decomposes the wavefront, propagates the residual component, and recombines, ensuring amplitude is preserved.
    """
    # Grid setup
    N_x, N_y = wavefront.shape
    x = np.linspace(-N_x / 2, N_x / 2, N_x) * pixel_size
    y = np.linspace(-N_y / 2, N_y / 2, N_y) * pixel_size
    X, Y = np.meshgrid(x, y)

    # Quadratic phase term representing the lens effect
    phi_Q = (np.pi / wavelength) * ((X ** 2 / Rx) + (Y ** 2 / Ry))
    lens_effect = np.exp(1j * phi_Q)

    # Remove the lens effect from the wavefront
    residual_wavefront = wavefront * np.exp(-1j * phi_Q)

    # Propagate the residual wavefront
    propagated_residual_wavefront = fresnel_propagation(residual_wavefront, distance, wavelength, pixel_size)

    # Recombine the propagated wavefront with the lens effect
    propagated_wavefront = propagated_residual_wavefront * lens_effect

    return propagated_wavefront


def gaussian_amplitude(N, width, pixel_size):
    """Generate a Gaussian amplitude distribution."""
    x = np.linspace(-N / 2, N / 2, N) * pixel_size
    y = np.linspace(-N / 2, N / 2, N) * pixel_size
    X, Y = np.meshgrid(x, y)
    return np.exp(-(X ** 2 + Y ** 2) / (2 * width ** 2))

class PropagatedWavefront:
    def __init__(self,
                 kind,
                 fwhm_x,
                 fwhm_y,
                 sigma_x,
                 sigma_y,
                 fwhm_x_gauss,
                 fwhm_y_gauss,
                 propagation_distance,
                 propagation_distance_x,
                 propagation_distance_y,
                 focus_z_position_x,
                 focus_z_position_y,
                 x_coordinates,
                 y_coordinates,
                 intensity,
                 intensity_x,
                 intensity_y,
                 integrated_intensity_x,
                 integrated_intensity_y):
            self.kind                    = kind
            self.fwhm_x                  = fwhm_x
            self.fwhm_y                  = fwhm_y
            self.sigma_x                 = sigma_x
            self.sigma_y                 = sigma_y
            self.fwhm_x_gauss            = fwhm_x_gauss
            self.fwhm_y_gauss            = fwhm_y_gauss
            self.propagation_distance    = propagation_distance
            self.propagation_distance_x  = propagation_distance_x
            self.propagation_distance_y  = propagation_distance_y
            self.focus_z_position_x      = focus_z_position_x
            self.focus_z_position_y      = focus_z_position_y
            self.x_coordinates           = x_coordinates
            self.y_coordinates           = y_coordinates
            self.intensity               = intensity
            self.intensity_x             = intensity_x
            self.intensity_y             = intensity_y
            self.integrated_intensity_x  = integrated_intensity_x
            self.integrated_intensity_y  = integrated_intensity_y

    def to_hdf5(self, file_path_results):
        with h5py.File(file_path_results, 'w') as h5file:
            wf = h5file.create_group("propagated_wavefront")

            wf.attrs["kind"]                 = self.kind
            wf.attrs["fwhm_x"]               = self.fwhm_x
            wf.attrs["fwhm_y"]               = self.fwhm_y
            wf.attrs["sigma_x"]              = self.sigma_x
            wf.attrs["sigma_y"]              = self.sigma_y
            wf.attrs["fwhm_x_gauss"]         = self.fwhm_x_gauss
            wf.attrs["fwhm_y_gauss"]         = self.fwhm_y_gauss
            wf.attrs["focus_z_position_x"]   = self.focus_z_position_x
            wf.attrs["focus_z_position_y"]   = self.focus_z_position_y

            wf.create_dataset('x_coordinates', data=self.x_coordinates)
            wf.create_dataset('y_coordinates', data=self.y_coordinates)

            if self.kind == "2D":
                wf.attrs["propagation_distance"] = self.propagation_distance

                wf.create_dataset('intensity',              data=self.intensity)
                wf.create_dataset('integrated_intensity_x', data=self.integrated_intensity_x)
                wf.create_dataset('integrated_intensity_y', data=self.integrated_intensity_y)
            elif self.kind == "1D":
                wf.attrs["propagation_distance_x"] = self.propagation_distance_x
                wf.attrs["propagation_distance_y"] = self.propagation_distance_y

                wf.create_dataset('intensity_x', data=self.intensity_x)
                wf.create_dataset('intensity_y', data=self.intensity_y)

    def to_dict(self):
        out = {}

        out["kind"]                   = self.kind
        out["fwhm_x"]                 = self.fwhm_x
        out["fwhm_y"]                 = self.fwhm_y
        out["sigma_x"]                = self.sigma_x
        out["sigma_y"]                = self.sigma_y
        out["fwhm_x_gauss"]           = self.fwhm_x_gauss
        out["fwhm_y_gauss"]           = self.fwhm_y_gauss
        out["focus_z_position_x"]     = self.focus_z_position_x
        out["focus_z_position_y"]     = self.focus_z_position_y
        out["coordinates_x"]          = self.x_coordinates
        out["coordinates_y"]          = self.y_coordinates

        if self.kind == "2D":
            out["propagation_distance"]   = self.propagation_distance
            out["intensity"]              = self.intensity
            out["integrated_intensity_x"] = self.integrated_intensity_x
            out["integrated_intensity_y"] = self.integrated_intensity_y
        elif self.kind == "1D":
            out["propagation_distance_x"] = self.propagation_distance_x
            out["propagation_distance_y"] = self.propagation_distance_y
            out["intensity_x"]            = self.intensity_x
            out["intensity_y"]            = self.intensity_y

        return out

def execute_back_propagation(**arguments) -> dict:
    arguments["folder"]            = arguments.get("folder", os.path.abspath(os.curdir))
    arguments["kind"]              = arguments.get("kind", "1D")
    arguments["distance"]          = arguments.get("distance", None)
    arguments["distance_x"]        = arguments.get("distance_x", None)
    arguments["distance_y"]        = arguments.get("distance_y", None)
    arguments["dim_x"]             = arguments.get("dim_x", 500) # crop region
    arguments["dim_y"]             = arguments.get("dim_y", 500) # crop region
    arguments["shift_x"]           = arguments.get("shift_x", 0) # crop central point
    arguments["shift_y"]           = arguments.get("shift_y", 0) # crop central point
    arguments["delta_f_x"]         = arguments.get("delta_f_x", 0.0) # Define the focal length changes in x and y directions (in meters)
    arguments["delta_f_y"]         = arguments.get("delta_f_y", 0.0)
    arguments["x_rms_range"]       = arguments.get("x_rms_range", [-2e-6, 2e-6])
    arguments["y_rms_range"]       = arguments.get("y_rms_range", [-2e-6, 2e-6])
    arguments["magnification_x"]   = arguments.get("magnification_x", 0.028)  # Magnification factor along X
    arguments["magnification_y"]   = arguments.get("magnification_y", 0.028)  # Magnification factor along Y
    arguments["shift_half_pixel"]  = arguments.get("shift_half_pixel", True)  # Whether to shift half a pixel
    arguments["show_figure"]       = arguments.get("show_figure", False)
    arguments["save_result"]       = arguments.get("save_result", False)
    arguments["scan_best_focus"]   = arguments.get("scan_best_focus", False)
    arguments["best_focus_from"]   = arguments.get("best_focus_from", "rms") # rms, fwhm, fwhmG
    arguments["scan_x_rel_range"]  = arguments.get("scan_x_rel_range", [-0.001, 0.001, 0.0001])
    arguments["scan_y_rel_range"]  = arguments.get("scan_y_rel_range", [-0.001, 0.001, 0.0001])
    arguments["verbose"]           = arguments.get("verbose", True)

    args = Args(arguments)

    dim_x            = args.dim_x
    dim_y            = args.dim_y
    shift_x          = args.shift_x
    shift_y          = args.shift_y
    delta_f_x        = args.delta_f_x
    delta_f_y        = args.delta_f_y
    x_rms_range      = args.x_rms_range     
    y_rms_range      = args.y_rms_range     
    magnification_x  = args.magnification_x 
    magnification_y  = args.magnification_y 
    shift_half_pixel = args.shift_half_pixel

    best_focus_from  = args.best_focus_from
    scan_x_rel_range = args.scan_x_rel_range
    scan_y_rel_range = args.scan_y_rel_range

    file_path         = os.path.join(args.folder, 'single_shot_1.hdf5')
    json_setting_path = os.path.join(args.folder, 'setting.json')
    json_result_path  = os.path.join(args.folder, 'result.json')

    # Load parameters
    params = load_parameters(json_setting_path)

    # Convert energy to wavelength
    wavelength = energy_to_wavelength(params['energy'])

    # Use pixel_size from parameters
    pixel_size = params['p_x']

    # Load results
    results = load_parameters(json_result_path)

    R_x = results['avg_source_d_x']
    R_y = results['avg_source_d_y']

    if args.kind.upper() == "2D":
        # Load the datasets
        intensity, phase = load_datasets(file_path, 'intensity', 'phase')
        # This transpose is to convert to my personal preference, x is the first dimension, y is the second dimension, it is against python tradition
        intensity = intensity.T
        intensity = intensity[:, ::-1]
        phase = phase.T
        phase = phase[:, ::-1]
        x_array = np.linspace(-pixel_size * intensity.shape[0] / 2, pixel_size * intensity.shape[0] / 2, intensity.shape[0])
        y_array = np.linspace(-pixel_size * intensity.shape[1] / 2, pixel_size * intensity.shape[1] / 2, intensity.shape[1])
    
        # crop wavefront before propagate
        start_x = max((phase.shape[0] - dim_x) // 2 + shift_x, 0)
        end_x   = min(start_x + dim_x, phase.shape[0])
        start_y = max((phase.shape[1] - dim_y) // 2 + shift_y, 0)
        end_y   = min(start_y + dim_y, phase.shape[1])
    
        intensity = intensity[start_x:end_x, start_y:end_y]
        phase     = phase[start_x:end_x, start_y:end_y]
        x_array   = x_array[start_x:end_x]
        y_array   = y_array[start_y:end_y]
    
        # Calculate the amplitude from the square root of A
        amplitude = np.sqrt(intensity)
    
        # Construct the complex wavefront
        wavefront = amplitude * np.exp(1j * phase)

        propagation_distance = args.distance if not args.distance is None else -(R_x + R_y) / 2  # propagation distance in meters

        # Assuming original wavefront has some curvature:
        # Apply the phase corrections
        if delta_f_x != 0:
            phase_x    = np.exp(1j * np.pi * (x_array ** 2) * delta_f_x / (wavelength * R_x ** 2))
            wavefront *= phase_x[:, np.newaxis]  # Apply phase_x to each column
        if delta_f_y != 0:
            phase_y    = np.exp(1j * np.pi * (y_array ** 2) * delta_f_y / (wavelength * R_y ** 2))
            wavefront *= phase_y[np.newaxis, :]  # Apply phase_y to each row

        initial_wavefront = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=x_array,
                                                                                y_array=y_array,
                                                                                z_array=wavefront,
                                                                                wavelength=wavelength)

        # New pixel sizes after magnification
        new_pixel_size_x = pixel_size * magnification_x
        new_pixel_size_y = pixel_size * magnification_y
        x_coordinates = np.linspace(-new_pixel_size_x * intensity.shape[0] / 2, new_pixel_size_x * intensity.shape[0] / 2, intensity.shape[0])
        y_coordinates = np.linspace(-new_pixel_size_y * intensity.shape[1] / 2, new_pixel_size_y * intensity.shape[1] / 2, intensity.shape[1])
        # Instantiate the propagator
        fresnel_propagator = FresnelZoomXY2D()

        if args.scan_best_focus: print("Scan Best Focus not available in 2D")

        focus_z_position_x = -(propagation_distance-R_x)
        focus_z_position_y = -(propagation_distance-R_y)

        # Perform the propagation
        sigma_x, \
        fwhm_x, \
        fwhm_x_gauss, \
        sigma_y, \
        fwhm_y, \
        fwhm_y_gauss, \
        intensity_wofry, \
        integrated_intensity_x, \
        integrated_intensity_y = __propagate_2D(fresnel_propagator,
                                                initial_wavefront,
                                                magnification_x,
                                                magnification_y,
                                                propagation_distance,
                                                shift_half_pixel,
                                                x_coordinates,
                                                x_rms_range,
                                                y_coordinates,
                                                y_rms_range)

        # note: inf is used for the purpose of best focus scan, while NaN is the failed return value, useful for optimization purposes
        propagated_wavefront = PropagatedWavefront("2D",
                                                   fwhm_x if not np.isinf(fwhm_x) else np.NaN,
                                                   fwhm_y if not np.isinf(fwhm_y) else np.NaN,
                                                   sigma_x if not np.isinf(sigma_x) else np.NaN,
                                                   sigma_y if not np.isinf(sigma_x) else np.NaN,
                                                   fwhm_x_gauss if not np.isinf(fwhm_x_gauss) else np.NaN,
                                                   fwhm_y_gauss if not np.isinf(fwhm_x_gauss) else np.NaN,
                                                   propagation_distance,
                                                   None,
                                                   None,
                                                   focus_z_position_x,
                                                   focus_z_position_y,
                                                   x_coordinates,
                                                   y_coordinates,
                                                   intensity_wofry,
                                                   None,
                                                   None,
                                                   integrated_intensity_x,
                                                   integrated_intensity_y)

        if args.show_figure:
            plt.figure(figsize=(12, 6))
            gamma = 1
            X, Y = np.meshgrid(x_coordinates, y_coordinates)
            plt.subplot(1, 2, 1)
            plt.pcolormesh(X, Y, intensity_wofry.T, shading='auto', norm=PowerNorm(gamma=gamma))
            plt.colorbar(label='Intensity')
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            plt.title(f'Intensity distribution at {propagation_distance} m')
            plt.subplot(1, 2, 2)
            plt.plot(x_coordinates, integrated_intensity_x)
            plt.plot(y_coordinates, integrated_intensity_y)
            plt.xlabel('X or Y (meters)')
            plt.ylabel('Integrated Intensity')
            plt.title('Integrated intensity profile')
            plt.show()

        if args.save_result:
            propagated_wavefront.to_hdf5(os.path.join(args.folder, 'propagated_results.hdf5'))

        return propagated_wavefront.to_dict()
    elif args.kind.upper() == "1D":
        # Load the datasets
        int_x, int_y, phase_x, phase_y = load_datasets1D(file_path, 'int_x', 'int_y', 'line_phase_x', 'line_phase_y')

        x_array = np.linspace(-pixel_size * int_x.shape[0] / 2, pixel_size * int_x.shape[0] / 2, int_x.shape[0])
        y_array = np.linspace(-pixel_size * int_y.shape[0] / 2, pixel_size * int_y.shape[0] / 2, int_y.shape[0])

        # Calculate the start and end indices for x and y, incorporating the shifts
        start_x = max((phase_x.shape[0] - dim_x) // 2 + shift_x, 0)
        end_x = min(start_x + dim_x, phase_x.shape[0])
        start_y = max((phase_y.shape[0] - dim_y) // 2 + shift_y, 0)
        end_y = min(start_y + dim_y, phase_y.shape[0])

        # Crop the phase array with the calculated indices
        int_x   = int_x[start_x:end_x]
        int_y   = int_y[start_y:end_y]
        phase_x = phase_x[start_x:end_x]
        phase_y = phase_y[start_y:end_y]
        x_array = x_array[start_x:end_x]
        y_array = y_array[start_y:end_y]

        from scipy.ndimage import gaussian_filter

        int_x   = gaussian_filter(int_x, 21)
        int_y   = gaussian_filter(int_y, 21)
        phase_x = gaussian_filter(phase_x, 100)
        phase_y = gaussian_filter(phase_y, 100)

        # Construct the complex wavefront
        wavefront_x = np.sqrt(int_x) * np.exp(1j * phase_x)
        wavefront_y = np.sqrt(int_y) * np.exp(1j * phase_y)

        propagation_distance_x = args.distance_x if not args.distance_x is None else -R_x  # propagation distance in meters
        propagation_distance_y = args.distance_y if not args.distance_y is None else -R_y  # propagation distance in meters

        if delta_f_x != 0: wavefront_x *= np.exp(1j * np.pi * (x_array ** 2) * delta_f_x / (wavelength * propagation_distance_x ** 2))
        if delta_f_y != 0: wavefront_y *= np.exp(1j * np.pi * (y_array ** 2) * delta_f_y / (wavelength * propagation_distance_y ** 2))

        initial_wavefront_x = GenericWavefront1D.initialize_wavefront_from_arrays(x_array=x_array, y_array=wavefront_x, wavelength=wavelength)
        initial_wavefront_y = GenericWavefront1D.initialize_wavefront_from_arrays(x_array=y_array, y_array=wavefront_y, wavelength=wavelength)

        # New pixel sizes after magnification
        new_pixel_size_x = pixel_size * magnification_x
        new_pixel_size_y = pixel_size * magnification_y
        x_coordinates = np.linspace(-new_pixel_size_x * int_x.shape[0] / 2, new_pixel_size_x * int_x.shape[0] / 2, int_x.shape[0])
        y_coordinates = np.linspace(-new_pixel_size_y * int_y.shape[0] / 2, new_pixel_size_y * int_y.shape[0] / 2, int_y.shape[0])
        # Instantiate the propagator
        fresnel_propagator = FresnelZoom1D()

        sigma_x, fwhm_x, fwhm_x_gauss, intensity_x_wofry, propagated_wf_x = __propagate_1D(fresnel_propagator,
                                                                                           initial_wavefront_x,
                                                                                           magnification_x,
                                                                                           propagation_distance_x,
                                                                                           x_coordinates,
                                                                                           x_rms_range,
                                                                                           "X")

        sigma_y , fwhm_y, fwhm_y_gauss, intensity_y_wofry, propagated_wf_y = __propagate_1D(fresnel_propagator,
                                                                                            initial_wavefront_y,
                                                                                            magnification_y,
                                                                                            propagation_distance_y,
                                                                                            y_coordinates,
                                                                                            y_rms_range,
                                                                                            "Y")

        if args.scan_best_focus:
            focus_z_position_x = __scan_best_focus(fresnel_propagator,
                                                   initial_wavefront_x,
                                                   magnification_x,
                                                   propagation_distance_x,
                                                   x_coordinates,
                                                   x_rms_range,
                                                   scan_x_rel_range,
                                                   best_focus_from,
                                                   "X",
                                                   args.show_figure)
            focus_z_position_y = __scan_best_focus(fresnel_propagator,
                                                   initial_wavefront_y,
                                                   magnification_y,
                                                   propagation_distance_y,
                                                   y_coordinates,
                                                   y_rms_range,
                                                   scan_y_rel_range,
                                                   best_focus_from,
                                                   "Y",
                                                   args.show_figure)
        else:
            focus_z_position_x = -(propagation_distance_x-R_x)
            focus_z_position_y = -(propagation_distance_y-R_y)

        # note: inf is used for the purpose of best focus scan, while NaN is the failed return value, useful for optimization purposes
        propagated_wavefront = PropagatedWavefront("1D",
                                                   fwhm_x if not np.isinf(fwhm_x) else np.NaN,
                                                   fwhm_y if not np.isinf(fwhm_y) else np.NaN,
                                                   sigma_x if not np.isinf(sigma_x) else np.NaN,
                                                   sigma_y if not np.isinf(sigma_x) else np.NaN,
                                                   fwhm_x_gauss if not np.isinf(fwhm_x_gauss) else np.NaN,
                                                   fwhm_y_gauss if not np.isinf(fwhm_x_gauss) else np.NaN,
                                                   None,
                                                   propagation_distance_x,
                                                   propagation_distance_y,
                                                   focus_z_position_x,
                                                   focus_z_position_y,
                                                   x_coordinates,
                                                   y_coordinates,
                                                   None,
                                                   intensity_x_wofry,
                                                   intensity_y_wofry,
                                                   None,
                                                   None)

        if args.show_figure:
            _, (axs) = plt.subplots(2, 2)

            ax1 = axs[0, 0]
            ax2 = axs[0, 1]
            ax3 = axs[1, 0]
            ax4 = axs[1, 1]

            ax1.plot(initial_wavefront_x.get_abscissas(), initial_wavefront_x.get_phase(unwrap=True))
            ax2.plot(initial_wavefront_y.get_abscissas(), initial_wavefront_y.get_phase(unwrap=True))
            ax1.set_xlabel('X (meters)')
            ax2.set_xlabel('Y (meters)')
            ax1.set_ylabel('Phase')

            ax3.plot(propagated_wf_x.get_abscissas(), propagated_wf_x.get_phase(unwrap=True))
            ax4.plot(propagated_wf_y.get_abscissas(), propagated_wf_y.get_phase(unwrap=True))
            ax3.set_xlabel('X (meters)')
            ax4.set_xlabel('Y (meters)')
            ax3.set_ylabel('Phase')

            plt.title(f'Phase profile at {propagation_distance_x}x{propagation_distance_y} distance')
            plt.show()

            _, (axs) = plt.subplots(2, 2)

            ax1 = axs[0, 0]
            ax2 = axs[0, 1]
            ax3 = axs[1, 0]
            ax4 = axs[1, 1]

            ax1.plot(initial_wavefront_x.get_abscissas(), initial_wavefront_x.get_intensity())
            ax2.plot(initial_wavefront_y.get_abscissas(), initial_wavefront_y.get_intensity())
            ax1.set_xlabel('X (meters)')
            ax2.set_xlabel('Y (meters)')
            ax1.set_ylabel('Integrated Intensity')

            ax3.plot(propagated_wf_x.get_abscissas(), propagated_wf_x.get_intensity())
            ax4.plot(propagated_wf_y.get_abscissas(), propagated_wf_y.get_intensity())
            ax3.set_xlabel('X (meters)')
            ax4.set_xlabel('Y (meters)')
            ax3.set_ylabel('Integrated Intensity')

            plt.title(f'Intensity profile at {propagation_distance_x}x{propagation_distance_y} distance')
            plt.show()

        if args.save_result:
            propagated_wavefront.to_hdf5(os.path.join(args.folder, 'propagated_results.hdf5'))

        return propagated_wavefront.to_dict()
    else:
        raise ValueError(f"Propagation kind not recognized: {args.kind}")


def __scan_best_focus(fresnel_propagator,
                      initial_wavefront,
                      magnification,
                      propagation_distance,
                      coordinates,
                      rms_range,
                      scan_rel_range,
                      best_focus_from,
                      direction,
                      show_figure):
    propagation_distances = np.arange(propagation_distance + scan_rel_range[0],
                                      propagation_distance + scan_rel_range[1],
                                      scan_rel_range[2])

    smallest_size  = np.inf
    best_distance  = 0
    best_intensity = None
    size_values    = []

    for distance in propagation_distances:
        sigma, fwhm, fwhm_gauss, intensity_wofry = __propagate_1D(fresnel_propagator,
                                                                  initial_wavefront,
                                                                  magnification,
                                                                  propagation_distance,
                                                                  coordinates,
                                                                  rms_range,
                                                                  direction)
        if   best_focus_from == "rms":   size = sigma
        elif best_focus_from == "fwhm":  size = fwhm
        elif best_focus_from == "fwhmG": size = fwhm_gauss
        else: raise ValueError(f"Best focus from not recognized {best_focus_from}")

        size_values.append(size)

        if size < smallest_size:
            smallest_size  = size
            best_distance  = distance
            best_intensity = intensity_wofry

    print(f"Smallest size in {direction}: {smallest_size} {best_focus_from} at distance {best_distance} m")

    if show_figure:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(propagation_distances, size_values, label=f"Size {direction}", marker='o')
        plt.xlabel('Distance (m)')
        plt.ylabel('Size (units)')
        plt.title('Size as a Function of Distance')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(coordinates, best_intensity)
        plt.xlabel('X (meters)')
        plt.ylabel('Intensity')
        plt.title(f"Intensity profile at {direction} waist")

        plt.tight_layout()  # Adjust spacing between plots
        plt.show()

    return best_distance, smallest_size


def __propagate_1D(fresnel_propagator,
                   initial_wavefront,
                   magnification,
                   propagation_distance,
                   coordinates,
                   rms_range,
                   direction):
    propagated_wavefront = fresnel_propagator.propagate_wavefront(initial_wavefront, propagation_distance, magnification_x=magnification)
    intensity_wofry      = propagated_wavefront.get_intensity()

    # Calculate beam size
    success, fwhm = find_fwhm(coordinates, intensity_wofry)
    if not success: fwhm = np.inf
    success, sigma = find_rms(coordinates, intensity_wofry, rms_range)
    if not success: sigma = np.inf
    success, _, fwhm_gauss = fit_gaussian_and_find_fwhm(coordinates, intensity_wofry)
    if not success: fwhm_gauss = np.inf

    print(f"{direction} direction: sigma = {sigma:.3g}, FWHM = {fwhm:.3g}, Gaussian fit FWHM = {fwhm_gauss:.3g}")

    return sigma, fwhm, fwhm_gauss, intensity_wofry, propagated_wavefront

def __propagate_2D(fresnel_propagator,
                   initial_wavefront,
                   magnification_x,
                   magnification_y,
                   propagation_distance,
                   shift_half_pixel,
                   x_coordinates,
                   x_rms_range,
                   y_coordinates,
                   y_rms_range):
    propagated_wavefront_wofry = fresnel_propagator.propagate_wavefront(initial_wavefront,
                                                                        propagation_distance,
                                                                        magnification_x=magnification_x,
                                                                        magnification_y=magnification_y,
                                                                        shift_half_pixel=shift_half_pixel)
    intensity_wofry = np.abs(propagated_wavefront_wofry.get_complex_amplitude()) ** 2

    integrated_intensity_x = np.sum(intensity_wofry, axis=1)  # Sum over y
    integrated_intensity_y = np.sum(intensity_wofry, axis=0)  # Sum over x

    # Calculate beam size
    success_x, fwhm_x = find_fwhm(x_coordinates, integrated_intensity_x)
    success_y, fwhm_y = find_fwhm(y_coordinates, integrated_intensity_y)
    if not success_x: fwhm_x = np.inf
    if not success_y: fwhm_y = np.inf
    success_x, sigma_x = find_rms(x_coordinates, integrated_intensity_x, x_rms_range)
    success_y, sigma_y = find_rms(y_coordinates, integrated_intensity_y, y_rms_range)
    if not success_x: sigma_x = np.inf
    if not success_y: sigma_y = np.inf
    success_x, _, fwhm_x_gauss = fit_gaussian_and_find_fwhm(x_coordinates, integrated_intensity_x)
    success_y, _, fwhm_y_gauss = fit_gaussian_and_find_fwhm(y_coordinates, integrated_intensity_y)
    if not success_x: fwhm_x_gauss = np.inf
    if not success_y: fwhm_y_gauss = np.inf

    print(f"X direction: sigma = {sigma_x:.3g}, FWHM = {fwhm_x:.3g}, Gaussian fit FWHM = {fwhm_x_gauss:.3g}")
    print(f"Y direction: sigma = {sigma_y:.3g}, FWHM = {fwhm_y:.3g}, Gaussian fit FWHM = {fwhm_y_gauss:.3g}")

    return sigma_x, \
           fwhm_x, \
           fwhm_x_gauss, \
           sigma_y,\
           fwhm_y, \
           fwhm_y_gauss, \
           intensity_wofry, \
           integrated_intensity_x, \
           integrated_intensity_y


