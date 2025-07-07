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

from syned.beamline.beamline_element import BeamlineElement, ElementCoordinates
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import PropagationParameters, PropagationElements
from wofryimpl.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D
from wofryimpl.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen
from wofrysrw.propagator.propagators2D.srw_fresnel_wofry import FresnelSRWWofry
from wofrysrw.propagator.wavefront2D.srw_wavefront import SRWWavefront, WavefrontPropagationParameters
from wofrysrw.beamline.optical_elements.ideal_elements.srw_screen import SRWScreen

from aps.wavefront_analysis.common.arguments import Args
from aps.common.plot.image import rebin_1D, rebin_2D
from aps.common.utilities import energy_to_wavelength

from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline
from scipy.optimize import fminbound

import threading
lock = threading.Lock()

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

        return fwhm
    else:
        print("FWHM calculation failed.")
        return np.inf

def find_rms(x, intensity, x_range=None):
    if x_range is None or x_range[0] >= x_range[1]: x_min, x_max = x.min(), x.max()
    else:                                           x_min, x_max = x_range

    # Filter the data within the specified range
    mask = (x >= x_min) & (x <= x_max)
    x_filtered = x[mask]
    intensity_filtered = intensity[mask]

    # Calculate the weighted mean and weighted mean of squares
    mean_x   = np.average(x_filtered, weights=intensity_filtered)
    mean_x2  = np.average(x_filtered ** 2, weights=intensity_filtered)
    rms_size = np.sqrt(mean_x2 - mean_x ** 2)

    return rms_size

def load_datasets1D(file_path, name_int_x, name_int_y, name_phase_x, name_phase_y):
    with h5py.File(file_path, 'r') as file:
        int_x = np.array(file[name_int_x])
        int_y = np.array(file[name_int_y])
        phase_x = np.array(file[name_phase_x])
        phase_y = np.array(file[name_phase_y])
    return int_x, int_y, phase_x, phase_y

def load_datasets2D(file_path, dataset_name_int, dataset_name_phase):
    with h5py.File(file_path, 'r') as file:
        intensity = np.array(file[dataset_name_int])  # Loading the specified 'A' dataset
        phase     = np.array(file[dataset_name_phase])  # Loading the specified 'phase' dataset

    return intensity, phase

def load_parameters(json_file_path):
    with open(json_file_path, 'r') as file: return json.load(file)

class PropagatedWavefront:
    def __init__(self,
                 kind=None,
                 fwhm_x=None,
                 fwhm_y=None,
                 sigma_x=None,
                 sigma_y=None,
                 propagation_distance=None,
                 propagation_distance_x=None,
                 propagation_distance_y=None,
                 focus_z_position_x=None,
                 focus_z_position_y=None,
                 wf_position_x=None,
                 wf_position_y=None,
                 x_coordinates=None,
                 y_coordinates=None,
                 intensity=None,
                 intensity_x=None,
                 intensity_y=None,
                 integrated_intensity_x=None,
                 integrated_intensity_y=None,
                 scan_best_focus=None,
                 scan_best_focus_from=None,
                 bf_propagation_distance_x=None,
                 bf_propagation_distance_y=None,
                 bf_x_coordinate=None,
                 bf_y_coordinate=None,
                 bf_intensity_x=None,
                 bf_intensity_y=None,
                 bf_integrated_intensity_x=None,
                 bf_integrated_intensity_y=None,
                 bf_size_value_x=None,
                 bf_size_value_y=None,
                 bf_propagation_distances=None,
                 bf_propagation_distances_x=None,
                 bf_propagation_distances_y=None,
                 bf_x_coordinates=None,
                 bf_y_coordinates=None,
                 bf_intensities=None,
                 bf_intensities_x=None,
                 bf_intensities_y=None,
                 bf_integrated_intensities_x=None,
                 bf_integrated_intensities_y=None,
                 bf_size_values_x=None,
                 bf_size_values_y=None,
                 bf_size_values_fit_x=None,
                 bf_size_values_fit_y=None):
            self.kind                    = kind
            self.fwhm_x                  = fwhm_x
            self.fwhm_y                  = fwhm_y
            self.sigma_x                 = sigma_x
            self.sigma_y                 = sigma_y
            self.propagation_distance    = propagation_distance
            self.propagation_distance_x  = propagation_distance_x
            self.propagation_distance_y  = propagation_distance_y
            self.focus_z_position_x      = focus_z_position_x
            self.focus_z_position_y      = focus_z_position_y
            self.wf_position_x           = wf_position_x
            self.wf_position_y           = wf_position_y
            self.x_coordinates           = x_coordinates
            self.y_coordinates           = y_coordinates
            self.intensity               = intensity
            self.intensity_x             = intensity_x
            self.intensity_y             = intensity_y
            self.integrated_intensity_x  = integrated_intensity_x
            self.integrated_intensity_y  = integrated_intensity_y
            
            self.scan_best_focus             = scan_best_focus
            self.scan_best_focus_from        = scan_best_focus_from
            # bf found
            self.bf_propagation_distance_x   = bf_propagation_distance_x
            self.bf_propagation_distance_y   = bf_propagation_distance_y
            self.bf_x_coordinate             = bf_x_coordinate
            self.bf_y_coordinate             = bf_y_coordinate
            self.bf_intensity_x              = bf_intensity_x
            self.bf_intensity_y              = bf_intensity_y
            self.bf_integrated_intensity_x   = bf_integrated_intensity_x
            self.bf_integrated_intensity_y   = bf_integrated_intensity_y
            self.bf_size_value_x             = bf_size_value_x
            self.bf_size_value_y             = bf_size_value_y
            # bf scan details
            self.bf_propagation_distances    = bf_propagation_distances
            self.bf_propagation_distances_x  = bf_propagation_distances_x
            self.bf_propagation_distances_y  = bf_propagation_distances_y
            self.bf_x_coordinates            = bf_x_coordinates
            self.bf_y_coordinates            = bf_y_coordinates
            self.bf_intensities              = bf_intensities
            self.bf_intensities_x            = bf_intensities_x
            self.bf_intensities_y            = bf_intensities_y
            self.bf_integrated_intensities_x = bf_integrated_intensities_x
            self.bf_integrated_intensities_y = bf_integrated_intensities_y
            self.bf_size_values_x            = bf_size_values_x
            self.bf_size_values_y            = bf_size_values_y
            self.bf_size_values_fit_x        = bf_size_values_fit_x
            self.bf_size_values_fit_y        = bf_size_values_fit_y

    def to_hdf5(self, file_path_results):
        with h5py.File(file_path_results, 'w') as h5file:
            wf = h5file.create_group("propagated_wavefront")

            wf.attrs["kind"]                 = self.kind
            wf.attrs["fwhm_x"]               = self.fwhm_x
            wf.attrs["fwhm_y"]               = self.fwhm_y
            wf.attrs["sigma_x"]              = self.sigma_x
            wf.attrs["sigma_y"]              = self.sigma_y
            wf.attrs["focus_z_position_x"]   = self.focus_z_position_x
            wf.attrs["focus_z_position_y"]   = self.focus_z_position_y
            wf.attrs["wf_position_x"]        = self.wf_position_x
            wf.attrs["wf_position_y"]        = self.wf_position_y

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

            wf.attrs["scan_best_focus"] = self.scan_best_focus

            if self.scan_best_focus:
                wf = h5file.create_group("best_focus_scan")

                wf.attrs["scan_best_focus_from"]      = self.scan_best_focus_from
                wf.attrs["bf_propagation_distance_x"] = self.bf_propagation_distance_x
                wf.attrs["bf_propagation_distance_y"] = self.bf_propagation_distance_y
                if self.kind == "2D":
                    wf.create_dataset('bf_x_coordinates',          data=self.bf_x_coordinate)
                    wf.create_dataset('bf_y_coordinates',          data=self.bf_y_coordinate)
                    wf.create_dataset('bf_intensity_x',            data=self.bf_intensity_x)
                    wf.create_dataset('bf_intensity_y',            data=self.bf_intensity_y)
                    wf.create_dataset('bf_integrated_intensity_x', data=self.bf_integrated_intensity_x)
                    wf.create_dataset('bf_integrated_intensity_y', data=self.bf_integrated_intensity_y)
                elif self.kind == "1D":
                    wf.create_dataset('bf_intensity_x', data=self.bf_intensity_x)
                    wf.create_dataset('bf_intensity_y', data=self.bf_intensity_y)
                wf.attrs["bf_size_value_x"]           = self.bf_size_value_x
                wf.attrs["bf_size_value_y"]           = self.bf_size_value_y
                if self.kind == "2D":
                    wf.create_dataset('bf_propagation_distances', data=self.bf_propagation_distances)
                    for i in range(len(self.bf_propagation_distances)):
                        sc = wf.create_group(f"{self.bf_propagation_distances[i]}")
                        sc.create_dataset('x_coordinates',          data=self.bf_x_coordinates[i])
                        sc.create_dataset('y_coordinates',          data=self.bf_y_coordinates[i])
                        sc.create_dataset('intensity',              data=self.bf_intensities[i])
                        sc.create_dataset('integrated_intensity_x', data=self.bf_integrated_intensities_x[i])
                        sc.create_dataset('integrated_intensity_y', data=self.bf_integrated_intensities_y[i])
                elif self.kind == "1D":
                    wf.create_dataset('bf_propagation_distances_x', data=self.bf_propagation_distances_x)
                    wf.create_dataset('bf_propagation_distances_y', data=self.bf_propagation_distances_y)
                    for i in range(len(self.bf_propagation_distances_x)):
                        sc = wf.create_group(f"{self.bf_propagation_distances[i]}_x")
                        sc.create_dataset('intensity_x', data=self.bf_intensities_x[i])
                    for i in range(len(self.bf_propagation_distances_y)):
                        sc = wf.create_group(f"{self.bf_propagation_distances[i]}_y")
                        sc.create_dataset('intensity_y', data=self.bf_intensities_y[i])
                wf.create_dataset('bf_size_values_x',     data=self.bf_size_values_x)
                wf.create_dataset('bf_size_values_y',     data=self.bf_size_values_y)
                if not self.bf_size_values_fit_x is None: wf.create_dataset('bf_size_values_fit_x', data=self.bf_size_values_fit_x)
                if not self.bf_size_values_fit_y is None: wf.create_dataset('bf_size_values_fit_y', data=self.bf_size_values_fit_y)
                
    def to_dict(self):
        out = {}

        out["kind"]                   = self.kind
        out["fwhm_x"]                 = self.fwhm_x
        out["fwhm_y"]                 = self.fwhm_y
        out["sigma_x"]                = self.sigma_x
        out["sigma_y"]                = self.sigma_y
        out["focus_z_position_x"]     = self.focus_z_position_x
        out["focus_z_position_y"]     = self.focus_z_position_y
        out["wf_position_x"]          = self.wf_position_x
        out["wf_position_y"]          = self.wf_position_y
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

        out["scan_best_focus"] = self.scan_best_focus

        if self.scan_best_focus:
            out["scan_best_focus_from"] = self.scan_best_focus_from
            out["bf_propagation_distance_x"] = self.bf_propagation_distance_x
            out["bf_propagation_distance_y"] = self.bf_propagation_distance_y
            out["bf_size_value_x"]          = self.bf_size_value_x
            out["bf_size_value_y"]          = self.bf_size_value_y
            out["bf_size_values_x"]         = self.bf_size_values_x
            out["bf_size_values_y"]         = self.bf_size_values_y
            out["bf_size_values_fit_x"]     = self.bf_size_values_fit_x
            out["bf_size_values_fit_y"]     = self.bf_size_values_fit_y
            if self.kind == "2D":
                out["bf_x_coordinate"]              = self.bf_x_coordinate
                out["bf_y_coordinate"]              = self.bf_y_coordinate
                out["bf_intensity_x"]              = self.bf_intensity_x
                out["bf_intensity_y"]              = self.bf_intensity_y
                out["bf_integrated_intensity_x"]   = self.bf_integrated_intensity_x
                out["bf_integrated_intensity_y"]   = self.bf_integrated_intensity_y
                out["bf_propagation_distances"]    = self.bf_propagation_distances
                out["bf_x_coordinates"]            = self.bf_x_coordinates
                out["bf_y_coordinates"]            = self.bf_y_coordinates
                out["bf_intensities"]              = self.bf_intensities
                out["bf_integrated_intensities_x"] = self.bf_integrated_intensities_x
                out["bf_integrated_intensities_y"] = self.bf_integrated_intensities_y
            elif self.kind == "1D":
                out["bf_x_coordinate"]           = self.bf_x_coordinate
                out["bf_y_coordinate"]           = self.bf_y_coordinate
                out["bf_intensity_x"]             = self.bf_intensity_x
                out["bf_intensity_y"]             = self.bf_intensity_y
                out["bf_propagation_distances_x"] = self.bf_propagation_distances_x
                out["bf_propagation_distances_y"] = self.bf_propagation_distances_y
                out["bf_x_coordinates"]           = self.bf_x_coordinates
                out["bf_y_coordinates"]           = self.bf_y_coordinates
                out["bf_intensities_x"]           = self.bf_intensities_x
                out["bf_intensities_y"]           = self.bf_intensities_y

        return out

def execute_back_propagation(**arguments) -> dict:
    arguments["folder"]                 = arguments.get("folder", os.path.abspath(os.curdir))
    arguments["ref_folder"]             = arguments.get("reference_folder", os.path.join(os.path.abspath(os.curdir), "simulated_mask"))
    arguments["kind"]                   = arguments.get("kind", "1D")
    arguments["mask_detector_distance"] = arguments.get("mask_detector_distance", 0.2)
    arguments["pixel_size"]             = arguments.get("pixel_size", 0.5e-6)
    arguments["image_rebinning"]        = arguments.get("image_rebinning", 1.0)
    arguments["distance"]               = arguments.get("distance", None)
    arguments["distance_x"]             = arguments.get("distance_x", None)
    arguments["distance_y"]             = arguments.get("distance_y", None)
    arguments["dim_x"]                  = arguments.get("dim_x", 500) # crop region
    arguments["dim_y"]                  = arguments.get("dim_y", 500) # crop region
    arguments["shift_x"]                = arguments.get("shift_x", 0) # crop central point
    arguments["shift_y"]                = arguments.get("shift_y", 0) # crop central point
    arguments["delta_f_x"]              = arguments.get("delta_f_x", 0.0) # Define the focal length changes in x and y directions (in meters)
    arguments["delta_f_y"]              = arguments.get("delta_f_y", 0.0)
    arguments["x_rms_range"]            = arguments.get("x_rms_range", [-2e-6, 2e-6])
    arguments["y_rms_range"]            = arguments.get("y_rms_range", [-2e-6, 2e-6])
    arguments["engine"]                 = arguments.get("engine", "srw")

    arguments["magnification_x"]        = arguments.get("magnification_x", 0.028)  # Magnification factor along X
    arguments["magnification_y"]        = arguments.get("magnification_y", 0.028)  # Magnification factor along Y
    arguments["shift_half_pixel"]       = arguments.get("shift_half_pixel", True)  # Whether to shift half a pixel

    arguments["auto_resize_before_propagation"]                         = arguments.get("auto_resize_before_propagation", False)
    arguments["auto_resize_after_propagation"]                          = arguments.get("auto_resize_after_propagation", False)
    arguments["relative_precision_for_propagation_with_autoresizing"]   = arguments.get("relative_precision_for_propagation_with_autoresizing", 1.0)
    arguments["allow_semianalytical_treatment_of_quadratic_phase_term"] = arguments.get("allow_semianalytical_treatment_of_quadratic_phase_term", 1)
    arguments["do_any_resizing_on_fourier_side_using_fft"]              = arguments.get("do_any_resizing_on_fourier_side_using_fft", False)
    arguments["horizontal_range_modification_factor_at_resizing"]       = arguments.get("horizontal_range_modification_factor_at_resizing", 1.0)
    arguments["horizontal_resolution_modification_factor_at_resizing"]  = arguments.get("horizontal_resolution_modification_factor_at_resizing", 1.0)
    arguments["vertical_range_modification_factor_at_resizing"]         = arguments.get("vertical_range_modification_factor_at_resizing", 1.0)
    arguments["vertical_resolution_modification_factor_at_resizing"]    = arguments.get("vertical_resolution_modification_factor_at_resizing", 1.0)

    arguments["show_figure"]            = arguments.get("show_figure", False)
    arguments["save_result"]            = arguments.get("save_result", False)
    arguments["scan_best_focus"]        = arguments.get("scan_best_focus", False)
    arguments["use_fit"]                = arguments.get("use_fit", True)
    arguments["best_focus_from"]        = arguments.get("best_focus_from", "rms") # rms, fwhm, fwhmG
    arguments["scan_rel_range"]         = arguments.get("scan_rel_range", [-0.001, 0.001, 0.0001])
    arguments["scan_x_rel_range"]       = arguments.get("scan_x_rel_range", [-0.001, 0.001, 0.0001])
    arguments["scan_y_rel_range"]       = arguments.get("scan_y_rel_range", [-0.001, 0.001, 0.0001])
    arguments["verbose"]                = arguments.get("verbose", True)
    arguments["rebinning"]              = arguments.get("rebinning", 1)
    arguments["smooth_intensity"]       = arguments.get("smooth_intensity", False)
    arguments["smooth_phase"]           = arguments.get("smooth_phase", False)
    arguments["sigma_intensity"]        = arguments.get("sigma_intensity", 21)
    arguments["sigma_phase"]            = arguments.get("sigma_phase", 21)

    args = Args(arguments)

    dim_x            = args.dim_x
    dim_y            = args.dim_y
    shift_x          = args.shift_x
    shift_y          = args.shift_y
    delta_f_x        = args.delta_f_x
    delta_f_y        = args.delta_f_y
    best_focus_from  = args.best_focus_from

    file_path                = os.path.join(args.folder, 'single_shot_1.hdf5')
    json_setting_path        = os.path.join(args.folder, 'setting.json')
    json_result_path         = os.path.join(args.folder, 'result.json')
    json_reference_path      = os.path.join(args.folder, 'reference.json')
    json_mask_reference_path = os.path.join(args.ref_folder, 'reference.json')

    # Load parameters
    params = load_parameters(json_setting_path)

    # Convert energy to wavelength
    wavelength = energy_to_wavelength(params['energy'])

    # Load results
    results        = load_parameters(json_result_path)
    reference      = load_parameters(json_reference_path)
    reference_mask = load_parameters(json_mask_reference_path)

    R_x          = results['avg_source_d_x']
    R_y          = results['avg_source_d_y']

    ref_speckle_shift  = reference_mask['speckle_shift']
    speckle_shift      = reference['speckle_shift']
    pixel_size         = args.pixel_size*args.image_rebinning

    speckle_shift_x = pixel_size*(speckle_shift[1] - ref_speckle_shift[1])
    speckle_shift_y = pixel_size*(speckle_shift[0] - ref_speckle_shift[0])

    rebin_factor  = args.rebinning

    def calculate_shift(speckle_shift, propagation_distance):
        return -speckle_shift * (abs(propagation_distance) - args.mask_detector_distance) / args.mask_detector_distance

    if args.kind.upper() == "2D":
        # Load the datasets
        intensity, phase = load_datasets2D(file_path, 'intensity', 'phase')
        # This transpose is to convert to my personal preference, x is the first dimension, y is the second dimension, it is against python tradition
        intensity = intensity.T
        intensity = intensity[:, ::-1]
        phase = phase.T
        phase = phase[:, ::-1]

        x_array = np.linspace(-pixel_size * intensity.shape[0] / 2, pixel_size * intensity.shape[0] / 2, intensity.shape[0])
        y_array = np.linspace(-pixel_size * intensity.shape[1] / 2, pixel_size * intensity.shape[1] / 2, intensity.shape[1])

        if rebin_factor > 1:
            x_array, y_array, intensity = rebin_2D(x_array, y_array, intensity, rebin_factor, exact=False)
            _, _,                 phase = rebin_2D(None, None, phase, rebin_factor, exact=False)
            dim_x          = dim_x // rebin_factor
            dim_y          = dim_y // rebin_factor

        if args.smooth_intensity: intensity = gaussian_filter(intensity, args.sigma_intensity)
        if args.smooth_phase:     phase     = gaussian_filter(phase, args.sigma_phase)

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
        wf_position_x = calculate_shift(speckle_shift_x, propagation_distance)
        wf_position_y = calculate_shift(speckle_shift_y, propagation_distance)

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

        if args.engine.lower() == "wofry":
            fresnel_propagator = FresnelZoomXY2D()
        elif args.engine.lower() == "srw":
            initial_wavefront = SRWWavefront.fromGenericWavefront(initial_wavefront,
                                                                  z=propagation_distance,
                                                                  Rx=R_x,
                                                                  Ry=R_y)
            fresnel_propagator = FresnelSRWWofry()


        if args.scan_best_focus:
            best_distance_x, best_x_coordinates, best_intensity_x, best_integrated_intensity_x, smallest_size_x, best_distance_y, best_y_coordinates, best_intensity_y, best_integrated_intensity_y, smallest_size_y, \
            bf_propagation_distances, bf_x_coordinates, bf_y_coordinates, bf_intensities, bf_integrated_intensities_x, bf_size_values_x, bf_size_values_x_fit, bf_integrated_intensities_y, bf_size_values_y, bf_size_values_y_fit = \
                __scan_best_focus_2D(initial_wavefront,
                                     fresnel_propagator,
                                     propagation_distance,
                                     args)
            focus_z_position_x = -(propagation_distance - best_distance_x)
            focus_z_position_y = -(propagation_distance - best_distance_y)
        else:
            focus_z_position_x = -(propagation_distance + R_x)
            focus_z_position_y = -(propagation_distance + R_y)
            best_distance_x = best_x_coordinates = best_intensity_x = best_integrated_intensity_x = smallest_size_x = best_distance_y = best_y_coordinates = best_intensity_y = best_integrated_intensity_y = smallest_size_y = \
            bf_propagation_distances = bf_x_coordinates = bf_y_coordinates = bf_intensities = bf_integrated_intensities_x = bf_size_values_x = bf_size_values_x_fit = bf_integrated_intensities_y = bf_size_values_y = bf_size_values_y_fit = None

        # Perform the propagation
        sigma_x, \
        fwhm_x, \
        sigma_y, \
        fwhm_y, \
        intensity_wofry, \
        integrated_intensity_x, \
        integrated_intensity_y, \
        x_coordinates, \
        y_coordinates = __propagate_2D(initial_wavefront,
                                       fresnel_propagator,
                                       propagation_distance,
                                       args)

        # note: inf is used for the purpose of best focus scan, while NaN is the failed return value, useful for optimization purposes
        if not args.scan_best_focus:
            propagated_wavefront = PropagatedWavefront(kind="2D",
                                                       fwhm_x=fwhm_x if not np.isinf(fwhm_x) else np.nan,
                                                       fwhm_y=fwhm_y if not np.isinf(fwhm_y) else np.nan,
                                                       sigma_x=sigma_x if not np.isinf(sigma_x) else np.nan,
                                                       sigma_y=sigma_y if not np.isinf(sigma_x) else np.nan,
                                                       propagation_distance=propagation_distance,
                                                       focus_z_position_x=focus_z_position_x,
                                                       focus_z_position_y=focus_z_position_y,
                                                       wf_position_x=wf_position_x,
                                                       wf_position_y=wf_position_y,
                                                       x_coordinates=x_coordinates,
                                                       y_coordinates=y_coordinates,
                                                       intensity=intensity_wofry,
                                                       integrated_intensity_x=integrated_intensity_x,
                                                       integrated_intensity_y=integrated_intensity_y,
                                                       scan_best_focus=False)
        else:
            propagated_wavefront = PropagatedWavefront(kind="2D",
                                                       fwhm_x=fwhm_x if not np.isinf(fwhm_x) else np.nan,
                                                       fwhm_y=fwhm_y if not np.isinf(fwhm_y) else np.nan,
                                                       sigma_x=sigma_x if not np.isinf(sigma_x) else np.nan,
                                                       sigma_y=sigma_y if not np.isinf(sigma_x) else np.nan,
                                                       propagation_distance=propagation_distance,
                                                       focus_z_position_x=focus_z_position_x,
                                                       focus_z_position_y=focus_z_position_y,
                                                       wf_position_x=wf_position_x,
                                                       wf_position_y=wf_position_y,
                                                       x_coordinates=x_coordinates,
                                                       y_coordinates=y_coordinates,
                                                       intensity=intensity_wofry,
                                                       integrated_intensity_x=integrated_intensity_x,
                                                       integrated_intensity_y=integrated_intensity_y,
                                                       scan_best_focus=True,
                                                       scan_best_focus_from=best_focus_from,
                                                       bf_propagation_distance_x=best_distance_x,
                                                       bf_propagation_distance_y=best_distance_y,
                                                       bf_x_coordinate=best_x_coordinates,
                                                       bf_y_coordinate=best_y_coordinates,
                                                       bf_intensity_x=best_intensity_x,
                                                       bf_intensity_y=best_intensity_y,
                                                       bf_integrated_intensity_x=best_integrated_intensity_x,
                                                       bf_integrated_intensity_y=best_integrated_intensity_y,
                                                       bf_size_value_x=smallest_size_x,
                                                       bf_size_value_y=smallest_size_y,
                                                       bf_propagation_distances=bf_propagation_distances,
                                                       bf_x_coordinates = bf_x_coordinates,
                                                       bf_y_coordinates = bf_y_coordinates,
                                                       bf_intensities=bf_intensities,
                                                       bf_integrated_intensities_x=bf_integrated_intensities_x,
                                                       bf_integrated_intensities_y=bf_integrated_intensities_y,
                                                       bf_size_values_x=bf_size_values_x,
                                                       bf_size_values_y=bf_size_values_y,
                                                       bf_size_values_fit_x=bf_size_values_x_fit,
                                                       bf_size_values_fit_y=bf_size_values_y_fit)

        if args.show_figure:
            gamma = 1
            intensity   = gaussian_filter(intensity_wofry, 1.5)
            intensity_x = intensity.sum(axis=1)
            intensity_y = intensity.sum(axis=0)

            X, Y = np.meshgrid(1e6*(x_coordinates + wf_position_x),
                               1e6*(y_coordinates + wf_position_y))

            with lock:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.pcolormesh(X, Y, intensity.T, shading='auto', norm=PowerNorm(gamma=gamma), cmap="rainbow")
                plt.colorbar(label='Intensity')
                plt.xlabel(f'X ($\mu$m) / tilt : {round(1e6*wf_position_x, 2)}')
                plt.ylabel(f'Y ($\mu$m) / tilt : {round(1e6*wf_position_y, 2)}')
                plt.xlim(1e6*(wf_position_x - 3*sigma_x), 1e6*(wf_position_x + 3*sigma_x))
                plt.ylim(1e6*(wf_position_y - 3*sigma_y), 1e6*(wf_position_y + 3*sigma_y))

                plt.title(f'Intensity distribution at {propagation_distance} m')
                plt.subplot(1, 2, 2)
                plt.plot(1e6*(x_coordinates + wf_position_x), intensity_x)
                plt.xlim(1e6*(wf_position_x - 3*sigma_x), 1e6*(wf_position_x + 3*sigma_x))
                plt.plot(1e6*(y_coordinates + wf_position_y), intensity_y)
                plt.xlim(1e6*(wf_position_y - 3*sigma_y), 1e6*(wf_position_y + 3*sigma_y))
                plt.xlabel('X or Y ($\mu$m)')
                plt.ylabel('Integrated Intensity')
                plt.title('Integrated intensity profile')

                plt.show()

        if args.save_result: propagated_wavefront.to_hdf5(os.path.join(args.folder, 'propagated_results.hdf5'))

        return propagated_wavefront.to_dict()
    elif args.kind.upper() == "1D":
        # Load the datasets
        int_x, int_y, phase_x, phase_y = load_datasets1D(file_path, 'int_x', 'int_y', 'line_phase_x', 'line_phase_y')

        x_array = np.linspace(-pixel_size * int_x.shape[0] / 2, pixel_size * int_x.shape[0] / 2, int_x.shape[0])
        y_array = np.linspace(-pixel_size * int_y.shape[0] / 2, pixel_size * int_y.shape[0] / 2, int_y.shape[0])

        if rebin_factor > 1:
            x_array, int_x = rebin_1D(x_array, int_x, rebin_factor, exact=False)
            _,     phase_x = rebin_1D(None, phase_x, rebin_factor, exact=False)
            dim_x          = dim_x // rebin_factor

            y_array, int_y = rebin_1D(y_array, int_y, rebin_factor, exact=False)
            _,     phase_y = rebin_1D(None, phase_y, rebin_factor, exact=False)
            dim_y          = dim_y // rebin_factor

        if args.smooth_intensity:
            int_x = gaussian_filter(int_x, args.sigma_intensity)
            int_y = gaussian_filter(int_y, args.sigma_intensity)

        if args.smooth_phase:
            phase_x = gaussian_filter(phase_x, args.sigma_phase)
            phase_y = gaussian_filter(phase_y, args.sigma_phase)

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

        # Construct the complex wavefront
        wavefront_x = np.sqrt(int_x) * np.exp(1j * phase_x)
        wavefront_y = np.sqrt(int_y) * np.exp(1j * phase_y)

        propagation_distance_x = args.distance_x if not args.distance_x is None else -R_x  # propagation distance in meters
        propagation_distance_y = args.distance_y if not args.distance_y is None else -R_y  # propagation distance in meters
        wf_position_x = calculate_shift(speckle_shift_x, propagation_distance_x)
        wf_position_y = calculate_shift(speckle_shift_y, propagation_distance_y)

        if delta_f_x != 0: wavefront_x *= np.exp(1j * np.pi * (x_array ** 2) * delta_f_x / (wavelength * propagation_distance_x ** 2))
        if delta_f_y != 0: wavefront_y *= np.exp(1j * np.pi * (y_array ** 2) * delta_f_y / (wavelength * propagation_distance_y ** 2))

        initial_wavefront_x = GenericWavefront1D.initialize_wavefront_from_arrays(x_array=x_array, y_array=wavefront_x, wavelength=wavelength)
        initial_wavefront_y = GenericWavefront1D.initialize_wavefront_from_arrays(x_array=y_array, y_array=wavefront_y, wavelength=wavelength)

        # Instantiate the propagator
        fresnel_propagator = FresnelZoom1D()

        sigma_x, fwhm_x, intensity_x_wofry, x_coordinates = __propagate_1D(initial_wavefront_x,
                                                                           fresnel_propagator,
                                                                           propagation_distance_x,
                                                                           args.magnification_x,
                                                                           args.x_rms_range,
                                                                           "X",
                                                                           args.verbose)

        sigma_y, fwhm_y, intensity_y_wofry, y_coordinates = __propagate_1D(initial_wavefront_y,
                                                                           fresnel_propagator,
                                                                           propagation_distance_y,
                                                                           args.magnification_y,
                                                                           args.y_rms_range,
                                                                           "Y",
                                                                           args.verbose)

        if args.scan_best_focus:
            best_distance_x, best_x_coordinate, best_intensity_x, smallest_size_x, \
            bf_propagation_distances_x, bf_x_coordinates, bf_intensities_x, bf_size_values_x, bf_size_values_fit_x \
                = __scan_best_focus_1D(initial_wavefront_x,
                                       fresnel_propagator,
                                       propagation_distance_x,
                                       args.magnification_x,
                                       args.x_rms_range,
                                       args.scan_x_rel_range,
                                       args.use_fit,
                                       args.best_focus_from,
                                       "X",
                                       args.show_figure,
                                       args.verbose)
            best_distance_y, best_y_coordinate, best_intensity_y, smallest_size_y, \
            bf_propagation_distances_y, bf_y_coordinates, bf_intensities_y, bf_size_values_y, bf_size_values_fit_y \
                = __scan_best_focus_1D(initial_wavefront_y,
                                       fresnel_propagator,
                                       propagation_distance_y,
                                       args.magnification_y,
                                       args.y_rms_range,
                                       args.scan_y_rel_range,
                                       args.use_fit,
                                       args.best_focus_from,
                                       "Y",
                                       args.show_figure,
                                       args.verbose)
            focus_z_position_x = -(propagation_distance_x - best_distance_x)
            focus_z_position_y = -(propagation_distance_y - best_distance_y)
        else:
            focus_z_position_x = -(propagation_distance_x + R_x)
            focus_z_position_y = -(propagation_distance_y + R_y)

            best_distance_x = best_intensity_x = smallest_size_x = \
            bf_propagation_distances_x = bf_x_coordinates = bf_intensities_x = bf_size_values_x = bf_size_values_fit_x = \
            best_distance_y = bf_y_coordinates = best_intensity_y = smallest_size_y = \
            bf_propagation_distances_y = bf_intensities_y = bf_size_values_y = bf_size_values_fit_y = None

        # note: inf is used for the purpose of best focus scan, while NaN is the failed return value, useful for optimization purposes

        if not args.scan_best_focus:
            propagated_wavefront = PropagatedWavefront(kind="1D",
                                                       fwhm_x=fwhm_x if not np.isinf(fwhm_x) else np.nan,
                                                       fwhm_y=fwhm_y if not np.isinf(fwhm_y) else np.nan,
                                                       sigma_x=sigma_x if not np.isinf(sigma_x) else np.nan,
                                                       sigma_y=sigma_y if not np.isinf(sigma_x) else np.nan,
                                                       propagation_distance_x=propagation_distance_x,
                                                       propagation_distance_y=propagation_distance_y,
                                                       focus_z_position_x=focus_z_position_x,
                                                       focus_z_position_y=focus_z_position_y,
                                                       wf_position_x=wf_position_x,
                                                       wf_position_y=wf_position_y,
                                                       x_coordinates=x_coordinates,
                                                       y_coordinates=y_coordinates,
                                                       intensity_x=intensity_x_wofry,
                                                       intensity_y=intensity_y_wofry,
                                                       scan_best_focus=False)
        else:
            propagated_wavefront = PropagatedWavefront(kind="1D",
                                                       fwhm_x=fwhm_x if not np.isinf(fwhm_x) else np.nan,
                                                       fwhm_y=fwhm_y if not np.isinf(fwhm_y) else np.nan,
                                                       sigma_x=sigma_x if not np.isinf(sigma_x) else np.nan,
                                                       sigma_y=sigma_y if not np.isinf(sigma_x) else np.nan,
                                                       propagation_distance_x=propagation_distance_x,
                                                       propagation_distance_y=propagation_distance_y,
                                                       focus_z_position_x=focus_z_position_x,
                                                       focus_z_position_y=focus_z_position_y,
                                                       wf_position_x=wf_position_x,
                                                       wf_position_y=wf_position_y,
                                                       x_coordinates=x_coordinates,
                                                       y_coordinates=y_coordinates,
                                                       intensity_x=intensity_x_wofry,
                                                       intensity_y=intensity_y_wofry,
                                                       scan_best_focus=True,
                                                       scan_best_focus_from=best_focus_from,
                                                       bf_propagation_distance_x=best_distance_x,
                                                       bf_propagation_distance_y=best_distance_y,
                                                       bf_x_coordinate=best_x_coordinate,
                                                       bf_y_coordinate=best_y_coordinate,
                                                       bf_intensity_x=best_intensity_x,
                                                       bf_intensity_y=best_intensity_y,
                                                       bf_size_value_x=smallest_size_x,
                                                       bf_size_value_y=smallest_size_y,
                                                       bf_propagation_distances_x=bf_propagation_distances_x,
                                                       bf_propagation_distances_y=bf_propagation_distances_y,
                                                       bf_x_coordinates=bf_x_coordinates,
                                                       bf_y_coordinates=bf_y_coordinates,
                                                       bf_intensities_x=bf_intensities_x,
                                                       bf_intensities_y=bf_intensities_y,
                                                       bf_size_values_x=bf_size_values_x,
                                                       bf_size_values_fit_x=bf_size_values_fit_x,
                                                       bf_size_values_y=bf_size_values_y,
                                                       bf_size_values_fit_y=bf_size_values_fit_y)


        if args.show_figure:
            with lock:
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

                ax3.plot(x_coordinates + wf_position_x, intensity_x_wofry)
                ax4.plot(y_coordinates + wf_position_y, intensity_y_wofry)
                ax3.set_xlim(wf_position_x - 3*sigma_x, wf_position_x + 3*sigma_x)
                ax4.set_xlim(wf_position_y - 3*sigma_y, wf_position_y + 3*sigma_y)
                ax3.set_xlabel('X (meters)')
                ax4.set_xlabel('Y (meters)')
                ax3.set_ylabel('Integrated Intensity')

                plt.title(f'Intensity profile at {propagation_distance_x}x{propagation_distance_y} distance')
                plt.show()

        if args.save_result: propagated_wavefront.to_hdf5(os.path.join(args.folder, 'propagated_results.hdf5'))

        return propagated_wavefront.to_dict()
    else:
        raise ValueError(f"Propagation kind not recognized: {args.kind}")


def __get_scan_fit(coordinates, size_values, indexes):
    spline = CubicSpline(coordinates, size_values)
    best_distance_fit = fminbound(spline, coordinates[indexes[0]], coordinates[indexes[1]])
    smallest_size_fit = spline(best_distance_fit)

    return best_distance_fit, smallest_size_fit, spline

def __scan_best_focus_2D(initial_wavefront,
                         fresnel_propagator,
                         propagation_distance,
                         args):
    propagation_distances = np.arange(propagation_distance + args.scan_rel_range[0],
                                      propagation_distance + args.scan_rel_range[1],
                                      args.scan_rel_range[2])

    smallest_size_x  = np.inf
    best_distance_x  = 0
    best_distance_index_x = 0
    best_x_coordinate = None
    best_intensity_x = None
    best_integrated_intensity_x = None
    size_values_x    = []

    smallest_size_y  = np.inf
    best_distance_y  = 0
    best_distance_index_y = 0
    best_y_coordinate = None
    best_intensity_y = None
    best_integrated_intensity_y = None
    size_values_y    = []

    x_coordinates            = []
    y_coordinates            = []
    intensities              = []
    integrated_intensities_x = []
    integrated_intensities_y = []


    for index in range(len(propagation_distances)):
        distance = propagation_distances[index]

        sigma_x, \
        fwhm_x, \
        sigma_y, \
        fwhm_y, \
        intensity_wofry, \
        integrated_intensity_x, \
        integrated_intensity_y, \
        x_coord, \
        y_coord = __propagate_2D(initial_wavefront,
                                 fresnel_propagator,
                                 distance,
                                 args)
        if   args.best_focus_from == "rms":
            size_x = sigma_x
            size_y = sigma_y
        elif args.best_focus_from == "fwhm":
            size_x = fwhm_x
            size_y = fwhm_y
        else:
            raise ValueError(f"Best focus from not recognized {args.best_focus_from}")

        size_values_x.append(size_x)
        size_values_y.append(size_y)

        x_coordinates.append(x_coord)
        y_coordinates.append(y_coord)
        intensities.append(intensity_wofry)
        integrated_intensities_x.append(integrated_intensity_x)
        integrated_intensities_y.append(integrated_intensity_y)

        if size_x < smallest_size_x:
            smallest_size_x  = size_x
            best_distance_x  = distance
            best_x_coordinate = x_coord
            best_intensity_x = intensity_wofry
            best_integrated_intensity_x = integrated_intensity_x
            best_distance_index_x = index

        if size_y < smallest_size_y:
            smallest_size_y  = size_y
            best_distance_y  = distance
            best_y_coordinate = y_coord
            best_intensity_y = intensity_wofry
            best_integrated_intensity_y = integrated_intensity_y
            best_distance_index_y = index

    size_values_x         = np.array(size_values_x)
    size_values_y         = np.array(size_values_y)
    propagation_distances = np.array(propagation_distances)

    if args.use_fit:
        indexes_x = [max(0, best_distance_index_x-2), min(best_distance_index_x+2, len(propagation_distances)-1)]
        indexes_y = [max(0, best_distance_index_y-2), min(best_distance_index_y+2, len(propagation_distances)-1)]

        try:    best_distance_x_fit, smallest_size_x_fit, spline_x = __get_scan_fit(propagation_distances, size_values_x, indexes_x)
        except: best_distance_x_fit, smallest_size_x_fit, spline_x = best_distance_x, smallest_size_x, None

        try:    best_distance_y_fit, smallest_size_y_fit, spline_y = __get_scan_fit(propagation_distances, size_values_y, indexes_y)
        except: best_distance_y_fit, smallest_size_y_fit, spline_y = best_distance_y, smallest_size_y, None
    else:
        best_distance_x_fit, smallest_size_x_fit, spline_x = best_distance_x, smallest_size_x, None
        best_distance_y_fit, smallest_size_y_fit, spline_y = best_distance_y, smallest_size_y, None

    if args.verbose:
        print(f"Smallest size in X    : {round(1e6*smallest_size_x, 3)} um {args.best_focus_from} at distance {best_distance_x} m")
        print(f"Smallest size in Y    : {round(1e6*smallest_size_y, 3)} um {args.best_focus_from} at distance {best_distance_y} m")
        print(f"Smallest size in X FIT: {round(1e6*smallest_size_x_fit, 3)} um {args.best_focus_from} at distance {best_distance_x_fit} m")
        print(f"Smallest size in Y FIT: {round(1e6*smallest_size_y_fit, 3)} um {args.best_focus_from} at distance {best_distance_y_fit} m")

    size_values_x_fit = spline_x(propagation_distances) if not spline_x is None else None
    size_values_y_fit = spline_y(propagation_distances) if not spline_y is None else None

    if args.show_figure:
        with lock:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(propagation_distances, 1e6*size_values_x, label=f"Size X", marker='o')
            if not spline_x is None: plt.plot(propagation_distances, 1e6*size_values_x_fit, label=f"Size X - FIT")
            plt.xlabel('Distance (m)')
            plt.ylabel('Size X (um)')
            plt.title('Size as a Function of Distance')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(1e6*best_x_coordinate, best_intensity_x.sum(axis=1))
            plt.xlim(-3e6 * (smallest_size_x if spline_x is None else smallest_size_x_fit),
                     3e6 * (smallest_size_x if spline_x is None else smallest_size_x_fit))
            plt.xlabel('X ($\mu$)')
            plt.ylabel('Intensity')
            plt.title(f"Intensity profile at X waist")

            plt.tight_layout()  # Adjust spacing between plots
            plt.show()

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(propagation_distances, 1e6*size_values_y, label=f"Size Y", marker='o')
            if not spline_y is None:plt.plot(propagation_distances, 1e6*size_values_y_fit, label=f"Size Y - FIT")
            plt.xlabel('Distance (m)')
            plt.ylabel('Size Y (um)')
            plt.title('Size as a Function of Distance')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(1e6*best_y_coordinate, best_intensity_y.sum(axis=0))
            plt.xlim(-3e6*(smallest_size_y if spline_y is None else smallest_size_y_fit),
                     3e6*(smallest_size_y if spline_y is None else smallest_size_y_fit))
            plt.xlabel('Y ($\mu$)')
            plt.ylabel('Intensity')
            plt.title(f"Intensity profile at Y waist")

            plt.tight_layout()  # Adjust spacing between plots
            plt.show()

    return best_distance_x_fit, best_x_coordinate, best_intensity_x, best_integrated_intensity_x, smallest_size_x_fit, best_distance_y_fit, best_y_coordinate, best_intensity_y, best_integrated_intensity_y, smallest_size_y_fit, \
           propagation_distances, x_coordinates, y_coordinates, intensities, integrated_intensities_x, size_values_x, size_values_x_fit, integrated_intensities_y, size_values_y, size_values_y_fit


def __scan_best_focus_1D(initial_wavefront,
                         fresnel_propagator,
                         propagation_distance,
                         magnification,
                         rms_range,
                         scan_rel_range,
                         use_fit,
                         best_focus_from,
                         direction,
                         show_figure,
                         verbose):
    propagation_distances = np.arange(propagation_distance + scan_rel_range[0],
                                      propagation_distance + scan_rel_range[1],
                                      scan_rel_range[2])

    smallest_size  = np.inf
    best_distance  = 0
    best_distance_index = 0
    best_coordinate = None
    best_intensity = None
    size_values    = []

    intensities = []
    coordinates = []

    for index in range(len(propagation_distances)):
        distance = propagation_distances[index]
        sigma, fwhm, intensity_wofry, coord = __propagate_1D(fresnel_propagator,
                                                             initial_wavefront,
                                                             magnification,
                                                             distance,
                                                             rms_range,
                                                             direction,
                                                             verbose)
        if   best_focus_from == "rms":   size = sigma
        elif best_focus_from == "fwhm":  size = fwhm
        else: raise ValueError(f"Best focus from not recognized {best_focus_from}")

        coordinates.append(coord)
        size_values.append(size)
        intensities.append(intensity_wofry)

        if size < smallest_size:
            smallest_size  = size
            best_distance  = distance
            best_coordinate = coord
            best_intensity = intensity_wofry
            best_distance_index = index

    size_values           = np.array(size_values)
    propagation_distances = np.array(propagation_distances)

    if use_fit:
        indexes = [max(0, best_distance_index - 2), min(best_distance_index + 2, len(propagation_distances) - 1)]

        try:    best_distance_fit, smallest_size_fit, spline = __get_scan_fit(propagation_distances, size_values, indexes)
        except: best_distance_fit, smallest_size_fit, spline = best_distance, smallest_size, None
    else:
        best_distance_fit, smallest_size_fit, spline = best_distance, smallest_size, None

    if verbose:
        print(f"Smallest size in {direction}: {smallest_size} {best_focus_from} at distance {best_distance} m")
        print(f"Smallest size from fit in {direction}: {smallest_size_fit} {best_focus_from} at distance {best_distance_fit} m")

    size_values_fit = spline(propagation_distances) if not spline is None else None

    if show_figure:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(propagation_distances, size_values,     label=f"Size {direction}", marker='o')
        if not spline is None:plt.plot(propagation_distances, size_values_fit, label=f"Size {direction}")
        plt.xlabel('Distance (m)')
        plt.ylabel('Size (units)')
        plt.title('Size as a Function of Distance')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(best_coordinate, best_intensity)
        plt.xlabel('X (meters)')
        plt.ylabel('Intensity')
        plt.title(f"Intensity profile at {direction} waist")

        plt.tight_layout()  # Adjust spacing between plots
        plt.show()

    return best_distance, best_coordinate, best_intensity, smallest_size, \
           propagation_distances, coordinates, intensities, size_values, size_values_fit

def __propagate_1D(initial_wavefront,
                   fresnel_propagator,
                   propagation_distance,
                   magnification,
                   rms_range,
                   direction,
                   verbose):
    propagated_wavefront = fresnel_propagator.propagate_wavefront(initial_wavefront, propagation_distance, magnification_x=magnification)
    intensity_wofry      = propagated_wavefront.get_intensity()
    coordinates          = propagated_wavefront.get_abscissas()

    # Calculate beam size
    fwhm  = find_fwhm(coordinates, intensity_wofry)
    sigma = find_rms(coordinates, intensity_wofry, rms_range)

    if verbose: print(f"{direction} direction: sigma = {sigma:.3g}, FWHM = {fwhm:.3g}")

    return sigma, fwhm, intensity_wofry, coordinates

def __propagate_2D(initial_wavefront,
                   fresnel_propagator,
                   propagation_distance,
                   args):

    if args.verbose: print(f"Propagation distance: {propagation_distance}")

    if args.engine.lower() == "wofry":
        additional_parameters = {
            "magnification_x":  args.magnification_x,
            "magnification_y":  args.magnification_y,
            "shift_half_pixel": args.shift_half_pixel
        }

        propagation_elements = PropagationElements()
        propagation_elements.add_beamline_element(BeamlineElement(optical_element=WOScreen(name="Focus"),
                                                                  coordinates=ElementCoordinates(p=propagation_distance)))

        propagation_parameters = PropagationParameters(wavefront=initial_wavefront,
                                                       propagation_elements=propagation_elements,
                                                       **additional_parameters)

        propagated_wavefront_wofry = fresnel_propagator.do_propagation(propagation_parameters)
    elif args.engine.lower() == "srw":
        wavefront = initial_wavefront.duplicate()

        additional_parameters = {
            "srw_drift_before_wavefront_propagation_parameters" : WavefrontPropagationParameters(
                auto_resize_before_propagation=int(args.auto_resize_before_propagation),
                auto_resize_after_propagation=int(args.auto_resize_after_propagation),
                relative_precision_for_propagation_with_autoresizing=args.relative_precision_for_propagation_with_autoresizing,
                allow_semianalytical_treatment_of_quadratic_phase_term=args.allow_semianalytical_treatment_of_quadratic_phase_term,
                do_any_resizing_on_fourier_side_using_fft=int(args.do_any_resizing_on_fourier_side_using_fft),
                horizontal_range_modification_factor_at_resizing=args.horizontal_range_modification_factor_at_resizing,
                horizontal_resolution_modification_factor_at_resizing=args.horizontal_resolution_modification_factor_at_resizing,
                vertical_range_modification_factor_at_resizing=args.vertical_range_modification_factor_at_resizing,
                vertical_resolution_modification_factor_at_resizing=args.vertical_resolution_modification_factor_at_resizing)
        }

        propagation_elements = PropagationElements()
        propagation_elements.add_beamline_element(BeamlineElement(optical_element=SRWScreen(name="Focus"),
                                                                  coordinates=ElementCoordinates(p=propagation_distance)))

        propagation_parameters = PropagationParameters(wavefront=wavefront,
                                                       propagation_elements=propagation_elements,
                                                       **additional_parameters)

        propagated_wavefront_srw   = fresnel_propagator.do_propagation(propagation_parameters)
        propagated_wavefront_wofry = propagated_wavefront_srw.toGenericWavefront()

    intensity_wofry = np.abs(propagated_wavefront_wofry.get_complex_amplitude()) ** 2
    x_coordinates   = propagated_wavefront_wofry.get_coordinate_x()
    y_coordinates   = propagated_wavefront_wofry.get_coordinate_y()

    integrated_intensity_x = np.sum(intensity_wofry, axis=1)  # Sum over y
    integrated_intensity_y = np.sum(intensity_wofry, axis=0)  # Sum over x

    # Calculate beam size
    fwhm_x  = find_fwhm(x_coordinates, integrated_intensity_x)
    fwhm_y  = find_fwhm(y_coordinates, integrated_intensity_y)
    sigma_x = find_rms(x_coordinates, integrated_intensity_x, args.x_rms_range)
    sigma_y = find_rms(y_coordinates, integrated_intensity_y, args.y_rms_range)

    if args.verbose:
        print(f"X direction: sigma = {sigma_x:.3g}, FWHM = {fwhm_x:.3g}")
        print(f"Y direction: sigma = {sigma_y:.3g}, FWHM = {fwhm_y:.3g}")

    return sigma_x, \
           fwhm_x, \
           sigma_y,\
           fwhm_y, \
           intensity_wofry, \
           integrated_intensity_x, \
           integrated_intensity_y, \
           x_coordinates, \
           y_coordinates


