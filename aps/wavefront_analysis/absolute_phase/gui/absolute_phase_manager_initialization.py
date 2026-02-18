
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
from aps.common.initializer import IniFacade
from aps.common.scripts.script_data import ScriptData
from aps.wavefront_analysis.driver.wavefront_sensor import WavefrontSensorInitializationFile
from aps.wavefront_analysis.absolute_phase import absolute_phase_analyzer as wa
from aps.wavefront_analysis.driver import wavefront_sensor as ws

def generate_initialization_parameters_from_ini(ini: IniFacade):
    # -----------------------------------------------------
    # Wavefront Analyzer

    absolute_phase_analyzer_configuration = {
        "data_analysis" : {
            "data_directory" : wa.DATA_DIRECTORY,
            "pattern_size" : wa.PATTERN_SIZE,
            "pattern_thickness" : wa.PATTERN_THICKNESS,
            "pattern_transmission" : wa.PATTERN_TRANSMISSION,
            "ran_mask" : wa.RAN_MASK,
            "propagation_distance" : wa.PROPAGATION_DISTANCE,
            "energy" : wa.ENERGY,
            "source_v" : wa.SOURCE_V,
            "source_h" : wa.SOURCE_H,
            "source_distance_v" : wa.SOURCE_DISTANCE_V,
            "source_distance_h" : wa.SOURCE_DISTANCE_H,
            "d_source_recal" : wa.D_SOURCE_RECAL,
            "find_transfer_matrix" : wa.FIND_TRANSFER_MATRIX,
            "crop" : wa.CROP,
            "estimation_method" : wa.ESTIMATION_METHOD,
            "propagator" : wa.PROPAGATOR,
            "calibration_path" : wa.CALIBRATION_PATH,
            "mode" : wa.MODE,
            "line_width" : wa.LINE_WIDTH,
            "rebinning" : wa.REBINNING,
            "down_sampling" : wa.DOWN_SAMPLING,
            "method" : wa.METHOD,
            "use_gpu" : wa.USE_GPU,
            "use_wavelet" : wa.USE_WAVELET,
            "wavelet_cut" : wa.WAVELET_CUT,
            "pyramid_level" : wa.PYRAMID_LEVEL,
            "n_iterations" : wa.N_ITERATIONS,
            "template_size" : wa.TEMPLATE_SIZE,
            "window_search" : wa.WINDOW_SEARCH,
            "crop_boundary" : wa.CROP_BOUNDARY,
            "n_cores" : wa.N_CORES,
            "n_group" : wa.N_GROUP,
            "image_transfer_matrix" : wa.IMAGE_TRANSFER_MATRIX,
            "show_align_figure" : wa.SHOW_ALIGN_FIGURE,
            "correct_scale" : wa.CORRECT_SCALE,
        },
        "back_propagation" :{
            "kind" : wa.KIND,
            "rebinning_bp" : wa.REBINNING_BP,
            "smooth_intensity" : wa.SMOOTH_INTENSITY,
            "sigma_intensity" : wa.SIGMA_INTENSITY,
            "smooth_phase" : wa.SMOOTH_PHASE,
            "sigma_phase" : wa.SIGMA_PHASE,
            "filter_intensity" : wa.FILTER_INTENSITY,
            "filter_phase" : wa.FILTER_PHASE,
            "crop_v" : wa.CROP_V,
            "crop_h" : wa.CROP_H,
            "crop_shift_v" : wa.CROP_SHIFT_V,
            "crop_shift_h" : wa.CROP_SHIFT_H,
            "distance" : wa.DISTANCE,
            "distance_v" : wa.DISTANCE_V,
            "distance_h" : wa.DISTANCE_H,
            "delta_f_v" : wa.DELTA_F_V,
            "delta_f_h" : wa.DELTA_F_H,
            "engine"  : wa.ENGINE,
            "magnification_v": wa.MAGNIFICATION_V,
            "magnification_h": wa.MAGNIFICATION_H,
            "shift_half_pixel": wa.SHIFT_HALF_PIXEL,
            "auto_resize_before_propagation" : wa.AUTO_RESIZE_BEFORE_PROPAGATION,
            "auto_resize_after_propagation" : wa.AUTO_RESIZE_AFTER_PROPAGATION,
            "relative_precision_for_propagation_with_autoresizing" : wa.RELATIVE_PRECISION_FOR_PROPAGATION_WITH_AUTORESIZING,
            "allow_semianalytical_treatment_of_quadratic_phase_term" : wa.ALLOW_SEMIANALYTICAL_TREATMENT_OF_QUADRATIC_PHASE_TERM,
            "do_any_resizing_on_fourier_side_using_fft" : wa.DO_ANY_RESIZING_ON_FOURIER_SIDE_USING_FFT,
            "horizontal_range_modification_factor_at_resizing" : wa.HORIZONTAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING,
            "horizontal_resolution_modification_factor_at_resizing" : wa.HORIZONTAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING,
            "vertical_range_modification_factor_at_resizing" : wa.VERTICAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING,
            "vertical_resolution_modification_factor_at_resizing" : wa.VERTICAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING,
            "rms_range_v" : wa.RMS_RANGE_V,
            "rms_range_h" : wa.RMS_RANGE_H,
            "scan_best_focus" : wa.SCAN_BEST_FOCUS,
            "use_fit" : wa.USE_FIT,
            "best_focus_from" : wa.BEST_FOCUS_FROM,
            "best_focus_scan_range" : wa.BEST_FOCUS_SCAN_RANGE,
            "best_focus_scan_range_v" : wa.BEST_FOCUS_SCAN_RANGE_V,
            "best_focus_scan_range_h" : wa.BEST_FOCUS_SCAN_RANGE_H,
        }
    }

    # Here GUI specific ini

    wavefront_sensor_mode            = ini.get_int_from_ini(section="Wavefront-Sensor", key="Wavefront-Sensor-Mode", default=0) # if offline, the file name can be built
    plot_rebinning_factor            = ini.get_int_from_ini(section="Wavefront-Sensor", key="Plot-Rebinning-Factor", default=4)

    image_index                      = ini.get_int_from_ini(    section="Wavefront-Analyzer", key="Image-Index",                       default=1)
    file_name_type                   = ini.get_int_from_ini(    section="Wavefront-Analyzer", key="File-Name-Type",                    default=0)
    index_digits_custom              = ini.get_int_from_ini(    section="Wavefront-Analyzer", key="Index-Digits-Custom",               default=ws.INDEX_DIGITS)
    file_name_prefix_custom          = ini.get_string_from_ini( section="Wavefront-Analyzer", key="File-Name-Prefix-Custom",           default="custom_file_prefix")
    pixel_size_type                  = ini.get_int_from_ini(    section="Wavefront-Analyzer", key="Pixel-Size-Type",                   default=0)
    pixel_size_custom                = ini.get_float_from_ini(  section="Wavefront-Analyzer", key="Pixel-Size-Custom",                 default=ws.PIXEL_SIZE)
    image_directory                  = ini.get_string_from_ini( section="Wavefront-Analyzer", key="Image-Directory",                   default=os.path.abspath(os.path.join(WavefrontSensorInitializationFile.DEFAULT_IMAGE_DIRECTORY, "wf_images")))
    image_directory_batch            = ini.get_string_from_ini( section="Wavefront-Analyzer", key="Image-Directory-Batch",             default=os.path.abspath(os.path.join(WavefrontSensorInitializationFile.DEFAULT_IMAGE_DIRECTORY, "wf_images")))
    simulated_mask_directory         = ini.get_string_from_ini( section="Wavefront-Analyzer", key="Simulated-Mask-Directory",          default=os.path.abspath(os.path.join(WavefrontSensorInitializationFile.DEFAULT_IMAGE_DIRECTORY, "wf_images", "simulated_mask")))
    simulated_mask_directory_batch   = ini.get_string_from_ini( section="Wavefront-Analyzer", key="Simulated-Mask-Directory-Batch",    default=os.path.abspath(os.path.join(WavefrontSensorInitializationFile.DEFAULT_IMAGE_DIRECTORY, "wf_images", "simulated_mask")))
    use_flat                         = ini.get_boolean_from_ini(section="Wavefront-Analyzer", key="Use-Flat",                          default=False)
    use_dark                         = ini.get_boolean_from_ini(section="Wavefront-Analyzer", key="Use-Dark",                          default=False)
    save_images                      = ini.get_boolean_from_ini(section="Wavefront-Analyzer", key="Save-Images",                       default=True)
    bp_calibration_mode              = ini.get_boolean_from_ini(section="Wavefront-Analyzer", key="Back-Propagation-Calibration-Mode", default=False)
    bp_plot_shift                    = ini.get_boolean_from_ini(section="Wavefront-Analyzer", key="Back-Propagation-Plot-Shift",       default=True)

    return ScriptData(wavefront_sensor_mode=wavefront_sensor_mode,
                      plot_rebinning_factor=plot_rebinning_factor,
                      image_index=image_index,
                      index_digits_custom=index_digits_custom,
                      pixel_size_type=pixel_size_type,
                      pixel_size_custom=pixel_size_custom,
                      file_name_type=file_name_type,
                      file_name_prefix_custom=file_name_prefix_custom,
                      image_directory=image_directory,
                      image_directory_batch=image_directory_batch,
                      simulated_mask_directory=simulated_mask_directory,
                      simulated_mask_directory_batch=simulated_mask_directory_batch,
                      use_dark=use_dark,
                      use_flat=use_flat,
                      save_images=save_images,
                      bp_calibration_mode=bp_calibration_mode,
                      bp_plot_shift=bp_plot_shift,
                      absolute_phase_analyzer_configuration=absolute_phase_analyzer_configuration)


def set_ini_from_initialization_parameters(initialization_parameters: ScriptData, ini: IniFacade):
    # -----------------------------------------------------
    # Wavefront Analyzer

    absolute_phase_analyzer_configuration = initialization_parameters.get_parameter("absolute_phase_analyzer_configuration")
    data_analysis_configuration      = absolute_phase_analyzer_configuration["data_analysis"]
    back_propagation_configuration   = absolute_phase_analyzer_configuration["back_propagation"]

    wa.DATA_DIRECTORY = data_analysis_configuration["data_directory"]
    wa.PATTERN_SIZE = data_analysis_configuration["pattern_size"]
    wa.PATTERN_THICKNESS = data_analysis_configuration["pattern_thickness"]
    wa.PATTERN_TRANSMISSION = data_analysis_configuration["pattern_transmission"]
    wa.RAN_MASK = data_analysis_configuration["ran_mask"]
    wa.PROPAGATION_DISTANCE = data_analysis_configuration["propagation_distance"]
    wa.ENERGY = data_analysis_configuration["energy"]
    wa.SOURCE_V = data_analysis_configuration["source_v"]
    wa.SOURCE_H = data_analysis_configuration["source_h"]
    wa.SOURCE_DISTANCE_V = data_analysis_configuration["source_distance_v"]
    wa.SOURCE_DISTANCE_H = data_analysis_configuration["source_distance_h"]
    wa.D_SOURCE_RECAL = data_analysis_configuration["d_source_recal"]
    wa.FIND_TRANSFER_MATRIX = data_analysis_configuration["find_transfer_matrix"]
    wa.CROP = data_analysis_configuration["crop"]
    wa.ESTIMATION_METHOD = data_analysis_configuration["estimation_method"]
    wa.PROPAGATOR = data_analysis_configuration["propagator"]
    wa.CALIBRATION_PATH = data_analysis_configuration["calibration_path"]
    wa.MODE = data_analysis_configuration["mode"]
    wa.LINE_WIDTH = data_analysis_configuration["line_width"]
    wa.REBINNING = data_analysis_configuration["rebinning"]
    wa.DOWN_SAMPLING = data_analysis_configuration["down_sampling"]
    wa.METHOD = data_analysis_configuration["method"]
    wa.USE_GPU = data_analysis_configuration["use_gpu"]
    wa.USE_WAVELET = data_analysis_configuration["use_wavelet"]
    wa.WAVELET_CUT = data_analysis_configuration["wavelet_cut"]
    wa.PYRAMID_LEVEL = data_analysis_configuration["pyramid_level"]
    wa.N_ITERATIONS = data_analysis_configuration["n_iterations"]
    wa.TEMPLATE_SIZE = data_analysis_configuration["template_size"]
    wa.WINDOW_SEARCH = data_analysis_configuration["window_search"]
    wa.CROP_BOUNDARY = data_analysis_configuration["crop_boundary"]
    wa.N_CORES = data_analysis_configuration["n_cores"]
    wa.N_GROUP = data_analysis_configuration["n_group"]
    wa.IMAGE_TRANSFER_MATRIX = data_analysis_configuration["image_transfer_matrix"]
    wa.SHOW_ALIGN_FIGURE = data_analysis_configuration["show_align_figure"]
    wa.CORRECT_SCALE = data_analysis_configuration["correct_scale"]
    
    wa.KIND = back_propagation_configuration["kind"]
    wa.REBINNING_BP = back_propagation_configuration["rebinning_bp"]
    wa.SMOOTH_INTENSITY = back_propagation_configuration["smooth_intensity"]
    wa.FILTER_INTENSITY = back_propagation_configuration["filter_intensity"]
    wa.SIGMA_INTENSITY = back_propagation_configuration["sigma_intensity"]
    wa.SMOOTH_PHASE = back_propagation_configuration["smooth_phase"]
    wa.FILTER_PHASE = back_propagation_configuration["filter_phase"]
    wa.SIGMA_PHASE = back_propagation_configuration["sigma_phase"]
    wa.CROP_V = back_propagation_configuration["crop_v"]
    wa.CROP_H = back_propagation_configuration["crop_h"]
    wa.CROP_SHIFT_V = back_propagation_configuration["crop_shift_v"]
    wa.CROP_SHIFT_H = back_propagation_configuration["crop_shift_h"]
    wa.DISTANCE = back_propagation_configuration["distance"]
    wa.DISTANCE_V = back_propagation_configuration["distance_v"]
    wa.DISTANCE_H = back_propagation_configuration["distance_h"]
    wa.DELTA_F_V = back_propagation_configuration["delta_f_v"]
    wa.DELTA_F_H = back_propagation_configuration["delta_f_h"]
    wa.ENGINE = back_propagation_configuration["engine"]

    wa.MAGNIFICATION_V = back_propagation_configuration["magnification_v"]
    wa.MAGNIFICATION_H = back_propagation_configuration["magnification_h"]
    wa.SHIFT_HALF_PIXEL = back_propagation_configuration["shift_half_pixel"]

    wa.AUTO_RESIZE_BEFORE_PROPAGATION                         = back_propagation_configuration["auto_resize_before_propagation"]
    wa.AUTO_RESIZE_AFTER_PROPAGATION                          = back_propagation_configuration["auto_resize_after_propagation"]
    wa.RELATIVE_PRECISION_FOR_PROPAGATION_WITH_AUTORESIZING   = back_propagation_configuration["relative_precision_for_propagation_with_autoresizing"]
    wa.ALLOW_SEMIANALYTICAL_TREATMENT_OF_QUADRATIC_PHASE_TERM = back_propagation_configuration["allow_semianalytical_treatment_of_quadratic_phase_term"]
    wa.DO_ANY_RESIZING_ON_FOURIER_SIDE_USING_FFT              = back_propagation_configuration["do_any_resizing_on_fourier_side_using_fft"]
    wa.HORIZONTAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING       = back_propagation_configuration["horizontal_range_modification_factor_at_resizing"]
    wa.HORIZONTAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING  = back_propagation_configuration["horizontal_resolution_modification_factor_at_resizing"]
    wa.VERTICAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING         = back_propagation_configuration["vertical_range_modification_factor_at_resizing"]
    wa.VERTICAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING    = back_propagation_configuration["vertical_resolution_modification_factor_at_resizing"]

    wa.RMS_RANGE_V = back_propagation_configuration["rms_range_v"]
    wa.RMS_RANGE_H = back_propagation_configuration["rms_range_h"]
    wa.SCAN_BEST_FOCUS = back_propagation_configuration["scan_best_focus"]
    wa.USE_FIT = back_propagation_configuration["use_fit"]
    wa.BEST_FOCUS_FROM = back_propagation_configuration["best_focus_from"]
    wa.BEST_FOCUS_SCAN_RANGE = back_propagation_configuration["best_focus_scan_range"]
    wa.BEST_FOCUS_SCAN_RANGE_V = back_propagation_configuration["best_focus_scan_range_v"]
    wa.BEST_FOCUS_SCAN_RANGE_H = back_propagation_configuration["best_focus_scan_range_h"]
    
    wa.store()
    
    # Here GUI specific ini

    ini.set_value_at_ini(section="Wavefront-Sensor", key="Wavefront-Sensor-Mode", value=initialization_parameters.get_parameter("wavefront_sensor_mode"))
    ini.set_value_at_ini(section="Wavefront-Sensor", key="Plot-Rebinning-Factor", value=initialization_parameters.get_parameter("plot_rebinning_factor"))

    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Image-Index",                       value=initialization_parameters.get_parameter("image_index"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="File-Name-Type",                    value=initialization_parameters.get_parameter("file_name_type"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Index-Digits-Custom",               value=initialization_parameters.get_parameter("index_digits_custom"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="File-Name-Prefix-Custom",           value=initialization_parameters.get_parameter("file_name_prefix_custom"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Pixel-Size-Type",                   value=initialization_parameters.get_parameter("pixel_size_type"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Pixel-Size-Custom",                 value=initialization_parameters.get_parameter("pixel_size_custom"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Image-Directory",                   value=initialization_parameters.get_parameter("image_directory"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Image-Directory-Batch",             value=initialization_parameters.get_parameter("image_directory_batch"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Simulated-Mask-Directory",          value=initialization_parameters.get_parameter("simulated_mask_directory"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Simulated-Mask-Directory-Batch",    value=initialization_parameters.get_parameter("simulated_mask_directory_batch"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Use-Flat",                          value=initialization_parameters.get_parameter("use_flat"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Use-Dark",                          value=initialization_parameters.get_parameter("use_dark"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Save-Images",                       value=initialization_parameters.get_parameter("save_images"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Back-Propagation-Calibration-Mode", value=initialization_parameters.get_parameter("bp_calibration_mode"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Back-Propagation-Plot-Shift",       value=initialization_parameters.get_parameter("bp_plot_shift"))

    ini.push()