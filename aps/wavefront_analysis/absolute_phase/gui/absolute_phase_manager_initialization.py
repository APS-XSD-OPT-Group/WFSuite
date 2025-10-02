
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
from aps.wavefront_analysis.driver.beamline.wavefront_sensor import WavefrontSensorInitializationFile
from aps.wavefront_analysis.absolute_phase import wavefront_analyzer as WavefrontAnalyzerModule

def generate_initialization_parameters_from_ini(ini: IniFacade):
    # -----------------------------------------------------
    # Wavefront Analyzer

    wavefront_analyzer_configuration = {
        "data_analysis" : {
            "pattern_size" : WavefrontAnalyzerModule.PATTERN_SIZE,
            "pattern_thickness" : WavefrontAnalyzerModule.PATTERN_THICKNESS,
            "pattern_transmission" : WavefrontAnalyzerModule.PATTERN_TRANSMISSION,
            "ran_mask" : WavefrontAnalyzerModule.RAN_MASK,
            "propagation_distance" : WavefrontAnalyzerModule.PROPAGATION_DISTANCE,
            "energy" : WavefrontAnalyzerModule.ENERGY,
            "source_v" : WavefrontAnalyzerModule.SOURCE_V,
            "source_h" : WavefrontAnalyzerModule.SOURCE_H,
            "source_distance_v" : WavefrontAnalyzerModule.SOURCE_DISTANCE_V,
            "source_distance_h" : WavefrontAnalyzerModule.SOURCE_DISTANCE_H,
            "d_source_recal" : WavefrontAnalyzerModule.D_SOURCE_RECAL,
            "find_transfer_matrix" : WavefrontAnalyzerModule.FIND_TRANSFER_MATRIX,
            "crop" : WavefrontAnalyzerModule.CROP,
            "estimation_method" : WavefrontAnalyzerModule.ESTIMATION_METHOD,
            "propagator" : WavefrontAnalyzerModule.PROPAGATOR,
            "calibration_path" : WavefrontAnalyzerModule.CALIBRATION_PATH,
            "mode" : WavefrontAnalyzerModule.MODE,
            "line_width" : WavefrontAnalyzerModule.LINE_WIDTH,
            "rebinning" : WavefrontAnalyzerModule.REBINNING,
            "down_sampling" : WavefrontAnalyzerModule.DOWN_SAMPLING,
            "method" : WavefrontAnalyzerModule.METHOD,
            "use_gpu" : WavefrontAnalyzerModule.USE_GPU,
            "use_wavelet" : WavefrontAnalyzerModule.USE_WAVELET,
            "wavelet_cut" : WavefrontAnalyzerModule.WAVELET_CUT,
            "pyramid_level" : WavefrontAnalyzerModule.PYRAMID_LEVEL,
            "n_iterations" : WavefrontAnalyzerModule.N_ITERATIONS,
            "template_size" : WavefrontAnalyzerModule.TEMPLATE_SIZE,
            "window_search" : WavefrontAnalyzerModule.WINDOW_SEARCH,
            "crop_boundary" : WavefrontAnalyzerModule.CROP_BOUNDARY,
            "n_cores" : WavefrontAnalyzerModule.N_CORES,
            "n_group" : WavefrontAnalyzerModule.N_GROUP,
            "image_transfer_matrix" : WavefrontAnalyzerModule.IMAGE_TRANSFER_MATRIX,
            "show_align_figure" : WavefrontAnalyzerModule.SHOW_ALIGN_FIGURE,
            "correct_scale" : WavefrontAnalyzerModule.CORRECT_SCALE,
        },
        "back_propagation" :{
            "kind" : WavefrontAnalyzerModule.KIND,
            "rebinning_bp" : WavefrontAnalyzerModule.REBINNING_BP,
            "smooth_intensity" : WavefrontAnalyzerModule.SMOOTH_INTENSITY,
            "sigma_intensity" : WavefrontAnalyzerModule.SIGMA_INTENSITY,
            "smooth_phase" : WavefrontAnalyzerModule.SMOOTH_PHASE,
            "sigma_phase" : WavefrontAnalyzerModule.SIGMA_PHASE,
            "filter_intensity" : WavefrontAnalyzerModule.FILTER_INTENSITY,
            "filter_phase" : WavefrontAnalyzerModule.FILTER_PHASE,
            "crop_v" : WavefrontAnalyzerModule.CROP_V,
            "crop_h" : WavefrontAnalyzerModule.CROP_H,
            "crop_shift_v" : WavefrontAnalyzerModule.CROP_SHIFT_V,
            "crop_shift_h" : WavefrontAnalyzerModule.CROP_SHIFT_H,
            "distance" : WavefrontAnalyzerModule.DISTANCE,
            "distance_v" : WavefrontAnalyzerModule.DISTANCE_V,
            "distance_h" : WavefrontAnalyzerModule.DISTANCE_H,
            "delta_f_v" : WavefrontAnalyzerModule.DELTA_F_V,
            "delta_f_h" : WavefrontAnalyzerModule.DELTA_F_H,
            "engine"  : WavefrontAnalyzerModule.ENGINE,
            "magnification_v": WavefrontAnalyzerModule.MAGNIFICATION_V,
            "magnification_h": WavefrontAnalyzerModule.MAGNIFICATION_H,
            "shift_half_pixel": WavefrontAnalyzerModule.SHIFT_HALF_PIXEL,
            "auto_resize_before_propagation" : WavefrontAnalyzerModule.AUTO_RESIZE_BEFORE_PROPAGATION,
            "auto_resize_after_propagation" : WavefrontAnalyzerModule.AUTO_RESIZE_AFTER_PROPAGATION,
            "relative_precision_for_propagation_with_autoresizing" : WavefrontAnalyzerModule.RELATIVE_PRECISION_FOR_PROPAGATION_WITH_AUTORESIZING,
            "allow_semianalytical_treatment_of_quadratic_phase_term" : WavefrontAnalyzerModule.ALLOW_SEMIANALYTICAL_TREATMENT_OF_QUADRATIC_PHASE_TERM,
            "do_any_resizing_on_fourier_side_using_fft" : WavefrontAnalyzerModule.DO_ANY_RESIZING_ON_FOURIER_SIDE_USING_FFT,
            "horizontal_range_modification_factor_at_resizing" : WavefrontAnalyzerModule.HORIZONTAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING,
            "horizontal_resolution_modification_factor_at_resizing" : WavefrontAnalyzerModule.HORIZONTAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING,
            "vertical_range_modification_factor_at_resizing" : WavefrontAnalyzerModule.VERTICAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING,
            "vertical_resolution_modification_factor_at_resizing" : WavefrontAnalyzerModule.VERTICAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING,
            "rms_range_v" : WavefrontAnalyzerModule.RMS_RANGE_V,
            "rms_range_h" : WavefrontAnalyzerModule.RMS_RANGE_H,
            "scan_best_focus" : WavefrontAnalyzerModule.SCAN_BEST_FOCUS,
            "use_fit" : WavefrontAnalyzerModule.USE_FIT,
            "best_focus_from" : WavefrontAnalyzerModule.BEST_FOCUS_FROM,
            "best_focus_scan_range" : WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE,
            "best_focus_scan_range_v" : WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE_V,
            "best_focus_scan_range_h" : WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE_H,
        }
    }

    # Here GUI specific ini

    wavefront_sensor_mode            = ini.get_int_from_ini(section="Wavefront-Sensor", key="Wavefront-Sensor-Mode", default=0) # if offline, the file name can be built
    plot_rebinning_factor            = ini.get_int_from_ini(section="Wavefront-Sensor", key="Plot-Rebinning-Factor", default=4)

    image_index                      = ini.get_int_from_ini(    section="Wavefront-Analyzer", key="Image-Index",                       default=1)
    file_name_type                   = ini.get_int_from_ini(    section="Wavefront-Analyzer", key="File-Name-Type",                    default=0)
    index_digits_custom              = ini.get_int_from_ini(    section="Wavefront-Analyzer", key="Index-Digits-Custom",               default=5)
    file_name_prefix_custom          = ini.get_string_from_ini( section="Wavefront-Analyzer", key="File-Name-Prefix-Custom",           default="custom_file_prefix")
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
                      wavefront_analyzer_configuration=wavefront_analyzer_configuration)


def set_ini_from_initialization_parameters(initialization_parameters: ScriptData, ini: IniFacade):
    # -----------------------------------------------------
    # Wavefront Analyzer

    wavefront_analyzer_configuration = initialization_parameters.get_parameter("wavefront_analyzer_configuration")
    data_analysis_configuration      = wavefront_analyzer_configuration["data_analysis"]
    back_propagation_configuration   = wavefront_analyzer_configuration["back_propagation"]

    WavefrontAnalyzerModule.PATTERN_SIZE = data_analysis_configuration["pattern_size"]          
    WavefrontAnalyzerModule.PATTERN_THICKNESS = data_analysis_configuration["pattern_thickness"]     
    WavefrontAnalyzerModule.PATTERN_TRANSMISSION = data_analysis_configuration["pattern_transmission"]  
    WavefrontAnalyzerModule.RAN_MASK = data_analysis_configuration["ran_mask"]              
    WavefrontAnalyzerModule.PROPAGATION_DISTANCE = data_analysis_configuration["propagation_distance"]  
    WavefrontAnalyzerModule.ENERGY = data_analysis_configuration["energy"]                
    WavefrontAnalyzerModule.SOURCE_V = data_analysis_configuration["source_v"]              
    WavefrontAnalyzerModule.SOURCE_H = data_analysis_configuration["source_h"]              
    WavefrontAnalyzerModule.SOURCE_DISTANCE_V = data_analysis_configuration["source_distance_v"]     
    WavefrontAnalyzerModule.SOURCE_DISTANCE_H = data_analysis_configuration["source_distance_h"]     
    WavefrontAnalyzerModule.D_SOURCE_RECAL = data_analysis_configuration["d_source_recal"]        
    WavefrontAnalyzerModule.FIND_TRANSFER_MATRIX = data_analysis_configuration["find_transfer_matrix"]
    WavefrontAnalyzerModule.CROP = data_analysis_configuration["crop"]
    WavefrontAnalyzerModule.ESTIMATION_METHOD = data_analysis_configuration["estimation_method"]     
    WavefrontAnalyzerModule.PROPAGATOR = data_analysis_configuration["propagator"]            
    WavefrontAnalyzerModule.CALIBRATION_PATH = data_analysis_configuration["calibration_path"]
    WavefrontAnalyzerModule.MODE = data_analysis_configuration["mode"]                  
    WavefrontAnalyzerModule.LINE_WIDTH = data_analysis_configuration["line_width"]            
    WavefrontAnalyzerModule.REBINNING = data_analysis_configuration["rebinning"]             
    WavefrontAnalyzerModule.DOWN_SAMPLING = data_analysis_configuration["down_sampling"]         
    WavefrontAnalyzerModule.METHOD = data_analysis_configuration["method"]                
    WavefrontAnalyzerModule.USE_GPU = data_analysis_configuration["use_gpu"]               
    WavefrontAnalyzerModule.USE_WAVELET = data_analysis_configuration["use_wavelet"]           
    WavefrontAnalyzerModule.WAVELET_CUT = data_analysis_configuration["wavelet_cut"]           
    WavefrontAnalyzerModule.PYRAMID_LEVEL = data_analysis_configuration["pyramid_level"]         
    WavefrontAnalyzerModule.N_ITERATIONS = data_analysis_configuration["n_iterations"]          
    WavefrontAnalyzerModule.TEMPLATE_SIZE = data_analysis_configuration["template_size"]         
    WavefrontAnalyzerModule.WINDOW_SEARCH = data_analysis_configuration["window_search"]         
    WavefrontAnalyzerModule.CROP_BOUNDARY = data_analysis_configuration["crop_boundary"]         
    WavefrontAnalyzerModule.N_CORES = data_analysis_configuration["n_cores"]               
    WavefrontAnalyzerModule.N_GROUP = data_analysis_configuration["n_group"]               
    WavefrontAnalyzerModule.IMAGE_TRANSFER_MATRIX = data_analysis_configuration["image_transfer_matrix"] 
    WavefrontAnalyzerModule.SHOW_ALIGN_FIGURE = data_analysis_configuration["show_align_figure"]     
    WavefrontAnalyzerModule.CORRECT_SCALE = data_analysis_configuration["correct_scale"]         
    
    WavefrontAnalyzerModule.KIND = back_propagation_configuration["kind"]
    WavefrontAnalyzerModule.REBINNING_BP = back_propagation_configuration["rebinning_bp"]
    WavefrontAnalyzerModule.SMOOTH_INTENSITY = back_propagation_configuration["smooth_intensity"]
    WavefrontAnalyzerModule.FILTER_INTENSITY = back_propagation_configuration["filter_intensity"]
    WavefrontAnalyzerModule.SIGMA_INTENSITY = back_propagation_configuration["sigma_intensity"]
    WavefrontAnalyzerModule.SMOOTH_PHASE = back_propagation_configuration["smooth_phase"]
    WavefrontAnalyzerModule.FILTER_PHASE = back_propagation_configuration["filter_phase"]
    WavefrontAnalyzerModule.SIGMA_PHASE = back_propagation_configuration["sigma_phase"]
    WavefrontAnalyzerModule.CROP_V = back_propagation_configuration["crop_v"]                
    WavefrontAnalyzerModule.CROP_H = back_propagation_configuration["crop_h"]                
    WavefrontAnalyzerModule.CROP_SHIFT_V = back_propagation_configuration["crop_shift_v"]    
    WavefrontAnalyzerModule.CROP_SHIFT_H = back_propagation_configuration["crop_shift_h"]    
    WavefrontAnalyzerModule.DISTANCE = back_propagation_configuration["distance"]            
    WavefrontAnalyzerModule.DISTANCE_V = back_propagation_configuration["distance_v"]        
    WavefrontAnalyzerModule.DISTANCE_H = back_propagation_configuration["distance_h"]        
    WavefrontAnalyzerModule.DELTA_F_V = back_propagation_configuration["delta_f_v"]          
    WavefrontAnalyzerModule.DELTA_F_H = back_propagation_configuration["delta_f_h"]          
    WavefrontAnalyzerModule.ENGINE = back_propagation_configuration["engine"]

    WavefrontAnalyzerModule.MAGNIFICATION_V = back_propagation_configuration["magnification_v"]
    WavefrontAnalyzerModule.MAGNIFICATION_H = back_propagation_configuration["magnification_h"]
    WavefrontAnalyzerModule.SHIFT_HALF_PIXEL = back_propagation_configuration["shift_half_pixel"]

    WavefrontAnalyzerModule.AUTO_RESIZE_BEFORE_PROPAGATION                         = back_propagation_configuration["auto_resize_before_propagation"]
    WavefrontAnalyzerModule.AUTO_RESIZE_AFTER_PROPAGATION                          = back_propagation_configuration["auto_resize_after_propagation"]
    WavefrontAnalyzerModule.RELATIVE_PRECISION_FOR_PROPAGATION_WITH_AUTORESIZING   = back_propagation_configuration["relative_precision_for_propagation_with_autoresizing"]
    WavefrontAnalyzerModule.ALLOW_SEMIANALYTICAL_TREATMENT_OF_QUADRATIC_PHASE_TERM = back_propagation_configuration["allow_semianalytical_treatment_of_quadratic_phase_term"]
    WavefrontAnalyzerModule.DO_ANY_RESIZING_ON_FOURIER_SIDE_USING_FFT              = back_propagation_configuration["do_any_resizing_on_fourier_side_using_fft"]
    WavefrontAnalyzerModule.HORIZONTAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING       = back_propagation_configuration["horizontal_range_modification_factor_at_resizing"]
    WavefrontAnalyzerModule.HORIZONTAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING  = back_propagation_configuration["horizontal_resolution_modification_factor_at_resizing"]
    WavefrontAnalyzerModule.VERTICAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING         = back_propagation_configuration["vertical_range_modification_factor_at_resizing"]
    WavefrontAnalyzerModule.VERTICAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING    = back_propagation_configuration["vertical_resolution_modification_factor_at_resizing"]

    WavefrontAnalyzerModule.RMS_RANGE_V = back_propagation_configuration["rms_range_v"]
    WavefrontAnalyzerModule.RMS_RANGE_H = back_propagation_configuration["rms_range_h"]      
    WavefrontAnalyzerModule.SCAN_BEST_FOCUS = back_propagation_configuration["scan_best_focus"]
    WavefrontAnalyzerModule.USE_FIT = back_propagation_configuration["use_fit"]                  
    WavefrontAnalyzerModule.BEST_FOCUS_FROM = back_propagation_configuration["best_focus_from"]   
    WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE = back_propagation_configuration["best_focus_scan_range"]     
    WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE_V = back_propagation_configuration["best_focus_scan_range_v"]  
    WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE_H = back_propagation_configuration["best_focus_scan_range_h"]  
    
    WavefrontAnalyzerModule.store()
    
    # Here GUI specific ini

    ini.set_value_at_ini(section="Wavefront-Sensor", key="Wavefront-Sensor-Mode", value=initialization_parameters.get_parameter("wavefront_sensor_mode"))
    ini.set_value_at_ini(section="Wavefront-Sensor", key="Plot-Rebinning-Factor", value=initialization_parameters.get_parameter("plot_rebinning_factor"))

    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Image-Index",                       value=initialization_parameters.get_parameter("image_index"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="File-Name-Type",                    value=initialization_parameters.get_parameter("file_name_type"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="Index-Digits-Custom",               value=initialization_parameters.get_parameter("index_digits_custom"))
    ini.set_value_at_ini(section="Wavefront-Analyzer", key="File-Name-Prefix-Custom",           value=initialization_parameters.get_parameter("file_name_prefix_custom"))
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