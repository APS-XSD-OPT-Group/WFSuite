
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
from aps.wavefront_analysis.absolute_phase import wavefront_analyzer as WavefrontAnalyzerModule

def get_data_from_int_to_string(data_from : int):
    if   data_from == 0: return "stream"
    elif data_from == 1: return "file"
    else: raise ValueError("Data From not recognized")

def get_data_from_string_to_int(data_from : str):
    if   data_from == "stream": return 0
    elif data_from == "file":   return 1
    else: raise ValueError("Data From not recognized")

def generate_initialization_parameters_from_ini(ini: IniFacade):    
    # -----------------------------------------------------
    # Wavefront Sensor

    wavefront_sensor_configuration = {
        "send_stop_command" : WavefrontSensorInitializationFile.SEND_STOP_COMMAND, 
        "send_save_command" : WavefrontSensorInitializationFile.SEND_SAVE_COMMAND, 
        "remove_image" : WavefrontSensorInitializationFile.REMOVE_IMAGE, 
        "wait_time" : WavefrontSensorInitializationFile.WAIT_TIME, 
        "exposure_time" : WavefrontSensorInitializationFile.EXPOSURE_TIME, 
        "pause_after_shot" : WavefrontSensorInitializationFile.PAUSE_AFTER_SHOT,
        "pixel_format" : WavefrontSensorInitializationFile.PIXEL_FORMAT,
        "index_digits" : WavefrontSensorInitializationFile.INDEX_DIGITS, 
        "is_stream_available" : WavefrontSensorInitializationFile.IS_STREAM_AVAILABLE, 
        "transpose_stream_ima" : WavefrontSensorInitializationFile.TRANSPOSE_STREAM_IMAGE, 
        "pixel_size" : WavefrontSensorInitializationFile.PIXEL_SIZE, 
        "detector_resolution" : WavefrontSensorInitializationFile.DETECTOR_RESOLUTION, 
        "cam_pixel_format" : WavefrontSensorInitializationFile.CAM_PIXEL_FORMAT, 
        "cam_acquire" : WavefrontSensorInitializationFile.CAM_ACQUIRE, 
        "cam_exposure_time" : WavefrontSensorInitializationFile.CAM_EXPOSURE_TIME, 
        "cam_image_mode" : WavefrontSensorInitializationFile.CAM_IMAGE_MODE, 
        "tiff_enable_callback" : WavefrontSensorInitializationFile.TIFF_ENABLE_CALLBACKS, 
        "tiff_filename" : WavefrontSensorInitializationFile.TIFF_FILENAME, 
        "tiff_filepath" : WavefrontSensorInitializationFile.TIFF_FILEPATH, 
        "tiff_filenumber" : WavefrontSensorInitializationFile.TIFF_FILENUMBER, 
        "tiff_autosave" : WavefrontSensorInitializationFile.TIFF_AUTOSAVE, 
        "tiff_savefile" : WavefrontSensorInitializationFile.TIFF_SAVEFILE, 
        "tiff_autoincrement" : WavefrontSensorInitializationFile.TIFF_AUTOINCREMENT, 
        "pva_image" : WavefrontSensorInitializationFile.PVA_IMAGE
    }

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
            "image_ops" : WavefrontAnalyzerModule.IMAGE_OPS,
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
            "crop_v" : WavefrontAnalyzerModule.CROP_V,
            "crop_h" : WavefrontAnalyzerModule.CROP_H,
            "crop_shift_v" : WavefrontAnalyzerModule.CROP_SHIFT_V,
            "crop_shift_h" : WavefrontAnalyzerModule.CROP_SHIFT_H,
            "distance" : WavefrontAnalyzerModule.DISTANCE,
            "distance_v" : WavefrontAnalyzerModule.DISTANCE_V,
            "distance_h" : WavefrontAnalyzerModule.DISTANCE_H,
            "delta_f_v" : WavefrontAnalyzerModule.DELTA_F_V,
            "delta_f_h" : WavefrontAnalyzerModule.DELTA_F_H,
            "rms_range_v" : WavefrontAnalyzerModule.RMS_RANGE_V,
            "rms_range_h" : WavefrontAnalyzerModule.RMS_RANGE_H,
            "magnification_v" : WavefrontAnalyzerModule.MAGNIFICATION_V,
            "magnification_h" : WavefrontAnalyzerModule.MAGNIFICATION_H,
            "shift_half_pixel" : WavefrontAnalyzerModule.SHIFT_HALF_PIXEL,
            "scan_best_focus" : WavefrontAnalyzerModule.SCAN_BEST_FOCUS,
            "use_fit" : WavefrontAnalyzerModule.USE_FIT,
            "best_focus_from" : WavefrontAnalyzerModule.BEST_FOCUS_FROM,
            "best_focus_scan_range" : WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE,
            "best_focus_scan_range_v" : WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE_V,
            "best_focus_scan_range_h" : WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE_H,
        }
    }

    # Here GUI specific ini

    wavefront_sensor_image_directory       = ini.get_string_from_ini("Wavefront-Sensor",   "Wavefront-Sensor-Image-Directory", default=os.path.abspath(os.path.join(os.path.curdir, "wf_images")))
    wavefront_sensor_image_directory_batch = ini.get_string_from_ini("Wavefront-Sensor",   "Wavefront-Sensor-Image-Directory-Batch", default=os.path.abspath(os.path.join(os.path.curdir, "wf_images")))
    save_images                      = ini.get_boolean_from_ini("Wavefront-Analyzer", "Save-Images", default=True)
    plot_raw_image                   = ini.get_boolean_from_ini("Wavefront-Analyzer", "Plot-Raw_image", default=True)
    data_from                        = ini.get_int_from_ini("Wavefront-Analyzer", "Data-From", default=1) # file
    bp_calibration_mode              = ini.get_boolean_from_ini("Wavefront-Analyzer", "BP-Calibration-Mode", default=False)

    return ScriptData(wavefront_sensor_image_directory=wavefront_sensor_image_directory,
                      wavefront_sensor_image_directory_batch=wavefront_sensor_image_directory_batch,
                      save_images=save_images,
                      plot_raw_image=plot_raw_image,
                      data_from=data_from,
                      bp_calibration_mode=bp_calibration_mode,
                      wavefront_sensor_configuration=wavefront_sensor_configuration,
                      wavefront_analyzer_configuration=wavefront_analyzer_configuration)


def set_ini_from_initialization_parameters(initialization_parameters: ScriptData, ini: IniFacade):
    # -----------------------------------------------------
    # Wavefront Sensor

    wavefront_sensor_configuration   = initialization_parameters.get_parameter("wavefront_sensor_configuration")

    WavefrontSensorInitializationFile.SEND_STOP_COMMAND      = wavefront_sensor_configuration["send_stop_command"]
    WavefrontSensorInitializationFile.SEND_SAVE_COMMAND      = wavefront_sensor_configuration["send_save_command"]
    WavefrontSensorInitializationFile.REMOVE_IMAGE           = wavefront_sensor_configuration["remove_image"]
    WavefrontSensorInitializationFile.WAIT_TIME              = wavefront_sensor_configuration["wait_time"]
    WavefrontSensorInitializationFile.EXPOSURE_TIME          = wavefront_sensor_configuration["exposure_time"]
    WavefrontSensorInitializationFile.PAUSE_AFTER_SHOT       = wavefront_sensor_configuration["pause_after_shot"]
    WavefrontSensorInitializationFile.PIXEL_FORMAT           = wavefront_sensor_configuration["pixel_format"]
    WavefrontSensorInitializationFile.INDEX_DIGITS           = wavefront_sensor_configuration["index_digits"]
    WavefrontSensorInitializationFile.IS_STREAM_AVAILABLE    = wavefront_sensor_configuration["is_stream_available"]
    WavefrontSensorInitializationFile.TRANSPOSE_STREAM_IMAGE = wavefront_sensor_configuration["transpose_stream_ima"]
    WavefrontSensorInitializationFile.PIXEL_SIZE             = wavefront_sensor_configuration["pixel_size"]
    WavefrontSensorInitializationFile.DETECTOR_RESOLUTION    = wavefront_sensor_configuration["detector_resolution"]
    WavefrontSensorInitializationFile.CAM_PIXEL_FORMAT       = wavefront_sensor_configuration["cam_pixel_format"]
    WavefrontSensorInitializationFile.CAM_ACQUIRE            = wavefront_sensor_configuration["cam_acquire"]
    WavefrontSensorInitializationFile.CAM_EXPOSURE_TIME      = wavefront_sensor_configuration["cam_exposure_time"]
    WavefrontSensorInitializationFile.CAM_IMAGE_MODE         = wavefront_sensor_configuration["cam_image_mode"]
    WavefrontSensorInitializationFile.TIFF_ENABLE_CALLBACKS  = wavefront_sensor_configuration["tiff_enable_callback"]
    WavefrontSensorInitializationFile.TIFF_FILENAME          = wavefront_sensor_configuration["tiff_filename"]
    WavefrontSensorInitializationFile.TIFF_FILEPATH          = wavefront_sensor_configuration["tiff_filepath"]
    WavefrontSensorInitializationFile.TIFF_FILENUMBER        = wavefront_sensor_configuration["tiff_filenumber"]
    WavefrontSensorInitializationFile.TIFF_AUTOSAVE          = wavefront_sensor_configuration["tiff_autosave"]
    WavefrontSensorInitializationFile.TIFF_SAVEFILE          = wavefront_sensor_configuration["tiff_savefile"]
    WavefrontSensorInitializationFile.TIFF_AUTOINCREMENT     = wavefront_sensor_configuration["tiff_autoincrement"]
    WavefrontSensorInitializationFile.PVA_IMAGE              = wavefront_sensor_configuration["pva_image"]

    WavefrontSensorInitializationFile.store()
    
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
    WavefrontAnalyzerModule.IMAGE_OPS = data_analysis_configuration["image_ops"]             
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
    WavefrontAnalyzerModule.SIGMA_INTENSITY = back_propagation_configuration["sigma_intensity"]
    WavefrontAnalyzerModule.SMOOTH_PHASE = back_propagation_configuration["smooth_phase"]
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
    WavefrontAnalyzerModule.RMS_RANGE_V = back_propagation_configuration["rms_range_v"]      
    WavefrontAnalyzerModule.RMS_RANGE_H = back_propagation_configuration["rms_range_h"]      
    WavefrontAnalyzerModule.MAGNIFICATION_V = back_propagation_configuration["magnification_v"]   
    WavefrontAnalyzerModule.MAGNIFICATION_H = back_propagation_configuration["magnification_h"]   
    WavefrontAnalyzerModule.SHIFT_HALF_PIXEL = back_propagation_configuration["shift_half_pixel"] 
    WavefrontAnalyzerModule.SCAN_BEST_FOCUS = back_propagation_configuration["scan_best_focus"]   
    WavefrontAnalyzerModule.USE_FIT = back_propagation_configuration["use_fit"]                  
    WavefrontAnalyzerModule.BEST_FOCUS_FROM = back_propagation_configuration["best_focus_from"]   
    WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE = back_propagation_configuration["best_focus_scan_range"]     
    WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE_V = back_propagation_configuration["best_focus_scan_range_v"]  
    WavefrontAnalyzerModule.BEST_FOCUS_SCAN_RANGE_H = back_propagation_configuration["best_focus_scan_range_h"]  
    
    WavefrontAnalyzerModule.store()
    
    # Here GUI specific ini
    # TBD

    ini.set_value_at_ini("Wavefront-Sensor", "Wavefront-Sensor-Image-Directory", initialization_parameters.get_parameter("wavefront_sensor_image_directory"))
    ini.set_value_at_ini("Wavefront-Sensor", "Wavefront-Sensor-Image-Directory-Batch", initialization_parameters.get_parameter("wavefront_sensor_image_directory_batch"))
    ini.set_value_at_ini("Wavefront-Analyzer", "Save-Images", value=initialization_parameters.get_parameter("save_images"))
    ini.set_value_at_ini("Wavefront-Analyzer", "Plot-Raw_image", value=initialization_parameters.get_parameter("plot_raw_image"))
    ini.set_value_at_ini("Wavefront-Analyzer", "Data-From", value=initialization_parameters.get_parameter("data_from"))
    ini.set_value_at_ini("Wavefront-Analyzer", "BP-Calibration-Mode", value=initialization_parameters.get_parameter("bp_calibration_mode"))

    ini.push()