
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

from aps.common.initializer import IniFacade
from aps.common.scripts.script_data import ScriptData

from aps.wavefront_analysis.driver.beamline.wavefront_sensor import WavefrontSensorInitializationFile

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
        "pva_image" : WavefrontSensorInitializationFile.PVA_IMAGE,
        "default_image_directory" : WavefrontSensorInitializationFile.DEFAULT_IMAGE_DIRECTORY,
        "current_image_directory" : WavefrontSensorInitializationFile.CURRENT_IMAGE_DIRECTORY,
        "data_from" : WavefrontSensorInitializationFile.DATA_FROM,
        "image_ops": WavefrontSensorInitializationFile.IMAGE_OPS,
    }

    plot_raw_image                   = ini.get_boolean_from_ini("Wavefront-Sensor", "Plot-Raw-Image", default=True)
    plot_rebinning_factor            = ini.get_int_from_ini(    "Wavefront-Sensor", "Plot-Rebinning-Factor", default=4)

    return ScriptData(wavefront_sensor_configuration=wavefront_sensor_configuration,
                      plot_raw_image=plot_raw_image,
                      plot_rebinning_factor=plot_rebinning_factor)


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
    WavefrontSensorInitializationFile.DEFAULT_IMAGE_DIRECTORY = wavefront_sensor_configuration["default_image_directory"]
    WavefrontSensorInitializationFile.CURRENT_IMAGE_DIRECTORY = wavefront_sensor_configuration["current_image_directory"]
    WavefrontSensorInitializationFile.DATA_FROM               = wavefront_sensor_configuration["data_from"]
    WavefrontSensorInitializationFile.IMAGE_OPS               = wavefront_sensor_configuration["image_ops"]

    WavefrontSensorInitializationFile.store()

    ini.set_value_at_ini("Wavefront-Sensor", "Plot-Raw-Image", value=initialization_parameters.get_parameter("plot_raw_image"))
    ini.set_value_at_ini("Wavefront-Sensor", "Plot-Rebinning-Factor", value=initialization_parameters.get_parameter("plot_rebinning_factor"))

    ini.push()
