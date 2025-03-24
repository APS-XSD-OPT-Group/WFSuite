#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2022, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2022. UChicago Argonne, LLC. This software was produced       #
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
import numpy as np
from aps.common.driver.beamline.generic_camera import GenericCamera, CameraInitializationFile, get_default_file_name_prefix as __gdfnp, get_image_data as __gid

WAVEFRONT_SENSOR_STATUS_FILE = "wavefront_sensor_status.pkl"

class WavefrontSensorInitializationFile(CameraInitializationFile):
    @classmethod
    def initialize(cls):
        super().initialize(application_name="WAVEFRONT-SENSOR",
                           ini_file_name=".wavefront_sensor.json")

WavefrontSensorInitializationFile.initialize()
WavefrontSensorInitializationFile.store()

PIXEL_SIZE          = WavefrontSensorInitializationFile.PIXEL_SIZE
DETECTOR_RESOLUTION = WavefrontSensorInitializationFile.DETECTOR_RESOLUTION
INDEX_DIGITS        = WavefrontSensorInitializationFile.INDEX_DIGITS
IS_STREAM_AVAILABLE = WavefrontSensorInitializationFile.IS_STREAM_AVAILABLE

def get_default_file_name_prefix(exposure_time=None):
    return __gdfnp(exposure_time=(exposure_time if not exposure_time is None else WavefrontSensorInitializationFile.EXPOSURE_TIME))

def get_image_data(measurement_directory, file_name_prefix, image_index: int, **kwargs) -> [np.ndarray, np.ndarray, np.ndarray]:
    return __gid(measurement_directory=measurement_directory,
                 file_name_prefix=file_name_prefix,
                 image_index=image_index,
                 configuration_file=WavefrontSensorInitializationFile, **kwargs)

class WavefrontSensor(GenericCamera):
    def __init__(self,
                 measurement_directory: str = None,
                 exposure_time: int = None,
                 status_file: str = WAVEFRONT_SENSOR_STATUS_FILE,
                 file_name_prefix: str = None,
                 detector_delay: float = None,
                 mocking_mode: bool = False):
        super(WavefrontSensor, self).__init__(
            measurement_directory,
            exposure_time,
            status_file,
            file_name_prefix,
            detector_delay,
            mocking_mode,
            configuration_file=WavefrontSensorInitializationFile
        )