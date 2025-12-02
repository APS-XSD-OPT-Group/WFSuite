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
import os
import pathlib

import numpy as np

from aps.common.initializer import get_registered_ini_instance
from aps.common.driver.beamline.generic_camera import GenericCamera, DataSource, CameraInitializationFile, \
    get_default_file_name_prefix as __gdfnp, \
    get_image_data as __gid,\
    get_image_file_path as __gifp, \
    get_file_name_prefix as __gfnp
from aps.wavefront_analysis.driver.mask_flipper import MaskFlipper


WAVEFRONT_SENSOR_STATUS_FILE = "wavefront_sensor_status.pkl"

class WavefrontSensorInitializationFile(CameraInitializationFile):
    @classmethod
    def initialize(cls):
        ini_file = super().initialize(application_name="WAVEFRONT-SENSOR", ini_file_name=".wavefront_sensor.json")

        cls.USE_FLIPPER = ini_file.get_boolean_from_ini(section="Execution", key="Use-Flipper", default=False)

        return ini_file

    @classmethod
    def store(cls, ini_file=None):
        if ini_file is None: ini_file = get_registered_ini_instance(cls.APPLICATION_NAME)

        ini_file.set_value_at_ini(section="Execution", key="Use-Flipper", value=cls.USE_FLIPPER)

        super().store(ini_file)


WavefrontSensorInitializationFile.store(WavefrontSensorInitializationFile.initialize())

def __getattr__(name):
    if   name == 'PIXEL_SIZE':          return WavefrontSensorInitializationFile.PIXEL_SIZE
    elif name == 'DETECTOR_RESOLUTION': return WavefrontSensorInitializationFile.DETECTOR_RESOLUTION
    elif name == 'INDEX_DIGITS':        return WavefrontSensorInitializationFile.INDEX_DIGITS
    else: raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def get_file_name_prefix(file_name_prefix_type=None, file_name_prefix_custom=None, exposure_time=None):
    if file_name_prefix_type is None:   file_name_prefix_type=WavefrontSensorInitializationFile.FILE_NAME_PREFIX_TYPE
    if file_name_prefix_custom is None: file_name_prefix_custom=WavefrontSensorInitializationFile.FILE_NAME_PREFIX_CUSTOM
    if exposure_time is None:           exposure_time = WavefrontSensorInitializationFile.EXPOSURE_TIME

    return __gfnp(file_name_prefix_type=file_name_prefix_type,
                  file_name_prefix_custom=file_name_prefix_custom,
                  exposure_time=exposure_time)

def get_default_file_name_prefix(exposure_time=None):
    return __gdfnp(exposure_time=(exposure_time if not exposure_time is None else WavefrontSensorInitializationFile.EXPOSURE_TIME))

def get_image_data(measurement_directory=None,
                   file_name_prefix=None,
                   image_index=1,
                   index_digits=None, **kwargs) -> [np.ndarray, np.ndarray, np.ndarray]:
    input_file_name = kwargs.get("file_name", None)

    if input_file_name is None:
        file_name = get_image_file_path(measurement_directory=measurement_directory,
                                        file_name_prefix=file_name_prefix,
                                        image_index=image_index,
                                        index_digits=index_digits,
                                        extension="hdf5",
                                        **kwargs)
        if os.path.exists(file_name): return __gid(file_name)
        else: # retrocompatibilty
            file_name_old = get_image_file_path(measurement_directory=measurement_directory,
                                                file_name_prefix=file_name_prefix,
                                                image_index=image_index,
                                                index_digits=index_digits,
                                                extension="tif",
                                                **kwargs)
            if os.path.exists(file_name_old):
                measurement_directory = measurement_directory if not measurement_directory is None else WavefrontSensorInitializationFile.CURRENT_IMAGE_DIRECTORY
                file_name_prefix      = file_name_prefix if not file_name_prefix is None else get_file_name_prefix()
                index_digits          = index_digits if not index_digits is None else WavefrontSensorInitializationFile.INDEX_DIGITS
                image_ops             = kwargs.pop("image_ops",  WavefrontSensorInitializationFile.IMAGE_OPS[DataSource.File])
                pixel_size            = kwargs.pop("pixel_size", WavefrontSensorInitializationFile.PIXEL_SIZE)

                image, h_coord, v_coord = GenericCamera._get_image_file_data(measurement_directory=measurement_directory,
                                                                             file_name_prefix=file_name_prefix,
                                                                             image_index=image_index,
                                                                             index_digits=index_digits,
                                                                             pixel_size=pixel_size,
                                                                             image_ops=image_ops,
                                                                             **kwargs)
                GenericCamera._store_image_data(h_coord, v_coord, image, file_name)

                return image, h_coord, v_coord
            else:
                raise IOError(f"File {file_name} not existing")
    else:
        if os.path.exists(input_file_name):
            if   pathlib.Path(input_file_name).suffix == ".hdf5": return __gid(input_file_name)
            elif pathlib.Path(input_file_name).suffix == ".tif":
                image_ops             = kwargs.pop("image_ops",  WavefrontSensorInitializationFile.IMAGE_OPS[DataSource.File])
                pixel_size            = kwargs.pop("pixel_size", WavefrontSensorInitializationFile.PIXEL_SIZE)

                image, h_coord, v_coord = GenericCamera._get_image_file_data(measurement_directory=None,
                                                                             file_name_prefix=None,
                                                                             image_index=None,
                                                                             index_digits=None,
                                                                             pixel_size=pixel_size,
                                                                             image_ops=image_ops,
                                                                             file_path=input_file_name,
                                                                             **kwargs)
                GenericCamera._store_image_data(h_coord, v_coord, image, pathlib.Path(input_file_name).with_suffix(".hdf5"))

                return image, h_coord, v_coord
        else:
            raise IOError(f"File {input_file_name} not existing")

def get_image_file_path(measurement_directory=None, file_name_prefix=None, image_index = 1, index_digits=None, extension="hdf5", **kwargs) -> str:
    measurement_directory = measurement_directory if not measurement_directory is None else WavefrontSensorInitializationFile.CURRENT_IMAGE_DIRECTORY
    index_digits          = index_digits if not index_digits is None else WavefrontSensorInitializationFile.INDEX_DIGITS
    file_name_prefix      = file_name_prefix if not file_name_prefix is None else get_file_name_prefix()

    return __gifp(measurement_directory=measurement_directory,
                  file_name_prefix=file_name_prefix,
                  image_index=image_index,
                  index_digits=index_digits,
                  extension=extension,
                  **kwargs)

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

        if WavefrontSensorInitializationFile.USE_FLIPPER: self.__mask_flipper = MaskFlipper()
        else:                                             self.__mask_flipper = None

    def collect_single_shot_image(self, image_index=1, **kwargs):
        if WavefrontSensorInitializationFile.USE_FLIPPER:
            if kwargs.get("flat", False): self.__mask_flipper.set_mask_off()
            else:                         self.__mask_flipper.set_mask_on()

        super(WavefrontSensor, self).collect_single_shot_image(image_index=image_index, **kwargs)
