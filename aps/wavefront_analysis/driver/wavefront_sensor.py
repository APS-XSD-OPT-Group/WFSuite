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
import time
from epics import PV
import pickle
from collections import OrderedDict

from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance

APPLICATION_NAME = "WAVEFRONT_ANALYSIS"

register_ini_instance(IniMode.LOCAL_JSON_FILE,
                      ini_file_name=".wavefront_sensor.json",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)

SEND_STOP_COMMAND = ini_file.get_boolean_from_ini(section="Execution", key="Send-Stop-Command", default=False)
WAIT_TIME         = ini_file.get_float_from_ini(  section="Execution", key="Wait-Time",         default=0.1)
EXPOSURE_TIME     = ini_file.get_float_from_ini(  section="Execution", key="Exposure-Time",     default=0.3)
PIXEL_FORMAT      = ini_file.get_int_from_ini(    section="Execution", key="Pixel-Format",      default=0)

CAM_PIXEL_FORMAT      = ini_file.get_string_from_ini(section="Epics", key="Cam-Pixel-Format",      default="dp_andor3_skylark:cam1:PixelFormat")
CAM_ACQUIRE           = ini_file.get_string_from_ini(section="Epics", key="Cam-Acquire",           default="dp_andor3_skylark:cam1:Acquire")
CAM_EXPOSURE_TIME     = ini_file.get_string_from_ini(section="Epics", key="Cam-Exposure-Time",     default="dp_andor3_skylark:cam1:AcquireTime")
CAM_IMAGE_MODE        = ini_file.get_string_from_ini(section="Epics", key="Cam-Image-Mode",        default="dp_andor3_skylark:cam1:ImageMode")
TIFF_ENABLE_CALLBACKS = ini_file.get_string_from_ini(section="Epics", key="Tiff-Enable-Callbacks", default="dp_andor3_skylark:TIFF1:EnableCallbacks")
TIFF_FILENAME         = ini_file.get_string_from_ini(section="Epics", key="Tiff-File-Name",        default="dp_andor3_skylark:TIFF1:FileName")
TIFF_FILEPATH         = ini_file.get_string_from_ini(section="Epics", key="Tiff-File-Path",        default="dp_andor3_skylark:TIFF1:FilePath")
TIFF_FILENUMBER       = ini_file.get_string_from_ini(section="Epics", key="Tiff-File-Number",      default="dp_andor3_skylark:TIFF1:FileNumber")
TIFF_AUTOSAVE         = ini_file.get_string_from_ini(section="Epics", key="Tiff-Auto-Save",        default="dp_andor3_skylark:TIFF1:AutoSave")
TIFF_SAVEFILE         = ini_file.get_string_from_ini(section="Epics", key="Tiff-Write-File",       default="dp_andor3_skylark:TIFF1:WriteFile")
TIFF_AUTOINCREMENT    = ini_file.get_string_from_ini(section="Epics", key="Tiff-Auto-Increment",   default="dp_andor3_skylark:TIFF1:AutoIncrement")

ini_file.set_value_at_ini(section="Execution",   key="Send-Stop-Command", value=SEND_STOP_COMMAND)
ini_file.set_value_at_ini(section="Execution",   key="Wait-Time",         value=WAIT_TIME)
ini_file.set_value_at_ini(section="Execution",   key="Exposure-Time",     value=EXPOSURE_TIME)
ini_file.set_value_at_ini(section="Execution",   key="Pixel-Format",      value=PIXEL_FORMAT)

ini_file.set_value_at_ini(section="Epics", key="Cam-Pixel-Format",      value=CAM_PIXEL_FORMAT)
ini_file.set_value_at_ini(section="Epics", key="Cam-Acquire",           value=CAM_ACQUIRE)
ini_file.set_value_at_ini(section="Epics", key="Cam-Exposure-Time",     value=CAM_EXPOSURE_TIME)
ini_file.set_value_at_ini(section="Epics", key="Cam-Image-Mode",        value=CAM_IMAGE_MODE)
ini_file.set_value_at_ini(section="Epics", key="Tiff-File-Name",        value=TIFF_FILENAME)
ini_file.set_value_at_ini(section="Epics", key="Tiff-File-Path",        value=TIFF_FILEPATH)
ini_file.set_value_at_ini(section="Epics", key="Tiff-File-Number",      value=TIFF_FILENUMBER)
ini_file.set_value_at_ini(section="Epics", key="Tiff-Auto-Save",        value=TIFF_AUTOSAVE)
ini_file.set_value_at_ini(section="Epics", key="Tiff-Write-File",       value=TIFF_SAVEFILE)
ini_file.set_value_at_ini(section="Epics", key="Tiff-Auto-Increment",   value=TIFF_AUTOINCREMENT)
ini_file.set_value_at_ini(section="Epics", key="Tiff-Enable-Callbacks", value=TIFF_ENABLE_CALLBACKS)

ini_file.push()

WAVEFRONT_SENSOR_STATUS_FILE = "image_collector_status.pkl"

def get_default_file_name_prefix(exposure_time=EXPOSURE_TIME):
    return "sample_" + str(int(exposure_time * 1000)) + "ms"

class WavefrontSensor():
    def __init__(self,
                 measurement_directory,
                 exposure_time=EXPOSURE_TIME,
                 file_name_prefix=None,
                 detector_delay=None,
                 mocking_mode=False):
        self.__exposure_time         = exposure_time
        self.__measurement_directory = measurement_directory
        self.__file_name_prefix      = get_default_file_name_prefix() if file_name_prefix is None else file_name_prefix
        self.__mocking_mode          = mocking_mode
        self.__send_stop_command     = SEND_STOP_COMMAND

        if not self.__mocking_mode:
            self.__PV_dict = {
                "cam_acquire"          : PV(CAM_ACQUIRE),       # 0="Done", 1="Acquire"
                "cam_exposure_time"    : PV(CAM_EXPOSURE_TIME),
                "cam_image_mode"       : PV(CAM_IMAGE_MODE),    # "0=Fixed" or "1=Continuous"
                "cam_pixel_format"     : PV(CAM_PIXEL_FORMAT),  # "0=Mono8, 5=Mono16, 8=Mono12"
                "tiff_filename"        : PV(TIFF_FILENAME),
                "tiff_filepath"        : PV(TIFF_FILEPATH),
                "tiff_filenumber"      : PV(TIFF_FILENUMBER),
                "tiff_autosave"        : PV(TIFF_AUTOSAVE),
                "tiff_savefile"        : PV(TIFF_SAVEFILE),
                "tiff_autoincrement"   : PV(TIFF_AUTOINCREMENT),
                "tiff_enable_callbacks": PV(TIFF_ENABLE_CALLBACKS)
            }

            if detector_delay is None:
                self.__has_delay = False
            else:
                self.__has_delay      = True
                self.__detector_delay = detector_delay

            self.__detector_stop()
            self.__set_defaults(1)
        else:
            print("ImageCollector initialized in Mocking Mode")

    def __to_dict(self):
        dictionary = OrderedDict()
        dictionary["cam_image_mode"]        = self.__PV_dict["cam_image_mode"].get()
        dictionary["cam_exposure_time"]     = self.__PV_dict["cam_exposure_time"].get()
        dictionary["cam_pixel_format"]      = self.__PV_dict["cam_pixel_format"].get()
        dictionary["tiff_filename"]         = self.__PV_dict["tiff_filename"].get()
        dictionary["tiff_filepath"]         = self.__PV_dict["tiff_filepath"].get()
        dictionary["tiff_filenumber"]       = self.__PV_dict["tiff_filenumber"].get()
        dictionary["tiff_autosave"]         = self.__PV_dict["tiff_autosave"].get()
        dictionary["tiff_savefile"]         = self.__PV_dict["tiff_savefile"].get()
        dictionary["tiff_autoincrement"]    = self.__PV_dict["tiff_autoincrement"].get()
        dictionary["tiff_enable_callbacks"] = self.__PV_dict["tiff_enable_callbacks"].get()

        return dictionary

    def __to_pickle_file(self):
        if not self.__mocking_mode:
            self.__detector_stop()

            file = open(WAVEFRONT_SENSOR_STATUS_FILE, 'wb')
            pickle.dump(self.__to_dict(), file)
            file.close()

    def __from_pickle_file(self):
        if not self.__mocking_mode:
            self.__detector_stop()

            file = open(WAVEFRONT_SENSOR_STATUS_FILE, 'rb')
            dictionary = pickle.load(file)
            file.close()

            self.__PV_dict["cam_image_mode"].put(       dictionary["cam_image_mode"])
            self.__PV_dict["cam_exposure_time"].put(    dictionary["cam_exposure_time"])
            self.__PV_dict["cam_pixel_format"].put(     dictionary["cam_pixel_format"])
            self.__PV_dict["tiff_filename"].put(        dictionary["tiff_filename"])
            self.__PV_dict["tiff_filepath"].put(        dictionary["tiff_filepath"])
            self.__PV_dict["tiff_filenumber"].put(      dictionary["tiff_filenumber"])
            self.__PV_dict["tiff_autosave"].put(        dictionary["tiff_autosave"])
            self.__PV_dict["tiff_savefile"].put(        dictionary["tiff_savefile"])
            self.__PV_dict["tiff_autoincrement"].put(   dictionary["tiff_autoincrement"])
            self.__PV_dict["tiff_enable_callbacks"].put(dictionary["tiff_enable_callbacks"])

    def save_status(self):
        self.__to_pickle_file()

    def restore_status(self):
        self.__detector_stop()  # 1 waiting time
        self.__from_pickle_file()

    def collect_single_shot_image(self, index=1):
        if not self.__mocking_mode:
            self.__initialize_current_image(index)
            self.__detector_acquire() # 2 waiting time + exposure time
        else:
            time.sleep(self.get_total_acquisition_time())
            print("Mocking Mode: collected image #" + str(index))

    def end_collection(self): # to be done at the end of the data collection
        if not self.__mocking_mode:
            self.__PV_dict["tiff_autosave"].put(1)
            self.__PV_dict["tiff_autoincrement"].put(0)

    def get_total_acquisition_time(self):
        return 3*WAIT_TIME + self.__exposure_time

    def __detector_acquire(self):
        self.__detector_start()
        self.__detector_delay()

    def __detector_start(self):
        self.__PV_dict["cam_acquire"].put(1)
        time.sleep(WAIT_TIME)

    def __detector_stop(self):
        if self.__send_stop_command: self.__PV_dict["cam_acquire"].put(0)
        time.sleep(WAIT_TIME)

    def __detector_acquiring(self):
        return self.__PV_dict["cam_acquire"].get() in (1, "Acquiring")

    def __detector_done(self):
        return self.__PV_dict["cam_acquire"].get() in (0, "Done")

    def __initialize_current_image(self, index):
        self.__detector_stop()  # 1 waiting time
        self.__set_defaults(index)

    def __set_defaults(self, index):
        self.__PV_dict["cam_image_mode"].put(0)
        self.__PV_dict["cam_pixel_format"].put(PIXEL_FORMAT)
        self.__PV_dict["cam_exposure_time"].put(self.__exposure_time)
        self.__PV_dict["tiff_enable_callbacks"].put(1)
        self.__PV_dict["tiff_filepath"].put(self.__measurement_directory)
        self.__PV_dict["tiff_autosave"].put(1)
        self.__PV_dict["tiff_autoincrement"].put(0)
        self.__PV_dict["tiff_filename"].put(self.__file_name_prefix)
        if index > 0: self.__PV_dict["tiff_filenumber"].put(index)

    def __detector_delay(self):
        if not self.__has_delay:
            time.sleep(self.__exposure_time + WAIT_TIME)
        else:
            time.sleep(WAIT_TIME)  # wait for the detector to start
            kk = 0
            while self.__detector_acquiring():
                time.sleep(self.__detector_delay)
                kk += self.__detector_delay
                if kk > 120:
                    self.__detector_stop()
                    time.sleep(1)
                    self.__detector_start()
                    time.sleep(1)
                    kk = 0
