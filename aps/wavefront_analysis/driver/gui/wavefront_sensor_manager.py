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
from AnyQt.QtCore import pyqtSignal

from aps.common.scripts.generic_process_manager import GenericProcessManager
from aps.common.widgets.context_widget import PlottingProperties, DefaultMainWindow
from aps.common.plotter import get_registered_plotter_instance
from aps.common.initializer import get_registered_ini_instance
from aps.common.logger import get_registered_logger_instance
from aps.common.scripts.script_data import ScriptData
from aps.common.plot.event_dispatcher import Receiver, Sender

from aps.wavefront_analysis.driver.factory import create_wavefront_sensor
from aps.wavefront_analysis.driver.wavefront_sensor import get_image_data

from aps.wavefront_analysis.driver.gui.wavefront_sensor_manager_initialization import generate_initialization_parameters_from_ini, set_ini_from_initialization_parameters
from aps.wavefront_analysis.driver.gui.wavefront_sensor_widget import WavefrontSensorWidget

APPLICATION_NAME = "Wavefront Sensor"

INITIALIZATION_PARAMETERS_KEY  = APPLICATION_NAME + " Manager: Initialization"
SHOW_WAVEFRONT_SENSOR            = APPLICATION_NAME + " Manager: Show Manager"

class IWavefrontSensorManager(GenericProcessManager):
    def activate_wavefront_sensor_manager(self, plotting_properties=PlottingProperties(), **kwargs): raise NotImplementedError()
    def take_shot(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def take_shot_as_flat_image(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def take_shot_and_generate_mask(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def take_shot_and_process_image(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def take_shot_and_back_propagate(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def read_from_file(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def generate_mask_from_file(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def process_image_from_file(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def back_propagate_from_file(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()

def create_wavefront_sensor_manager(**kwargs): return _WavefrontSensorManager(**kwargs)

class _WavefrontSensorManager(IWavefrontSensorManager, Receiver, Sender):
    interrupt = pyqtSignal()

    take_shot_received                      = pyqtSignal()
    take_shot_as_flat_image_received        = pyqtSignal()
    read_image_from_file_received           = pyqtSignal()
    image_files_parameters_changed_received = pyqtSignal(dict)
    close_application_received              = pyqtSignal()

    crop_changed_sent  = pyqtSignal(str, list)

    def __init__(self, **kwargs):
        super().__init__()

        self.reload_utils()

        self.__log_stream_widget       = kwargs.get("log_stream_widget", None)
        self.__working_directory       = kwargs.get("working_directory")

        self.__wavefront_sensor  = None

    def reload_utils(self):
        self.__plotter = get_registered_plotter_instance(application_name=APPLICATION_NAME)
        self.__logger  = get_registered_logger_instance(application_name=APPLICATION_NAME)
        self.__ini     = get_registered_ini_instance(application_name=APPLICATION_NAME)

    def get_delegate_signals(self):
        return {
            "take_shot":                      self.take_shot_received,
            "take_shot_as_flat_image":        self.take_shot_as_flat_image_received,
            "read_image_from_file":           self.read_image_from_file_received,
            "image_files_parameters_changed": self.image_files_parameters_changed_received,
            "close_wavefront_sensor":         self.close_application_received,

        }

    def get_delegated_signals(self):
        return {
            "crop_changed": self.crop_changed_sent,
        }

    def activate_wavefront_sensor_manager(self, plotting_properties=PlottingProperties(), **kwargs):
        initialization_parameters = generate_initialization_parameters_from_ini(ini=self.__ini)

        if self.__plotter.is_active():
            add_context_label = plotting_properties.get_parameter("add_context_label", False)
            use_unique_id     = plotting_properties.get_parameter("use_unique_id", False)

            self.__plotter.register_context_window(SHOW_WAVEFRONT_SENSOR,
                                                   context_window=DefaultMainWindow(title=SHOW_WAVEFRONT_SENSOR),
                                                   use_unique_id=use_unique_id)

            self.__plotter.push_plot_on_context(SHOW_WAVEFRONT_SENSOR, WavefrontSensorWidget, None,
                                                log_stream_widget=self.__log_stream_widget,
                                                working_directory=self.__working_directory,
                                                initialization_parameters=initialization_parameters,
                                                close_method=self.close,
                                                close_application_signal=self.close_application_received,
                                                connect_wavefront_sensor_method=self.connect_wavefront_sensor,
                                                save_configuration_method=self.save_configuration,
                                                crop_changed_method=self.crop_changed,
                                                take_shot_method=self.take_shot,
                                                take_shot_signal=self.take_shot_received,
                                                take_shot_as_flat_image_method=self.take_shot_as_flat_image,
                                                take_shot_as_flat_image_signal=self.take_shot_as_flat_image_received,
                                                read_image_from_file_method=self.read_image_from_file,
                                                read_image_from_file_signal=self.read_image_from_file_received,
                                                image_files_parameters_changed_method=self.image_files_parameters_changed,
                                                image_files_parameters_changed_signal=self.image_files_parameters_changed_received,
                                                allows_saving=False,
                                                **kwargs)

            self.__plotter.draw_context(SHOW_WAVEFRONT_SENSOR, add_context_label=add_context_label, unique_id=None, **kwargs)
            self.__plotter.show_context_window(SHOW_WAVEFRONT_SENSOR)
        else:
           raise ValueError(f"Batch Mode not possible")

    def save_configuration(self, initialization_parameters: ScriptData):
        set_ini_from_initialization_parameters(initialization_parameters, self.__ini)

    def connect_wavefront_sensor(self, initialization_parameters: ScriptData):
        if not self.__wavefront_sensor is None:
            try:    self.__wavefront_sensor.set_idle()
            except: pass
            try:    self.__wavefront_sensor.save_status()
            except: pass

        set_ini_from_initialization_parameters(initialization_parameters, self.__ini)

        try:
            self.__wavefront_sensor = create_wavefront_sensor() # the init will read the configuration and act accordingly.
        except Exception as e:
            self.__wavefront_sensor = None
            raise e

    def crop_changed(self, crop_array):
        self.crop_changed_sent.emit("crop_changed", crop_array)

    def take_shot(self, initialization_parameters: ScriptData, **kwargs):
        return self.__take_shot(initialization_parameters, flat=False)

    def take_shot_as_flat_image(self, initialization_parameters: ScriptData, **kwargs):
        return self.__take_shot(initialization_parameters, flat=True)

    def read_image_from_file(self, initialization_parameters: ScriptData):
        set_ini_from_initialization_parameters(initialization_parameters, self.__ini)

        image, hh, vv = get_image_data(image_index=1, units="mm")

        return hh, vv, image

    def image_files_parameters_changed(self, initialization_parameters: ScriptData):
        set_ini_from_initialization_parameters(initialization_parameters, self.__ini)

    def close(self, initialization_parameters: ScriptData):
        set_ini_from_initialization_parameters(initialization_parameters, self.__ini)
        self.__ini.push()
        print("Wavefront Sensor Configuration saved")

        if self.__plotter.is_active():
            self.__plotter.close_context_window(context_key=SHOW_WAVEFRONT_SENSOR)

    # --------------------------------------------------------------------------------------
    # PRIVATE METHODS
    # --------------------------------------------------------------------------------------

    def __take_shot(self, initialization_parameters: ScriptData, flat=False):
        if self.__wavefront_sensor is None: raise EnvironmentError("Wavefront Sensor is not connected")
        set_ini_from_initialization_parameters(initialization_parameters, ini=self.__ini)  # all arguments are read from the Ini

        try:
            self.__wavefront_sensor.collect_single_shot_image(image_index=1, flat=flat)
            image, h_coord, v_coord = self.__wavefront_sensor.get_image_data(image_index=1)

            try:    self.__wavefront_sensor.save_status()
            except: pass
            try:    self.__wavefront_sensor.end_collection()
            except: pass

            return h_coord, v_coord, image
        except Exception as e:
            try:    self.__wavefront_sensor.save_status()
            except: pass
            try:    self.__wavefront_sensor.end_collection()
            except: pass

            raise e