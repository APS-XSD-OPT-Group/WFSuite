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
import json
import os.path
import shutil

import numpy as np
from PyQt5.QtCore import pyqtSignal

from aps.common.scripts.generic_process_manager import GenericProcessManager
from aps.common.widgets.context_widget import PlottingProperties, DefaultMainWindow
from aps.common.plotter import get_registered_plotter_instance
from aps.common.initializer import get_registered_ini_instance
from aps.common.logger import get_registered_logger_instance
from aps.common.scripts.script_data import ScriptData
from aps.common.plot.event_dispatcher import Sender

from aps.wavefront_analysis.absolute_phase.factory import create_wavefront_analyzer
from aps.wavefront_analysis.absolute_phase.wavefront_analyzer import ProcessingMode

from aps.wavefront_analysis.absolute_phase.gui.absolute_phase_manager_initialization import generate_initialization_parameters_from_ini, set_ini_from_initialization_parameters
from aps.wavefront_analysis.absolute_phase.gui.absolute_phase_widget import AbsolutePhaseWidget

APPLICATION_NAME = "Absolute Phase"

INITIALIZATION_PARAMETERS_KEY  = APPLICATION_NAME + " Manager: Initialization"
SHOW_ABSOLUTE_PHASE            = APPLICATION_NAME + " Manager: Show Manager"

class IAbsolutePhaseManager(GenericProcessManager):
    def activate_absolute_phase_manager(self, plotting_properties=PlottingProperties(), **kwargs): raise NotImplementedError()
    def take_shot(self): raise NotImplementedError() # delegated
    def take_shot_as_flat_image(self): raise NotImplementedError() # delegated
    def read_from_file(self): raise NotImplementedError() # delegated
    def generate_mask(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def process_image(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def back_propagate(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()

def create_absolute_phase_manager(**kwargs): return _AbsolutePhaseManager(**kwargs)

class _AbsolutePhaseManager(IAbsolutePhaseManager, Sender):
    interrupt = pyqtSignal()

    take_shot_sent                      = pyqtSignal(str)
    take_shot_as_flat_image_sent        = pyqtSignal(str)
    read_image_from_file_sent           = pyqtSignal(str)
    image_files_parameters_changed_sent = pyqtSignal(str, dict)

    def __init__(self, **kwargs):
        super().__init__()

        self.reload_utils()

        self.__log_stream_widget       = kwargs.get("log_stream_widget", None)
        self.__working_directory       = kwargs.get("working_directory")

        self.__wavefront_sensor  = None
        self.__wavefront_analyzer = None

        self.__forms_registry = {}

    def reload_utils(self):
        self.__plotter = get_registered_plotter_instance(application_name=APPLICATION_NAME)
        self.__logger  = get_registered_logger_instance(application_name=APPLICATION_NAME)
        self.__ini     = get_registered_ini_instance(application_name=APPLICATION_NAME)

    def get_delegated_signals(self):
        return {
            "take_shot":               self.take_shot_sent,
            "take_shot_as_flat_image": self.take_shot_as_flat_image_sent,
            "read_image_from_file":    self.read_image_from_file_sent,
            "image_files_parameters_changed": self.image_files_parameters_changed_sent,
        }

    def activate_absolute_phase_manager(self, plotting_properties=PlottingProperties(), **kwargs):
        initialization_parameters = generate_initialization_parameters_from_ini(ini=self.__ini)
        unique_id = None

        if self.__plotter.is_active():
            add_context_label = plotting_properties.get_parameter("add_context_label", False)
            use_unique_id     = plotting_properties.get_parameter("use_unique_id", True)

            unique_id = self.__plotter.register_context_window(SHOW_ABSOLUTE_PHASE,
                                                               context_window=DefaultMainWindow(title=SHOW_ABSOLUTE_PHASE),
                                                               use_unique_id=use_unique_id)

            plot_widget_instance = self.__plotter.push_plot_on_context(SHOW_ABSOLUTE_PHASE, AbsolutePhaseWidget, unique_id,
                                                                       log_stream_widget=self.__log_stream_widget,
                                                                       working_directory=self.__working_directory,
                                                                       initialization_parameters=initialization_parameters,
                                                                       close_method=self.close,
                                                                       image_files_parameters_changed_method=self.image_files_parameters_changed,
                                                                       take_shot_method=self.take_shot,
                                                                       take_shot_as_flat_image_method=self.take_shot_as_flat_image,
                                                                       read_image_from_file_method=self.read_image_from_file,
                                                                       generate_mask_method=self.generate_mask,
                                                                       process_image_method=self.process_image,
                                                                       back_propagate_method=self.back_propagate,
                                                                       allows_saving=False,
                                                                       **kwargs)

            self.__plotter.draw_context(SHOW_ABSOLUTE_PHASE, add_context_label=add_context_label, unique_id=unique_id, **kwargs)
            self.__plotter.show_context_window(SHOW_ABSOLUTE_PHASE, unique_id)

            self.image_files_parameters_changed(initialization_parameters) # change directory at startup

            self.__forms_registry[plot_widget_instance] = unique_id
        else:
            action = kwargs.get("ACTION", None)

            if action is None: raise ValueError("Batch Mode without specified action ( use -a<ACTION>)")

            if "PIS" == str(action).upper():
                self.__check_wavefront_analyzer(initialization_parameters, batch_mode=True)

                wavefront_analyzer_configuration = initialization_parameters.get_parameter("wavefront_analyzer_configuration")
                data_analysis_configuration = wavefront_analyzer_configuration["data_analysis"]

                self.__wavefront_analyzer.process_images(mode=ProcessingMode.BATCH,
                                                         n_threads=data_analysis_configuration.get("n_cores"),
                                                         use_dark=initialization_parameters.get_parameter("use_dark", False),
                                                         use_flat=initialization_parameters.get_parameter("use_flat", False),
                                                         save_images=initialization_parameters.get_parameter("save_result", True))
            else:
                raise ValueError(f"Batch Mode: action not recognized {action}")

        return unique_id

    def close(self, initialization_parameters: ScriptData, plot_widget_instance):
        set_ini_from_initialization_parameters(initialization_parameters, self.__ini)
        self.__ini.push()

        if self.__plotter.is_active():
            unique_id = self.__forms_registry.get(plot_widget_instance, None)

            self.__plotter.get_context_container_widget(context_key=SHOW_ABSOLUTE_PHASE, unique_id=unique_id).parent().close()

    def image_files_parameters_changed(self, initialization_parameters: ScriptData):
        if initialization_parameters.get_parameter("wavefront_sensor_mode") == 0:
            parameters = {
                "file_name_type": initialization_parameters.get_parameter("file_name_type"),
                "file_name_prefix_custom": initialization_parameters.get_parameter("file_name_prefix_custom"),
                "index_digits_custom": initialization_parameters.get_parameter("index_digits_custom"),
                "image_directory": initialization_parameters.get_parameter("image_directory"),
            }

            self.image_files_parameters_changed_sent.emit("image_files_parameters_changed", parameters)

    def take_shot(self):
        self.take_shot_sent.emit("take_shot")

    def take_shot_as_flat_image(self):
        self.take_shot_as_flat_image_sent.emit("take_shot_as_flat_image")

    def read_image_from_file(self):
        self.read_image_from_file_sent.emit("read_image_from_file")

    def generate_mask(self, initialization_parameters: ScriptData):
        self.__set_wavefront_ready(initialization_parameters)

        wavefront_sensor_mode = initialization_parameters.get_parameter("wavefront_sensor_mode")
        image_index_for_mask  = 1 if wavefront_sensor_mode == 0 else initialization_parameters.get_parameter("image_index")
        kwargs                = {
            "use_dark": initialization_parameters.get_parameter("use_dark", False),
            "use_flat": initialization_parameters.get_parameter("use_flat", False),
        }
        if wavefront_sensor_mode == 0:
            kwargs["index_digits"] = initialization_parameters.get_parameter("index_digits")

        image_transfer_matrix, is_new_mask = self.__wavefront_analyzer.generate_simulated_mask(image_index_for_mask=image_index_for_mask, **kwargs)

        if not is_new_mask: raise ValueError("Simulated Mask is already present in the Wavefront Image Directory")
        else:               return image_transfer_matrix

    def process_image(self, initialization_parameters: ScriptData):
        self.__set_wavefront_ready(initialization_parameters)

        wavefront_sensor_mode = initialization_parameters.get_parameter("wavefront_sensor_mode")
        image_index           = 1 if wavefront_sensor_mode == 0 else initialization_parameters.get_parameter("image_index")
        kwargs                = {
            "use_dark": initialization_parameters.get_parameter("use_dark", False),
            "use_flat": initialization_parameters.get_parameter("use_flat", False),
            "save_images": initialization_parameters.get_parameter("save_result", True),
        }
        if wavefront_sensor_mode == 0:
            kwargs["index_digits"] = initialization_parameters.get_parameter("index_digits")

        return self.__wavefront_analyzer.process_image(image_index=image_index, **kwargs)

    def back_propagate(self, initialization_parameters: ScriptData, **kwargs):
        self.__set_wavefront_ready(initialization_parameters)

        wavefront_sensor_mode = initialization_parameters.get_parameter("wavefront_sensor_mode")
        image_index           = 1 if wavefront_sensor_mode == 0 else initialization_parameters.get_parameter("image_index")
        kwargs                = {
            "verbose": True,
            "show_figure" : False,
            "save_result": initialization_parameters.get_parameter("save_result", True),
        }
        if wavefront_sensor_mode == 0:
            kwargs["index_digits"] = initialization_parameters.get_parameter("index_digits")

        return self.__wavefront_analyzer.back_propagate_wavefront(image_index=image_index, **kwargs)

    # --------------------------------------------------------------------------------------
    # PRIVATE METHODS
    # --------------------------------------------------------------------------------------

    def __set_wavefront_ready(self, initialization_parameters: ScriptData):
        set_ini_from_initialization_parameters(initialization_parameters, ini=self.__ini)  # all arguments are read from the Ini
        self.__check_wavefront_analyzer(initialization_parameters)

    def __check_wavefront_analyzer(self, initialization_parameters: ScriptData, batch_mode=False):
        data_analysis_configuration = initialization_parameters.get_parameter("wavefront_analyzer_configuration")["data_analysis"]

        data_collection_directory   = initialization_parameters.get_parameter("image_directory" if not batch_mode else "image_directory_batch")
        simulated_mask_directory    = initialization_parameters.get_parameter("simulated_mask_directory" if not batch_mode else "simulated_mask_directory_batch")
        energy                      = data_analysis_configuration['energy']
        file_name_prefix_type       = initialization_parameters.get_parameter("file_name_prefix_type")
        file_name_prefix            = initialization_parameters.get_parameter("file_name_prefix_custom") if file_name_prefix_type == 1 else None

        if self.__wavefront_analyzer is None: generate = True
        else:
            current_setup = self.__wavefront_analyzer.get_current_setup()
            generate = current_setup['data_collection_directory'] != data_collection_directory or \
                       current_setup['energy'] != energy or \
                       current_setup['simulated_mask_directory'] != simulated_mask_directory or \
                       (file_name_prefix_type == 1 and current_setup['file_name_prefix'] != file_name_prefix)


        if generate: self.__wavefront_analyzer = create_wavefront_analyzer(data_collection_directory=data_collection_directory,
                                                                           simulated_mask_directory=simulated_mask_directory,
                                                                           file_name_prefix=file_name_prefix,
                                                                           energy=energy)