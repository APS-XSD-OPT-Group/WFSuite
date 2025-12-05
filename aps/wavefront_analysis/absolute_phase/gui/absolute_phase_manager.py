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
import pathlib

from PyQt5.QtCore import pyqtSignal

from aps.common.scripts.generic_process_manager import GenericProcessManager
from aps.common.widgets.context_widget import PlottingProperties, DefaultMainWindow
from aps.common.plotter import get_registered_plotter_instance
from aps.common.initializer import get_registered_ini_instance
from aps.common.logger import get_registered_logger_instance
from aps.common.scripts.script_data import ScriptData
from aps.common.plot.event_dispatcher import Receiver, Sender

from aps.wavefront_analysis.absolute_phase.factory import create_absolute_phase_analyzer
from aps.wavefront_analysis.absolute_phase.absolute_phase_analyzer import ProcessingMode

from aps.wavefront_analysis.absolute_phase.gui.absolute_phase_manager_initialization import generate_initialization_parameters_from_ini, set_ini_from_initialization_parameters
from aps.wavefront_analysis.absolute_phase.gui.absolute_phase_widget import AbsolutePhaseWidget
from aps.wavefront_analysis.common.gui.read_image_file_widget import PlotImageFile

from aps.wavefront_analysis.driver.wavefront_sensor import get_image_data, get_image_file_path
import aps.wavefront_analysis.driver.wavefront_sensor as ws

APPLICATION_NAME = "Absolute Phase"

INITIALIZATION_PARAMETERS_KEY  = APPLICATION_NAME + " Manager: Initialization"
SHOW_ABSOLUTE_PHASE            = APPLICATION_NAME + " Manager: Show Manager"
READ_IMAGE_FILE                = APPLICATION_NAME + " Manager: Show Image"

class IAbsolutePhaseManager(GenericProcessManager):
    def activate_absolute_phase_manager(self, plotting_properties=PlottingProperties(), **kwargs): raise NotImplementedError()
    def take_shot(self): raise NotImplementedError() # delegated
    def take_shot_as_flat_image(self): raise NotImplementedError() # delegated
    def read_from_file(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError() # delegated
    def generate_mask(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def process_image(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def back_propagate(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()

def create_absolute_phase_manager(**kwargs): return _AbsolutePhaseManager(**kwargs)

class _AbsolutePhaseManager(IAbsolutePhaseManager, Receiver, Sender):
    take_shot_sent                      = pyqtSignal(str)
    take_shot_as_flat_image_sent        = pyqtSignal(str)
    read_image_from_file_sent           = pyqtSignal(str)
    image_files_parameters_changed_sent = pyqtSignal(str, dict)

    crop_changed_received      = pyqtSignal(list)
    close_application_received = pyqtSignal()

    def __init__(self, **kwargs):
        super().__init__()

        self.reload_utils()

        self.__log_stream_widget       = kwargs.get("log_stream_widget", None)
        self.__working_directory       = kwargs.get("working_directory")

        self.__wavefront_sensor  = None
        self.__absolute_phase_analyzer = None

        self.__unique_id = None

    def reload_utils(self):
        self.__plotter = get_registered_plotter_instance(application_name=APPLICATION_NAME)
        self.__logger  = get_registered_logger_instance(application_name=APPLICATION_NAME)
        self.__ini     = get_registered_ini_instance(application_name=APPLICATION_NAME)

    def get_delegated_signals(self):
        return {
            "take_shot":                      self.take_shot_sent,
            "take_shot_as_flat_image":        self.take_shot_as_flat_image_sent,
            "read_image_from_file":           self.read_image_from_file_sent,
            "image_files_parameters_changed": self.image_files_parameters_changed_sent,
        }

    def get_delegate_signals(self):
        return {
            "crop_changed":         self.crop_changed_received,
            "close_absolute_phase": self.close_application_received
        }

    def activate_absolute_phase_manager(self, plotting_properties=PlottingProperties(), **kwargs):
        initialization_parameters = generate_initialization_parameters_from_ini(ini=self.__ini)

        if self.__plotter.is_active():
            if self.__unique_id is None:
                add_context_label = plotting_properties.get_parameter("add_context_label", False)
                use_unique_id     = plotting_properties.get_parameter("use_unique_id", True)

                unique_id = self.__plotter.register_context_window(SHOW_ABSOLUTE_PHASE,
                                                                   context_window=DefaultMainWindow(title=SHOW_ABSOLUTE_PHASE),
                                                                   use_unique_id=use_unique_id)

                self.__plotter.push_plot_on_context(SHOW_ABSOLUTE_PHASE, AbsolutePhaseWidget, unique_id,
                                                    log_stream_widget=self.__log_stream_widget,
                                                    working_directory=self.__working_directory,
                                                    initialization_parameters=initialization_parameters,
                                                    close_method=self.close,
                                                    image_files_parameters_changed_method=self.image_files_parameters_changed,
                                                    crop_changed_signal=self.crop_changed_received,
                                                    close_application_signal=self.close_application_received,
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

                self.__unique_id = unique_id
            else:
                self.__plotter.show_context_window(SHOW_ABSOLUTE_PHASE, self.__unique_id)
        else:
            action = kwargs.get("ACTION", None)
            if action is None: raise ValueError("Batch Mode without specified action ( use -a<ACTION>)")

            if "PIS" == str(action).upper():
                self.__check_absolute_phase_analyzer(initialization_parameters, batch_mode=True)

                wavefront_sensor_mode = initialization_parameters.get_parameter("wavefront_sensor_mode", 0)
                pixel_size_type       = initialization_parameters.get_parameter("pixel_size_type", 0)

                if wavefront_sensor_mode == 1 and pixel_size_type == 1: pixel_size = initialization_parameters.get_parameter("pixel_size_custom", ws.PIXEL_SIZE)
                else:                                                   pixel_size = ws.PIXEL_SIZE

                absolute_phase_analyzer_configuration = initialization_parameters.get_parameter("absolute_phase_analyzer_configuration")
                data_analysis_configuration = absolute_phase_analyzer_configuration["data_analysis"]

                self.__absolute_phase_analyzer.process_images(mode=ProcessingMode.BATCH,
                                                         n_threads=data_analysis_configuration.get("n_cores"),
                                                         pixel_size=pixel_size,
                                                         use_dark=initialization_parameters.get_parameter("use_dark", False),
                                                         use_flat=initialization_parameters.get_parameter("use_flat", False),
                                                         save_images=initialization_parameters.get_parameter("save_result", True))
            else:
                raise ValueError(f"Batch Mode: action not recognized {action}")

        return self.__unique_id

    def close(self, initialization_parameters: ScriptData):
        set_ini_from_initialization_parameters(initialization_parameters, self.__ini)
        self.__ini.push()
        print("Absolute Phase Manager Configuration saved")

        if self.__plotter.is_active():
            self.__plotter.close_context_window(context_key=SHOW_ABSOLUTE_PHASE, unique_id=self.__unique_id)
            self.__unique_id = None

    def image_files_parameters_changed(self, initialization_parameters: ScriptData):
        set_ini_from_initialization_parameters(initialization_parameters, self.__ini)
        self.__ini.push()

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

    def read_image_from_file(self, initialization_parameters: ScriptData = None, **kwargs):
        if initialization_parameters is None:
            self.read_image_from_file_sent.emit("read_image_from_file")
        else:
            data_collection_directory = initialization_parameters.get_parameter("image_directory")
            file_name_type            = initialization_parameters.get_parameter("file_name_type")
            file_name_prefix          = initialization_parameters.get_parameter("file_name_prefix_custom") if file_name_type == 1 else None
            image_index               = initialization_parameters.get_parameter("image_index")
            index_digits              = initialization_parameters.get_parameter("index_digits_custom") if file_name_type == 1 else None
            plot_rebinning_factor     = initialization_parameters.get_parameter("plot_rebinning_factor")

            wavefront_sensor_mode = initialization_parameters.get_parameter("wavefront_sensor_mode", 0)
            pixel_size_type       = initialization_parameters.get_parameter("pixel_size_type", 0)
            pixel_size_custom     = initialization_parameters.get_parameter("pixel_size_custom", ws.PIXEL_SIZE)

            if wavefront_sensor_mode == 1 and pixel_size_type == 1: pixel_size = pixel_size_custom
            else:                                                   pixel_size = ws.PIXEL_SIZE

            image, h_coord, v_coord = get_image_data(measurement_directory=data_collection_directory,
                                                     file_name_prefix=file_name_prefix,
                                                     image_index=image_index,
                                                     index_digits=index_digits,
                                                     image_ops=[],
                                                     pixel_size=pixel_size)

            file_name = get_image_file_path(measurement_directory=data_collection_directory,
                                            file_name_prefix=file_name_prefix,
                                            image_index=image_index,
                                            index_digits=index_digits)

            figure_name = pathlib.Path(file_name).with_suffix('')

            if self.__plotter.is_active():
                unique_id = self.__plotter.register_context_window(READ_IMAGE_FILE,
                                                                   context_window=DefaultMainWindow(title=READ_IMAGE_FILE),
                                                                   use_unique_id=True)
                self.__plotter.push_plot_on_context(READ_IMAGE_FILE, PlotImageFile, unique_id,
                                                    image=image,
                                                    h_coord=h_coord,
                                                    v_coord=v_coord,
                                                    figure_name=figure_name,
                                                    pixel_size=pixel_size,
                                                    plot_rebinning_factor=plot_rebinning_factor,
                                                    allows_saving=False,
                                                    **kwargs)
                self.__plotter.draw_context(READ_IMAGE_FILE, add_context_label=False, unique_id=unique_id)
                self.__plotter.show_context_window(READ_IMAGE_FILE, unique_id=unique_id)

    def generate_mask(self, initialization_parameters: ScriptData):
        self.__set_absolute_phase_analyzer_ready(initialization_parameters)

        wavefront_sensor_mode = initialization_parameters.get_parameter("wavefront_sensor_mode")
        image_index_for_mask  = 1 if wavefront_sensor_mode == 0 else initialization_parameters.get_parameter("image_index")
        kwargs                = {
            "use_dark": initialization_parameters.get_parameter("use_dark", False),
            "use_flat": initialization_parameters.get_parameter("use_flat", False),
        }
        if wavefront_sensor_mode == 0:
            kwargs["index_digits"] = initialization_parameters.get_parameter("index_digits")
        else:
            pixel_size_type = initialization_parameters.get_parameter("pixel_size_type", 0)

            if pixel_size_type == 1: pixel_size = initialization_parameters.get_parameter("pixel_size_custom", ws.PIXEL_SIZE)
            else:                    pixel_size = ws.PIXEL_SIZE
            kwargs["pixel_size"] = pixel_size

        image_transfer_matrix, is_new_mask = self.__absolute_phase_analyzer.generate_simulated_mask(image_index_for_mask=image_index_for_mask, **kwargs)

        if not is_new_mask: raise ValueError("Simulated Mask is already present in the Wavefront Image Directory")
        else:               return image_transfer_matrix

    def process_image(self, initialization_parameters: ScriptData):
        self.__set_absolute_phase_analyzer_ready(initialization_parameters)

        wavefront_sensor_mode = initialization_parameters.get_parameter("wavefront_sensor_mode")
        image_index           = 1 if wavefront_sensor_mode == 0 else initialization_parameters.get_parameter("image_index")
        kwargs                = {
            "use_dark": initialization_parameters.get_parameter("use_dark", False),
            "use_flat": initialization_parameters.get_parameter("use_flat", False),
            "save_images": initialization_parameters.get_parameter("save_result", True),
        }
        if wavefront_sensor_mode == 0:
            kwargs["index_digits"] = initialization_parameters.get_parameter("index_digits")
        else:
            pixel_size_type = initialization_parameters.get_parameter("pixel_size_type", 0)

            if pixel_size_type == 1:  pixel_size = initialization_parameters.get_parameter("pixel_size_custom", ws.PIXEL_SIZE)
            else:                     pixel_size = ws.PIXEL_SIZE
            kwargs["pixel_size"] = pixel_size

        return self.__absolute_phase_analyzer.process_image(image_index=image_index, **kwargs)

    def back_propagate(self, initialization_parameters: ScriptData, **kwargs):
        self.__set_absolute_phase_analyzer_ready(initialization_parameters)

        wavefront_sensor_mode = initialization_parameters.get_parameter("wavefront_sensor_mode")
        image_index           = 1 if wavefront_sensor_mode == 0 else initialization_parameters.get_parameter("image_index")
        kwargs                = {
            "verbose": True,
            "show_figure" : False,
            "save_result": initialization_parameters.get_parameter("save_result", True),
        }
        if wavefront_sensor_mode == 0:
            kwargs["index_digits"] = initialization_parameters.get_parameter("index_digits")
        else:
            pixel_size_type = initialization_parameters.get_parameter("pixel_size_type", 0)

            if pixel_size_type == 1:  pixel_size = initialization_parameters.get_parameter("pixel_size_custom", ws.PIXEL_SIZE)
            else:                     pixel_size = ws.PIXEL_SIZE
            kwargs["pixel_size"] = pixel_size

        return self.__absolute_phase_analyzer.back_propagate_wavefront(image_index=image_index, **kwargs)

    # --------------------------------------------------------------------------------------
    # PRIVATE METHODS
    # --------------------------------------------------------------------------------------

    def __set_absolute_phase_analyzer_ready(self, initialization_parameters: ScriptData):
        set_ini_from_initialization_parameters(initialization_parameters, ini=self.__ini)
        self.__check_absolute_phase_analyzer(initialization_parameters)
        self.__ini.push()

    def __check_absolute_phase_analyzer(self, initialization_parameters: ScriptData, batch_mode=False):
        data_analysis_configuration = initialization_parameters.get_parameter("absolute_phase_analyzer_configuration")["data_analysis"]

        data_collection_directory   = initialization_parameters.get_parameter("image_directory" if not batch_mode else "image_directory_batch")
        simulated_mask_directory    = initialization_parameters.get_parameter("simulated_mask_directory" if not batch_mode else "simulated_mask_directory_batch")
        energy                      = data_analysis_configuration['energy']
        file_name_type              = initialization_parameters.get_parameter("file_name_type")
        file_name_prefix            = initialization_parameters.get_parameter("file_name_prefix_custom") if file_name_type == 1 else None

        if self.__absolute_phase_analyzer is None: generate = True
        else:
            current_setup = self.__absolute_phase_analyzer.get_current_setup()
            generate = current_setup['data_collection_directory'] != data_collection_directory or \
                       current_setup['energy'] != energy or \
                       current_setup['simulated_mask_directory'] != simulated_mask_directory or \
                       (file_name_type == 1 and current_setup['file_name_prefix'] != file_name_prefix)

        if generate: self.__absolute_phase_analyzer = create_absolute_phase_analyzer(data_collection_directory=data_collection_directory,
                                                                           simulated_mask_directory=simulated_mask_directory,
                                                                           file_name_prefix=file_name_prefix,
                                                                           energy=energy)