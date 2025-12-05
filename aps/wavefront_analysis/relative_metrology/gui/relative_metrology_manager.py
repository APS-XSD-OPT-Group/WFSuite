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
from aps.common.plot.event_dispatcher import Receiver

from aps.wavefront_analysis.relative_metrology.factory import create_relative_metrology_analyzer
from aps.wavefront_analysis.relative_metrology.relative_metrology_analyzer import ProcessingMode

from aps.wavefront_analysis.relative_metrology.gui.relative_metrology_manager_initialization import generate_initialization_parameters_from_ini, set_ini_from_initialization_parameters
from aps.wavefront_analysis.relative_metrology.gui.relative_metrology_widget import RelativeMetrologyWidget
from aps.wavefront_analysis.common.gui.read_image_file_widget import PlotImageFile

from aps.wavefront_analysis.driver.wavefront_sensor import get_image_data

APPLICATION_NAME = "Relative-Metrology"

INITIALIZATION_PARAMETERS_KEY  = APPLICATION_NAME + " Manager: Initialization"
SHOW_RELATIVE_METROLOGY                  = APPLICATION_NAME + " Manager: Show Manager"
RECROP_FROM_FILE               = APPLICATION_NAME + " Manager: Recrop"

class IRelativeMetrologyManager(GenericProcessManager):
    def activate_relative_metrology_manager(self, plotting_properties=PlottingProperties(), **kwargs): raise NotImplementedError()
    def process_image_WXST(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def process_images_WSVT(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def recrop_from_file(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()

def create_relative_metrology_manager(**kwargs): return _RelativeMetrologyManager(**kwargs)

class _RelativeMetrologyManager(IRelativeMetrologyManager, Receiver):
    close_application_received = pyqtSignal()

    def __init__(self, **kwargs):
        super().__init__()

        self.reload_utils()

        self.__log_stream_widget = kwargs.get("log_stream_widget", None)
        self.__working_directory = kwargs.get("working_directory")

        self.__relative_metrology_analyzer = None

        self.__unique_id = None

    def get_delegate_signals(self):
        return {
            "close_relative_metrology": self.close_application_received
        }

    def reload_utils(self):
        self.__plotter = get_registered_plotter_instance(application_name=APPLICATION_NAME)
        self.__logger  = get_registered_logger_instance(application_name=APPLICATION_NAME)
        self.__ini     = get_registered_ini_instance(application_name=APPLICATION_NAME)

    def activate_relative_metrology_manager(self, plotting_properties=PlottingProperties(), **kwargs):
        initialization_parameters = generate_initialization_parameters_from_ini(ini=self.__ini)

        if self.__plotter.is_active():
            if self.__unique_id is None:
                add_context_label = plotting_properties.get_parameter("add_context_label", False)
                use_unique_id     = plotting_properties.get_parameter("use_unique_id", True)

                unique_id = self.__plotter.register_context_window(SHOW_RELATIVE_METROLOGY,
                                                                   context_window=DefaultMainWindow(title=SHOW_RELATIVE_METROLOGY),
                                                                   use_unique_id=use_unique_id)

                self.__plotter.push_plot_on_context(SHOW_RELATIVE_METROLOGY, RelativeMetrologyWidget, unique_id,
                                                    log_stream_widget=self.__log_stream_widget,
                                                    working_directory=self.__working_directory,
                                                    initialization_parameters=initialization_parameters,
                                                    close_application_signal=self.close_application_received,
                                                    close_method=self.close,
                                                    process_image_WXST_method=self.process_image_WXST,
                                                    process_images_WSVT_method=self.process_images_WSVT,
                                                    recrop_from_file_method=self.recrop_from_file,
                                                    allows_saving=False,
                                                    **kwargs)

                self.__plotter.draw_context(SHOW_RELATIVE_METROLOGY, add_context_label=add_context_label, unique_id=unique_id, **kwargs)
                self.__plotter.show_context_window(SHOW_RELATIVE_METROLOGY, unique_id)

                self.__unique_id = unique_id
            else:
                self.__plotter.show_context_window(SHOW_RELATIVE_METROLOGY, self.__unique_id)
        else:
            action = kwargs.get("ACTION", None)

            if action is None: raise ValueError("Batch Mode without specified action ( use -a<ACTION>)")

            if "WXST" == str(action).upper():
                self.__check_relative_metrology_analyzer(initialization_parameters, batch_mode=True)

                relative_metrology_analyzer_configuration = initialization_parameters.get_parameter("relative_metrology_analyzer_configuration")
                common_configuration = relative_metrology_analyzer_configuration["common"]

                self.__relative_metrology_analyzer.process_image_WXST(mode=ProcessingMode.BATCH,
                                                                      n_threads=common_configuration.get("n_cores"))
            elif "WSVT" == str(action).upper():
                    self.__check_relative_metrology_analyzer(initialization_parameters, batch_mode=True)

                    relative_metrology_analyzer_configuration = initialization_parameters.get_parameter("relative_metrology_analyzer_configuration")
                    common_configuration = relative_metrology_analyzer_configuration["common"]

                    self.__relative_metrology_analyzer.process_images_WSVT(mode=ProcessingMode.BATCH,
                                                                           n_threads=common_configuration.get("n_cores"))
            else:
                raise ValueError(f"Batch Mode: action not recognized {action}")

        return self.__unique_id

    def close(self, initialization_parameters: ScriptData):
        set_ini_from_initialization_parameters(initialization_parameters, self.__ini)
        self.__ini.push()
        print("Relative Metrology Manager Configuration saved")

        if self.__plotter.is_active():
            self.__plotter.close_context_window(context_key=SHOW_RELATIVE_METROLOGY, unique_id=self.__unique_id)
            self.__unique_id = None

    def recrop_from_file(self, initialization_parameters: ScriptData = None, **kwargs):
        relative_metrology_analyzer_configuration = initialization_parameters.get_parameter("relative_metrology_analyzer_configuration")
        common_configuration = relative_metrology_analyzer_configuration["common"]

        file_name             = kwargs.get("crop_file_name")
        plot_rebinning_factor = initialization_parameters.get_parameter("plot_rebinning_factor")
        pixel_size            = common_configuration["pixel_size"]
        image, h_coord, v_coord = get_image_data(file_name=file_name)
        figure_name             = pathlib.Path(file_name).with_suffix('')

        if self.__plotter.is_active():
            unique_id = self.__plotter.register_context_window(RECROP_FROM_FILE,
                                                               context_window=DefaultMainWindow(title=RECROP_FROM_FILE),
                                                               use_unique_id=True)
            self.__plotter.push_plot_on_context(RECROP_FROM_FILE, PlotImageFile, unique_id,
                                                image=image,
                                                h_coord=h_coord,
                                                v_coord=v_coord,
                                                figure_name=figure_name,
                                                pixel_size=pixel_size,
                                                plot_rebinning_factor=plot_rebinning_factor,
                                                allows_saving=False,
                                                **kwargs)
            self.__plotter.draw_context(RECROP_FROM_FILE, add_context_label=False, unique_id=unique_id)
            self.__plotter.show_context_window(RECROP_FROM_FILE, unique_id=unique_id)

    def process_image_WXST(self, initialization_parameters: ScriptData, **kwargs):
        self.__set_relative_metrology_analyzer_ready(initialization_parameters)

        return self.__relative_metrology_analyzer.process_image_WXST(**kwargs)

    def process_images_WSVT(self, initialization_parameters: ScriptData, **kwargs):
        self.__set_relative_metrology_analyzer_ready(initialization_parameters)

        return self.__relative_metrology_analyzer.process_images_WSVT(**kwargs)

    # --------------------------------------------------------------------------------------
    # PRIVATE METHODS
    # --------------------------------------------------------------------------------------

    def __set_relative_metrology_analyzer_ready(self, initialization_parameters: ScriptData):
        set_ini_from_initialization_parameters(initialization_parameters, ini=self.__ini)
        self.__check_relative_metrology_analyzer(initialization_parameters)
        self.__ini.push()

    def __check_relative_metrology_analyzer(self, initialization_parameters: ScriptData, batch_mode=False):
        generate = self.__relative_metrology_analyzer is None

        if generate: self.__relative_metrology_analyzer = create_relative_metrology_analyzer()