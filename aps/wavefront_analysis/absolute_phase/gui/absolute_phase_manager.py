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
import sys
import copy

import time
import traceback
from PyQt5.QtCore import QThread, pyqtSignal, QObject

from aps.common.scripts.generic_process_manager import GenericProcessManager
from aps.common.widgets.context_widget import PlottingProperties, DefaultMainWindow
from aps.common.plotter import get_registered_plotter_instance
from aps.common.initializer import get_registered_ini_instance
from aps.common.logger import get_registered_logger_instance, LoggerColor
from aps.common.scripts.script_data import ScriptData
from aps.common.singleton import synchronized_method
from aps.common.io.printout import date_now_str, time_now_str
from aps.common.widgets.stream_proxy import StreamProxy

from aps.wavefront_analysis.absolute_phase.factory import create_wavefront_analyzer
from aps.wavefront_analysis.driver.factory import create_wavefront_sensor

from aps.wavefront_analysis.absolute_phase.gui.absolute_phase_manager_initialization import generate_initialization_parameters_from_ini, set_ini_from_initialization_parameters
from aps.wavefront_analysis.absolute_phase.gui.absolute_phase_widget import AbsolutePhaseWidget

APPLICATION_NAME = "Absolute Phase"

INITIALIZATION_PARAMETERS_KEY  = APPLICATION_NAME + " Manager: Initialization"
SHOW_ABSOLUTE_PHASE            = APPLICATION_NAME + " Manager: Show Manager"

class IAbsolutePhaseManager(GenericProcessManager):
    def show_absolute_phase_manager(self, plotting_properties=PlottingProperties(), **kwargs): raise NotImplementedError()
    def take_shot(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def take_shot_and_generate_mask(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def take_shot_and_process_image(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def take_shot_and_back_propagate(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def read_from_file(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def generate_mask_from_file(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def process_image_from_file(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()
    def back_propagate_from_file(self, initialization_parameters: ScriptData, **kwargs): raise NotImplementedError()

def create_absolute_phase_manager(**kwargs): return _AbsolutePhaseManager(**kwargs)

class _AbsolutePhaseManager(IAbsolutePhaseManager, QObject):
    interrupt         = pyqtSignal()
    analysis_completed = pyqtSignal()

    def __init__(self, **kwargs):
        super().__init__()

        self.reload_utils()

        self.__log_stream_widget       = kwargs.get("log_stream_widget", None)
        self.__working_directory       = kwargs.get("working_directory")

        self.__wavefront_sensor  = None
        self.__wavefron_analyzer = None

    def reload_utils(self):
        self.__plotter = get_registered_plotter_instance(application_name=APPLICATION_NAME)
        self.__logger  = get_registered_logger_instance(application_name=APPLICATION_NAME)
        self.__ini     = get_registered_ini_instance(application_name=APPLICATION_NAME)

    def show_absolute_phase_manager(self, plotting_properties=PlottingProperties(), **kwargs):
        initialization_parameters = generate_initialization_parameters_from_ini(ini=self.__ini)

        if self.__plotter.is_active():
            add_context_label = plotting_properties.get_parameter("add_context_label", False)
            use_unique_id     = plotting_properties.get_parameter("use_unique_id", False)

            self.__plotter.register_context_window(SHOW_ABSOLUTE_PHASE,
                                                   context_window=DefaultMainWindow(title=SHOW_ABSOLUTE_PHASE),
                                                   use_unique_id=use_unique_id)

            self.__plotter.push_plot_on_context(SHOW_ABSOLUTE_PHASE, AbsolutePhaseWidget, None,
                                                log_stream_widget=self.__log_stream_widget,
                                                working_directory=self.__working_directory,
                                                initialization_parameters=initialization_parameters,
                                                connect_wavefront_sensor_method=self.connect_wavefront_sensor,
                                                close_method=self.close,
                                                #abort_method=self.abort,
                                                #get_result_method=self.get_current_result,
                                                #get_initialization_parameters_method=self.get_current_initialization_parameters,
                                                #get_trial_method=self.get_trial,
                                                #get_pareto_trial_number_method=self.get_pareto_trial_number,
                                                #get_optimization_history_method=self.get_optimization_history,
                                                #finish_method=self.finish,
                                                #move_motors_to_trial_data_method=self.move_motors_to_trial_data,
                                                #set_current_motor_positions_as_initial_points=self.set_current_motor_positions_as_initial_points,
                                                #move_motors_to_initial_points=self.move_motors_to_initial_points,
                                                #transfer_current_position_to_beam_manager_search_space=self.transfer_current_position_to_beam_manager_search_space,
                                                #load_motors_info=self.load_motors_info,
                                                #save_motors_info=self.save_motors_info,
                                                allows_saving=False,
                                                **kwargs)
            self.analysis_completed.connect(getattr(self.__plotter.get_plots_of_context(SHOW_ABSOLUTE_PHASE)[0], "analysis_completed"))

            self.__plotter.draw_context(SHOW_ABSOLUTE_PHASE, add_context_label=add_context_label, unique_id=None, **kwargs)
            self.__plotter.show_context_window(SHOW_ABSOLUTE_PHASE)
        else:
            pass
            #self.start_ai_agent(initialization_parameters)

    def connect_wavefront_sensor(self, initialization_parameters: ScriptData):
        if not self.__wavefront_sensor is None:
            try:   self.__wavefront_sensor.set_idle()
            except: pass

        self.__wavefront_sensor = create_wavefront_sensor(measurement_directory=initialization_parameters.get_parameter("wavefront_sensor_image_directory"))

        try:   self.__wavefront_sensor.restore_status()
        except: pass

    def close(self, initialization_parameters):
        set_ini_from_initialization_parameters(initialization_parameters, self.__ini)
        self.__ini.push()

        if not self.__wavefront_sensor is None:
            try:    self.__wavefront_sensor.save_status()
            except: pass

        if self.__plotter.is_active(): self.__plotter.get_context_container_widget(context_key=SHOW_ABSOLUTE_PHASE).parent().close()

        sys.exit(0)