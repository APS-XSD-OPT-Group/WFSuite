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
import sys

from AnyQt.QtCore import QTimer, pyqtSignal
from AnyQt.QtWidgets import QApplication

from aps.common.scripts.generic_process_manager import GenericProcessManager
from aps.common.widgets.context_widget import PlottingProperties, DefaultMainWindow
from aps.common.plotter import get_registered_plotter_instance
from aps.common.initializer import get_registered_ini_instance
from aps.common.logger import get_registered_logger_instance
from aps.common.scripts.script_data import ScriptData
from aps.common.plot.event_dispatcher import EventDispacther, Sender

from aps.wavefront_analysis.launcher.launcher_manager_initialization import generate_initialization_parameters_from_ini, set_ini_from_initialization_parameters
from aps.wavefront_analysis.launcher.launcher_widget import LauncherWidget

from aps.wavefront_analysis.absolute_phase.gui.main_absolute_phase import MainAbsolutePhase
from aps.wavefront_analysis.driver.gui.main_wavefront_sensor import MainWavefrontSensor
from aps.wavefront_analysis.relative_metrology.gui.main_relative_metrology import MainRelativeMetrology

APPLICATION_NAME = "Launcher"

SHOW_LAUNCHER = APPLICATION_NAME + ": Show Manager"

class ILauncherManager(GenericProcessManager):
    def activate_launcher_manager(self, plotting_properties=PlottingProperties(), **kwargs): raise NotImplementedError()

def create_launcher_manager(**kwargs): return _LauncherManager(**kwargs)

class _LauncherManager(ILauncherManager, Sender):
    close_absolute_phase_sent = pyqtSignal(str)
    close_relative_metrology_sent = pyqtSignal(str)
    close_wavefront_sensor_sent = pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__()

        self.reload_utils()

        self.__log_stream_widget = kwargs.get("log_stream_widget", None)
        self.__working_directory = kwargs.get("working_directory")
        self.__event_dispatcher  = EventDispacther()

        kwargs.pop("log_stream_widget")
        kwargs.pop("working_directory")

        sys_argv = kwargs.get("sys_argv", [])

        self.__wavefront_sensor_main   = MainWavefrontSensor(sys_argv=sys_argv, standalone=False, **kwargs)
        self.__absolute_phase_main     = MainAbsolutePhase(sys_argv=sys_argv, standalone=False, **kwargs)
        self.__relative_metrology_main = MainRelativeMetrology(sys_argv=sys_argv, standalone=False, **kwargs)
        self.__wavefront_sensor_main.run_script()
        self.__absolute_phase_main.run_script()
        self.__relative_metrology_main.run_script()

        wavefront_sensor_manager   = self.__wavefront_sensor_main.get_manager()
        absolute_phase_manager     = self.__absolute_phase_main.get_manager()
        relative_metrology_manager = self.__relative_metrology_main.get_manager()

        sender_signals   = absolute_phase_manager.get_delegated_signals() | \
                           wavefront_sensor_manager.get_delegated_signals() | \
                           self.get_delegated_signals()
        receiver_signals = absolute_phase_manager.get_delegate_signals() | \
                           wavefront_sensor_manager.get_delegate_signals() | \
                           relative_metrology_manager.get_delegate_signals()

        for signal_name in sender_signals.keys():
            self.__event_dispatcher.register_event_handler(sender_signal=sender_signals[signal_name],
                                                           sender_signal_name=signal_name,
                                                           receiver_signal=receiver_signals[signal_name])

    def reload_utils(self):
        self.__plotter = get_registered_plotter_instance(application_name=APPLICATION_NAME)
        self.__logger  = get_registered_logger_instance(application_name=APPLICATION_NAME)
        self.__ini     = get_registered_ini_instance(application_name=APPLICATION_NAME)

    def get_delegated_signals(self):
        return {
            "close_absolute_phase": self.close_absolute_phase_sent,
            "close_relative_metrology":       self.close_relative_metrology_sent,
            "close_wavefront_sensor" : self.close_wavefront_sensor_sent,
        }

    def activate_launcher_manager(self, plotting_properties=PlottingProperties(), **kwargs):
        initialization_parameters = generate_initialization_parameters_from_ini(ini=self.__ini)

        if self.__plotter.is_active():
            add_context_label = plotting_properties.get_parameter("add_context_label", False)
            use_unique_id     = plotting_properties.get_parameter("use_unique_id", False)

            self.__plotter.register_context_window(SHOW_LAUNCHER,
                                                   context_window=DefaultMainWindow(title=SHOW_LAUNCHER),
                                                   use_unique_id=use_unique_id)

            self.__plotter.push_plot_on_context(SHOW_LAUNCHER, LauncherWidget, None,
                                                log_stream_widget=self.__log_stream_widget,
                                                working_directory=self.__working_directory,
                                                initialization_parameters=initialization_parameters,
                                                open_absolute_phase_method=self.open_absolute_phase,
                                                open_relative_metrology_method=self.open_relative_metrology,
                                                close_method=self.close,
                                                allows_saving=False,
                                                **kwargs)

            self.__plotter.draw_context(SHOW_LAUNCHER, add_context_label=add_context_label, unique_id=None, **kwargs)
            self.__plotter.show_context_window(SHOW_LAUNCHER)

            self.__wavefront_sensor_main.activate_manager(**kwargs)
        else:
           raise ValueError(f"Batch Mode not possible")

    def open_absolute_phase(self, **kwargs):
        self.__absolute_phase_main.activate_manager(**kwargs)

    def open_relative_metrology(self, **kwargs):
        self.__relative_metrology_main.activate_manager(**kwargs)

    def close(self, initialization_parameters: ScriptData):
        set_ini_from_initialization_parameters(initialization_parameters, self.__ini)
        self.__ini.push()

        self.close_absolute_phase_sent.emit("close_absolute_phase")
        self.close_relative_metrology_sent.emit("close_relative_metrology")
        self.close_wavefront_sensor_sent.emit("close_wavefront_sensor")

        if self.__plotter.is_active(): self.__plotter.get_context_container_widget(context_key=SHOW_LAUNCHER).parent().close()

        QTimer.singleShot(0, lambda: QApplication.instance().exit(0))