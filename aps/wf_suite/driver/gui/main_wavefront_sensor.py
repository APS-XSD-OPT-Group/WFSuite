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

from aps.common.initializer import get_registered_ini_instance, IniMode
from aps.common.plot.qt_application import get_registered_qt_application_instance
from aps.common.plotter import get_registered_plotter_instance
from aps.common.scripts.generic_qt_script import GenericQTScript
from aps.common.widgets.log_stream_widget import LogStreamWidget
from aps.common.logger import register_logger_pool_instance, register_logger_single_instance, DEFAULT_STREAM
from aps.common.io.printout import datetime_now_str

from aps.wf_suite.driver.gui.wavefront_sensor_manager import APPLICATION_NAME, SHOW_WAVEFRONT_SENSOR, create_wavefront_sensor_manager

class MainWavefrontSensor(GenericQTScript):
    SCRIPT_ID = "Wavefront-Sensor"

    def _parse_additional_parameters(self, **kwargs):
        __args = super(MainWavefrontSensor, self)._parse_additional_parameters(**kwargs)
        __args["INI_MODE"] = IniMode.LOCAL_JSON_FILE
        __args["STANDALONE"] = kwargs.get("standalone", True)

        return __args

    def _get_script_id(self): return MainWavefrontSensor.SCRIPT_ID
    def _get_ini_file_name(self): return ".GUI_wavefront_sensor.json"
    def _get_application_name(self): return APPLICATION_NAME
    def _get_script_package(self): return "aps.wf_suite"

    def get_manager(self):
        return self.__wavefront_sensor_manager

    def _run_script(self, **args):
        self.__standalone               = args.get("STANDALONE", False)
        self.__plotter                  = get_registered_plotter_instance(application_name=APPLICATION_NAME)
        self.__wavefront_sensor_manager = create_wavefront_sensor_manager(log_stream_widget=self._log_stream_widget,
                                                                          working_directory=self._working_directory)

        if self.__standalone: self.activate_manager(**args)

    def activate_manager(self, **args):
        # ==========================================================================
        # %% Initialization parameters
        # ==========================================================================

        self.__wavefront_sensor_manager.activate_wavefront_sensor_manager(**args)

        # ==========================================================================
        # %% Final Operations
        # ==========================================================================

        get_registered_ini_instance(self._get_application_name()).push()

        # ==========================================================================

        if self.__plotter.is_active():
            self.__plotter.raise_context_window(context_key=SHOW_WAVEFRONT_SENSOR, close_button=False, stay_on_top=False)
            if self.__standalone: get_registered_qt_application_instance().run_qt_application()

    def _parse_additional_sys_argument(self, sys_argument, args):
        if "-m" == sys_argument[:2]:   args["LOG_POOL"] = int(sys_argument[2:])

    def _help_additional_parameters(self):
        help = "  -m<use multiple loggers>\n"
        help += "   use multiple loggers:\n" + \
                "     0 on GUI only - Default value\n" + \
                "     1 on GUI and on File\n" + \
                "     2 on stdout - Default value when p=2,3\n" + \
                "     3 on stdout and on File\n"

        return help

    def _manage_working_directory(self, **args):
        self._working_directory = os.path.abspath(os.getcwd())

    def _register_logger_instance(self, logger_mode, application_name, **args):
        self._manage_working_directory(**args)

        log_stream_prefix = "wavefront_sensor"

        if args.get("LOG_POOL") is None: args["LOG_POOL"] = 0

        if args.get("PLOTTER_MODE") == 3 or args.get("PLOTTER_MODE") == 2:
            if   args.get("LOG_POOL") == 0: args["LOG_POOL"] = 2
            elif args.get("LOG_POOL") == 1: args["LOG_POOL"] = 3

        self._log_stream_widget = self._log_stream_file = self._log_stream_default = None

        if args.get("LOG_POOL") in [0, 1]: self._log_stream_widget = LogStreamWidget(width=850, height=400, color='\'light grey\'')
        if args.get("LOG_POOL") in [1, 3]: self._log_stream_file = open(log_stream_prefix + "_" + datetime_now_str() + ".log", "wt")
        if args.get("LOG_POOL") in [2, 3]: self._log_stream_default = DEFAULT_STREAM

        if args.get("LOG_POOL") == 0 or args.get("LOG_POOL") is None:
            print("Log Pool: GUI")
            register_logger_single_instance(stream=self._log_stream_widget, logger_mode=logger_mode, application_name=application_name)
        else:
            if args.get("LOG_POOL") == 1:
                print("Log Pool: GUI, File")
                register_logger_pool_instance(stream_list=[self._log_stream_widget, self._log_stream_file],
                                              logger_mode=logger_mode, application_name=application_name)
            elif args.get("LOG_POOL") == 2:
                print("Log Pool: StdOut")
                register_logger_single_instance(stream=self._log_stream_default,
                                                logger_mode=logger_mode, application_name=application_name)
            elif args.get("LOG_POOL") == 3:
                print("Log Pool: StdOut, File")
                register_logger_pool_instance(stream_list=[self._log_stream_default, self._log_stream_file],
                                              logger_mode=logger_mode, application_name=application_name)


if __name__=="__main__":
    MainWavefrontSensor().show_help()
