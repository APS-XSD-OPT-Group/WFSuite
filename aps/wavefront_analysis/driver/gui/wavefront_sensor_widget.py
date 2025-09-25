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
import copy
import sys

import numpy as np

from aps.common.plot import gui
from aps.common.plot.gui import MessageDialog
from aps.common.plot.splitter import ToggleSplitter, ToggleDirection
from aps.common.widgets.generic_widget import GenericWidget
from aps.common.widgets.congruence import *
from aps.common.scripts.script_data import ScriptData
from aps.common.utilities import list_to_string, string_to_list

from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from cmasher import cm as cmm

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QScrollArea
from PyQt5.QtCore import QRect, Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap

from aps.common.driver.beamline.generic_camera import get_data_from_int_to_string, get_data_from_string_to_int
from aps.wavefront_analysis.common.gui.util import ShowWaitDialog

import warnings
warnings.filterwarnings("ignore")

DEBUG_MODE = int(os.environ.get("DEBUG_MODE", 0)) == 1

class WavefrontSensorWidget(GenericWidget):
    wavefront_sensor_changed = pyqtSignal()

    def __init__(self, parent, application_name=None, **kwargs):
        self._log_stream_widget             = kwargs["log_stream_widget"]
        self._working_directory             = kwargs["working_directory"]
        self._initialization_parameters     = kwargs["initialization_parameters"]

        # METHODS
        self._connect_wavefront_sensor      = kwargs["connect_wavefront_sensor_method"]
        self._take_shot                     = kwargs["take_shot_method"]
        self._take_shot_as_flat_image       = kwargs["take_shot_as_flat_image_method"]
        self._read_image_from_file          = kwargs["read_image_from_file_method"]

        take_shot_signal                     = kwargs["take_shot_signal"]
        take_shot_as_flat_image_signal       = kwargs["take_shot_as_flat_image_signal"]
        read_image_from_file_signal          = kwargs["read_image_from_file_signal"]

        take_shot_signal.connect(self._take_shot_callback)
        take_shot_as_flat_image_signal.connect(self._take_shot_as_flat_image_callback)
        read_image_from_file_signal.connect(self._read_image_from_file_callback)

        self._set_values_from_initialization_parameters()

        icons_path = os.path.join(os.path.dirname(__import__("aps.wavefront_analysis.driver.gui", fromlist=[""]).__file__), 'icons')
        self.__ws_pixmaps = {
            "red": QPixmap(os.path.join(icons_path, "red_light.png")).scaled(25, 25, Qt.KeepAspectRatio, Qt.SmoothTransformation),
            "green": QPixmap(os.path.join(icons_path, "green_light.png")).scaled(25, 25, Qt.KeepAspectRatio, Qt.SmoothTransformation),
            "orange" : QPixmap(os.path.join(icons_path, "orange_light.png")).scaled(25, 25, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        }

        self.__is_wavefront_sensor_initialized = False

        super(WavefrontSensorWidget, self).__init__(parent=parent, application_name=application_name, **kwargs)

        self.wavefront_sensor_changed.connect(self._wavefront_sensor_changed)

    def _set_values_from_initialization_parameters(self):
        self.working_directory = self._working_directory

        initialization_parameters: ScriptData = self._initialization_parameters

        self.plot_raw_image                   = initialization_parameters.get_parameter("plot_raw_image", True)
        self.plot_rebinning_factor            = initialization_parameters.get_parameter("plot_rebinning_factor", 4)

        # -----------------------------------------------------
        # Wavefront Sensor

        wavefront_sensor_configuration = initialization_parameters.get_parameter("wavefront_sensor_configuration")
    
        self.send_stop_command = wavefront_sensor_configuration["send_stop_command"]
        self.send_save_command = wavefront_sensor_configuration["send_save_command"]
        self.remove_image = wavefront_sensor_configuration["remove_image"]
        self.wait_time = wavefront_sensor_configuration["wait_time"]
        self.exposure_time = wavefront_sensor_configuration["exposure_time"]
        self.pause_after_shot = wavefront_sensor_configuration["pause_after_shot"]
        self.pixel_format = wavefront_sensor_configuration["pixel_format"]
        self.index_digits = wavefront_sensor_configuration["index_digits"]
        self.is_stream_available = wavefront_sensor_configuration["is_stream_available"]
        self.pixel_size = wavefront_sensor_configuration["pixel_size"]
        self.detector_resolution = wavefront_sensor_configuration["detector_resolution"]
        self.cam_pixel_format = wavefront_sensor_configuration["cam_pixel_format"]
        self.cam_acquire = wavefront_sensor_configuration["cam_acquire"]
        self.cam_exposure_time = wavefront_sensor_configuration["cam_exposure_time"]
        self.cam_image_mode = wavefront_sensor_configuration["cam_image_mode"]
        self.tiff_enable_callbacks = wavefront_sensor_configuration["tiff_enable_callback"]
        self.tiff_filename = wavefront_sensor_configuration["tiff_filename"]
        self.tiff_filepath = wavefront_sensor_configuration["tiff_filepath"]
        self.tiff_filenumber = wavefront_sensor_configuration["tiff_filenumber"]
        self.tiff_autosave = wavefront_sensor_configuration["tiff_autosave"]
        self.tiff_savefile = wavefront_sensor_configuration["tiff_savefile"]
        self.tiff_autoincrement = wavefront_sensor_configuration["tiff_autoincrement"]
        self.pva_image = wavefront_sensor_configuration["pva_image"]
        self.default_image_directory = wavefront_sensor_configuration["default_image_directory"]
        self.data_from = wavefront_sensor_configuration["data_from"]

        self._image_ops = wavefront_sensor_configuration["image_ops"]
        self.image_ops = list_to_string(self._image_ops.get(get_data_from_int_to_string(self.data_from), []))

    def get_plot_tab_name(self): return "Wavefront Sensor Driver"

    def build_widget(self, **kwargs):
        try:    widget_width = kwargs["widget_width"]
        except: widget_width = 1720
        try:    widget_height = kwargs["widget_height"]
        except:
            if sys.platform == 'darwin' : widget_height = 750
            else:                         widget_height = 850

        self.setGeometry(QRect(10, 10, int(widget_width), int(widget_height)))
        self.setFixedWidth(int(widget_width))
        self.setFixedHeight(int(widget_height))

        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignLeft)
        self.setLayout(layout)

        main_box_width    = 720
        input_box_width   = 450
        command_box_width = 260

        self._main_box    = gui.widgetBox(self, "", width=main_box_width,    height=self.height() - 20)

        splitter = ToggleSplitter(ToggleDirection.RemoveLeft, Qt.Horizontal)

        self._main_box.layout().addWidget(splitter)

        forms_left  = QWidget()
        forms_right = QWidget()

        forms_left.setMinimumWidth(0)
        forms_left.setFixedWidth(input_box_width)
        forms_left.setFixedHeight(self.height() - 20)
        forms_right.setMinimumWidth(150)
        forms_right.setFixedWidth(command_box_width)
        forms_right.setFixedHeight(self.height() - 20)

        splitter.addWidget(forms_left)
        splitter.addWidget(forms_right)
        splitter.setHandleWidth(28)
        splitter.setSizes([input_box_width, command_box_width])
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, True)

        self._input_box   = gui.widgetBox(forms_left, "", width=input_box_width - 20, height=self.height() - 20)
        self._command_box = gui.widgetBox(forms_right, "", width=command_box_width - 20, height=self.height() - 20)

        self._input_tab_widget = gui.tabWidget(self._input_box)
        ws_tab     = gui.createTabPage(self._input_tab_widget, "Wavefront Sensor")

        self._command_tab_widget = gui.tabWidget(self._command_box)
        ex_tab     = gui.createTabPage(self._command_tab_widget, "Execution")

        labels_width_1 = 300
        labels_width_2 = 150

        #########################################################################################
        # WAVEFRONT SENSOR

        self._ws_box  = gui.widgetBox(ws_tab, "", width=self._input_box.width()-10, height=self._input_box.height()-40)

        gui.separator(self._ws_box)

        self._default_image_directory_box = gui.widgetBox(self._ws_box, "", width=self._ws_box.width(), orientation='horizontal', addSpace=False)
        self.le_default_image_directory  = gui.lineEdit(self._default_image_directory_box, self, "default_image_directory", "Store image from detector at", orientation='vertical', valueType=str)
        gui.button(self._default_image_directory_box, self, "...", width=30, callback=self._set_default_image_directory)

        tab_widget = gui.tabWidget( self._ws_box)
        ws_tab_1     = gui.createTabPage(tab_widget, "Image Capture")
        ws_tab_2     = gui.createTabPage(tab_widget, "IOC")

        if sys.platform == 'darwin' : ws_box_1 = gui.widgetBox(ws_tab_1, "Execution", width=self._ws_box.width()-15, height=290)
        else:                         ws_box_1 = gui.widgetBox(ws_tab_1, "Execution", width=self._ws_box.width()-15, height=320)

        ws_send_stop_command      = gui.checkBox(ws_box_1, self, "send_stop_command",      "Send Stop Command")
        ws_send_save_command      = gui.checkBox(ws_box_1, self, "send_save_command",      "Send Save Command")
        ws_remove_image           = gui.checkBox(ws_box_1, self, "remove_image",           "Remove Image")
        ws_is_stream_available    = gui.checkBox(ws_box_1, self, "is_stream_available",    "Is Stream Available")

        gui.separator(ws_box_1)

        ws_wait_time        = gui.lineEdit(ws_box_1, self, "wait_time",     "Wait Time [s]",         labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        ws_exposure_time    = gui.lineEdit(ws_box_1, self, "exposure_time", "Exposure Time [s]",     labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        ws_pause_after_shot = gui.lineEdit(ws_box_1, self, "pause_after_shot", "Pause After Shot [s]", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        ws_pixel_format     = gui.lineEdit(ws_box_1, self, "pixel_format",  "Pixel Format",          labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        ws_index_digits     = gui.lineEdit(ws_box_1, self, "index_digits",  "Digits on Image Index", labelWidth=labels_width_1, orientation='horizontal', valueType=int)

        ws_box_2 = gui.widgetBox(ws_tab_1, "Detector", width=self._ws_box.width()-15, height=90)

        ws_pixel_size          = gui.lineEdit(ws_box_2, self, "pixel_size",          "Pixel Size [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        ws_detector_resolution = gui.lineEdit(ws_box_2, self, "detector_resolution", "Resolution [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float)

        ws_box_4 = gui.widgetBox(ws_tab_1, "Image", width=self._ws_box.width()-15, height=145)

        ws_data_from = gui.comboBox(ws_box_4, self, "data_from", label="Data From", labelWidth=labels_width_1, orientation='horizontal', items=["stream", "file"], callback=self._set_data_from)
        self.le_ws_image_ops = gui.lineEdit(ws_box_4, self, "image_ops", "Image Transformations (T, FV, FH)", labelWidth=labels_width_1, orientation='horizontal', valueType=str, callback=self._set_image_ops)
        gui.checkBox(ws_box_4, self, "plot_raw_image", "Plot Raw Image after Shot")
        gui.lineEdit(ws_box_4, self, "plot_rebinning_factor", label="Rebinning Factor for plotting", labelWidth=labels_width_1, orientation='horizontal', valueType=int)

        if sys.platform == 'darwin' : ws_box_3 = gui.widgetBox(ws_tab_2, "Epics", width=self._ws_box.width()-15, height=380)
        else:                         ws_box_3 = gui.widgetBox(ws_tab_2, "Epics", width=self._ws_box.width()-15, height=420)

        ws_cam_pixel_format      = gui.lineEdit(ws_box_3, self, "cam_pixel_format",      "Cam: Pixel Format",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        ws_cam_acquire           = gui.lineEdit(ws_box_3, self, "cam_acquire",           "Cam: Acquire",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        ws_cam_exposure_time     = gui.lineEdit(ws_box_3, self, "cam_exposure_time",     "Cam: Acquire Time",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        ws_cam_image_mode        = gui.lineEdit(ws_box_3, self, "cam_image_mode",        "Cam: Image Mode",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        ws_tiff_enable_callback  = gui.lineEdit(ws_box_3, self, "tiff_enable_callbacks", "Tiff: Enable Callbacks",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        ws_tiff_filename         = gui.lineEdit(ws_box_3, self, "tiff_filename",         "Tiff: File Name",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        ws_tiff_filepath         = gui.lineEdit(ws_box_3, self, "tiff_filepath",         "Tiff: File Path",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        ws_tiff_filenumber       = gui.lineEdit(ws_box_3, self, "tiff_filenumber",       "Tiff: File Number",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        ws_tiff_autosave         = gui.lineEdit(ws_box_3, self, "tiff_autosave",         "Tiff: Auto-Save",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        ws_tiff_savefile         = gui.lineEdit(ws_box_3, self, "tiff_savefile",         "Tiff: Write File",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        ws_tiff_autoincrement    = gui.lineEdit(ws_box_3, self, "tiff_autoincrement",    "Tiff: Auto-Increment",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        ws_pva_image             = gui.lineEdit(ws_box_3, self, "pva_image",             "Pva Image",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)

        def emit_wavefront_sensor_changed(): self.wavefront_sensor_changed.emit()

        ws_send_stop_command.stateChanged.connect(emit_wavefront_sensor_changed)
        ws_send_save_command.stateChanged.connect(emit_wavefront_sensor_changed)
        ws_remove_image.stateChanged.connect(emit_wavefront_sensor_changed)
        ws_is_stream_available.stateChanged.connect(emit_wavefront_sensor_changed)
        ws_wait_time.textChanged.connect(emit_wavefront_sensor_changed)
        ws_exposure_time.textChanged.connect(emit_wavefront_sensor_changed)
        ws_pause_after_shot.textChanged.connect(emit_wavefront_sensor_changed)
        ws_pixel_format.textChanged.connect(emit_wavefront_sensor_changed)
        ws_index_digits.textChanged.connect(emit_wavefront_sensor_changed)
        ws_pixel_size.textChanged.connect(emit_wavefront_sensor_changed)
        ws_detector_resolution.textChanged.connect(emit_wavefront_sensor_changed)
        ws_data_from.currentIndexChanged.connect(emit_wavefront_sensor_changed)
        self.le_ws_image_ops.textChanged.connect(emit_wavefront_sensor_changed)
        ws_cam_pixel_format.textChanged.connect(emit_wavefront_sensor_changed)
        ws_cam_acquire.textChanged.connect(emit_wavefront_sensor_changed)
        ws_cam_exposure_time.textChanged.connect(emit_wavefront_sensor_changed)
        ws_cam_image_mode.textChanged.connect(emit_wavefront_sensor_changed)
        ws_tiff_enable_callback.textChanged.connect(emit_wavefront_sensor_changed)
        ws_tiff_filename.textChanged.connect(emit_wavefront_sensor_changed)
        ws_tiff_filepath.textChanged.connect(emit_wavefront_sensor_changed)
        ws_tiff_filenumber.textChanged.connect(emit_wavefront_sensor_changed)
        ws_tiff_autosave.textChanged.connect(emit_wavefront_sensor_changed)
        ws_tiff_savefile.textChanged.connect(emit_wavefront_sensor_changed)
        ws_tiff_autoincrement.textChanged.connect(emit_wavefront_sensor_changed)
        ws_pva_image.textChanged.connect(emit_wavefront_sensor_changed)

        #########################################################################################
        # Execution

        self._ex_box = gui.widgetBox(ex_tab, "", width=self._command_box.width() - 10, height=self._command_box.height() - 85)

        gui.separator(self._ex_box)

        ex_box_0 = gui.widgetBox(self._ex_box , "Wavefront Sensor",  width=self._ex_box.width()-5, orientation='vertical', addSpace=False)
        ex_box_1 = gui.widgetBox(self._ex_box , "Online",            width=self._ex_box.width()-5, orientation='vertical', addSpace=False)
        ex_box_2 = gui.widgetBox(self._ex_box , "Offline (no W.S.)", width=self._ex_box.width()-5, orientation='vertical', addSpace=False)

        ws_button = gui.button(ex_box_0, None, "Reconnect\nWavefront Sensor", callback=self._connect_wavefront_sensor_callback, width=ex_box_0.width()-20, height=60)
        font = QFont(ws_button.font())
        font.setBold(True)
        font.setItalic(False)
        font.setPixelSize(16)
        ws_button.setFont(font)
        palette = QPalette(ws_button.palette())
        palette.setColor(QPalette.ButtonText, QColor('Dark Red'))
        ws_button.setPalette(palette)

        gui.button(ex_box_1, None, "Take Shot",                    callback=self._take_shot_callback, width=ex_box_1.width()-20, height=35)
        gui.button(ex_box_1, None, "Take Shot As Flat Image",      callback=self._take_shot_as_flat_image_callback, width=ex_box_1.width()-20, height=35)
        gui.separator(ex_box_1)
        gui.button(ex_box_2, None, "Read Image From File",     callback=self._read_image_from_file_callback, width=ex_box_2.width()-20, height=35)

        #########################################################################################
        #########################################################################################
        # output
        #########################################################################################
        #########################################################################################

        self._out_box     = gui.widgetBox(self, "", width=self.width() - main_box_width - 20, height=self.height() - 20, orientation="vertical")
        self._ws_dir_box  = gui.widgetBox(self._out_box, "", width=self._out_box.width(), height=50, orientation="horizontal")

        self._ws_text  = gui.widgetLabel(self._ws_dir_box, "Wavefront Sensor  ")
        self._ws_label = gui.widgetLabel(self._ws_dir_box)

        self.le_working_directory = gui.lineEdit(self._ws_dir_box, self, "working_directory", "  Working Directory", labelWidth=120, orientation='horizontal', valueType=str)
        self.le_working_directory.setReadOnly(True)
        font = QFont(self.le_working_directory.font())
        font.setBold(True)
        font.setItalic(False)
        self.le_working_directory.setFont(font)
        self.le_working_directory.setStyleSheet("QLineEdit {color : darkgreen; background : rgb(243, 240, 160)}")

        self._tab_box    = gui.widgetBox(self._out_box, "", width=self._out_box.width(), height=self._out_box.height() - 55, orientation="vertical")

        self._out_tab_widget = gui.tabWidget(self._tab_box)

        self._out_tab_0 = gui.createTabPage(self._out_tab_widget, "Image")
        self._out_tab_2 = gui.createTabPage(self._out_tab_widget, "Log")

        self._image_box     = gui.widgetBox(self._out_tab_0, "")
        self._log_box       = gui.widgetBox(self._out_tab_2, "Log", width=self._tab_box.width() - 20, height=self._tab_box.height() - 40)

        if sys.platform == 'darwin':  self._image_figure = Figure(figsize=(9.65, 5.9), constrained_layout=True)
        else:                         self._image_figure = Figure(figsize=(9.65, 6.9), constrained_layout=True)

        self._image_figure_canvas = FigureCanvas(self._image_figure)
        self._image_scroll = QScrollArea(self._image_box)
        self._image_scroll.setWidget(self._image_figure_canvas)
        self._image_box.layout().addWidget(NavigationToolbar(self._image_figure_canvas, self))
        self._image_box.layout().addWidget(self._image_scroll)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self._log_box.setLayout(layout)
        if not self._log_stream_widget is None:
            self._log_box.layout().addWidget(self._log_stream_widget.get_widget())
            self._log_stream_widget.set_widget_size(width=self._log_box.width() - 15, height=self._log_box.height() - 35)
        else:
            self._log_box.layout().addWidget(QLabel("Log on file only"))

        self._set_wavefront_sensor_icon()

    def _set_default_image_directory(self):
        self.le_default_image_directory.setText(
            gui.selectDirectoryFromDialog(self,
                                          previous_directory_path=self.default_image_directory,
                                          start_directory=self.working_directory))


    def _set_data_from(self):
        data_from = get_data_from_int_to_string(self.data_from)

        self.image_ops = list_to_string(self._image_ops.get(data_from, []))
        self.le_ws_image_ops.setText(str(self.image_ops))

    def _set_image_ops(self):
        data_from = get_data_from_int_to_string(self.data_from)

        self._image_ops[data_from] = string_to_list(self.image_ops, str)

    def _check_fields(self, raise_errors=True):
        pass

    def _collect_initialization_parameters(self, raise_errors=True):
        initialization_parameters: ScriptData = self._initialization_parameters

        self._check_fields(raise_errors)

        # -----------------------------------------------------
        # Wavefront Sensor

        wavefront_sensor_configuration = initialization_parameters.get_parameter("wavefront_sensor_configuration")

        wavefront_sensor_configuration["send_stop_command"] = self.send_stop_command
        wavefront_sensor_configuration["send_save_command"] = self.send_save_command
        wavefront_sensor_configuration["remove_image"] = self.remove_image
        wavefront_sensor_configuration["wait_time"] = self.wait_time
        wavefront_sensor_configuration["exposure_time"] = self.exposure_time
        wavefront_sensor_configuration["pause_after_shot"] = self.pause_after_shot
        wavefront_sensor_configuration["pixel_format"] = self.pixel_format
        wavefront_sensor_configuration["index_digits"] = self.index_digits
        wavefront_sensor_configuration["is_stream_available"] = self.is_stream_available
        wavefront_sensor_configuration["pixel_size"] = self.pixel_size
        wavefront_sensor_configuration["detector_resolution"] = self.detector_resolution
        wavefront_sensor_configuration["cam_pixel_format"] = self.cam_pixel_format
        wavefront_sensor_configuration["cam_acquire"] = self.cam_acquire
        wavefront_sensor_configuration["cam_exposure_time"] = self.cam_exposure_time
        wavefront_sensor_configuration["cam_image_mode"] = self.cam_image_mode
        wavefront_sensor_configuration["tiff_enable_callback"] = self.tiff_enable_callbacks
        wavefront_sensor_configuration["tiff_filename"] = self.tiff_filename
        wavefront_sensor_configuration["tiff_filepath"] = self.tiff_filepath
        wavefront_sensor_configuration["tiff_filenumber"] = self.tiff_filenumber
        wavefront_sensor_configuration["tiff_autosave"] = self.tiff_autosave
        wavefront_sensor_configuration["tiff_savefile"] = self.tiff_savefile
        wavefront_sensor_configuration["tiff_autoincrement"] = self.tiff_autoincrement
        wavefront_sensor_configuration["pva_image"] = self.pva_image
        wavefront_sensor_configuration["default_image_directory"] = self.default_image_directory
        wavefront_sensor_configuration["data_from"] = self.data_from
        wavefront_sensor_configuration["image_ops"] = self._image_ops

        initialization_parameters.set_parameter("plot_raw_image",                   bool(self.plot_raw_image))
        initialization_parameters.set_parameter("plot_rebinning_factor",            self.plot_rebinning_factor)

    def _connect_wavefront_sensor_callback(self):
        try:
            self._collect_initialization_parameters(raise_errors=True)
            self._connect_wavefront_sensor(self._initialization_parameters)

            MessageDialog.message(self, title="Wavefront Sensor", message="Wavefront Sensor is connected", type="information", width=500)

            self.__is_wavefront_sensor_initialized = True
        except ValueError as error:
            self.__is_wavefront_sensor_initialized = False
            MessageDialog.message(self, title="Input Error", message=str(error.args[0]), type="critical", width=500)
        except Exception as exception:
            self.__is_wavefront_sensor_initialized = False
            MessageDialog.message(self, title="Unexpected Exception", message=str(exception.args[0]), type="critical", width=700)

        self._set_wavefront_sensor_icon()

    def _set_wavefront_sensor_icon(self):
        if self.__is_wavefront_sensor_initialized:
            self._ws_text.setText("Wavefront Sensor  \n(Connected)")
            self._ws_label.setPixmap(self.__ws_pixmaps["green"])
        else:
            self._ws_text.setText("Wavefront Sensor  \n(NOT CONNECTED)")
            self._ws_label.setPixmap(self.__ws_pixmaps["red"])

    def _wavefront_sensor_changed(self):
        if self.__is_wavefront_sensor_initialized:
            self._ws_label.setPixmap(self.__ws_pixmaps["orange"])
            self._ws_text.setText("Wavefront Sensor  \n(Reconnect if changed)")

    # Online -------------------------------------------

    def _take_shot_callback(self):
        dialog = ShowWaitDialog(title="Operation in Progress", text="Taking Shot", parent=self._tab_box)
        dialog.show()

        def _execute():
            try:
                self._collect_initialization_parameters(raise_errors=True)
                h_coord, v_coord, image = self._take_shot(self._initialization_parameters)
                if self.plot_raw_image: self.__plot_shot_image(h_coord, v_coord, image)
            except ValueError as error:
                MessageDialog.message(self, title="Input Error", message=str(error.args[0]), type="critical", width=500)
                if DEBUG_MODE: raise error
            except Exception as exception:
                MessageDialog.message(self, title="Unexpected Exception", message=str(exception.args[0]), type="critical", width=700)
                if DEBUG_MODE: raise exception
            finally:
                dialog.accept()

        try:    QTimer.singleShot(100, _execute)
        except: pass

    def _take_shot_as_flat_image_callback(self):
        dialog = ShowWaitDialog(title="Operation in Progress", text="Taking Shot as Flat Image", parent=self._tab_box)
        dialog.show()

        def _execute():
            try:
                self._collect_initialization_parameters(raise_errors=True)
                h_coord, v_coord, image = self._take_shot_as_flat_image(self._initialization_parameters)
                if self.plot_raw_image: self.__plot_shot_image(h_coord, v_coord, image)
            except ValueError as error:
                MessageDialog.message(self, title="Input Error", message=str(error.args[0]), type="critical", width=500)
                if DEBUG_MODE: raise error
            except Exception as exception:
                MessageDialog.message(self, title="Unexpected Exception", message=str(exception.args[0]), type="critical", width=700)
                if DEBUG_MODE: raise exception
            finally:
                dialog.accept()

        try:    QTimer.singleShot(100, _execute)
        except: pass


    # Offline -------------------------------------------

    def _read_image_from_file_callback(self):
        dialog = ShowWaitDialog(title="Operation in Progress", text="Reading Image From File", parent=self._tab_box)
        dialog.show()

        def _execute():
            try:
                self._collect_initialization_parameters(raise_errors=True)
                h_coord, v_coord, image = self._read_image_from_file(self._initialization_parameters)
                self.__plot_shot_image(h_coord, v_coord, image)
            except ValueError as error:
                MessageDialog.message(self, title="Input Error", message=str(error.args[0]), type="critical", width=500)
                if DEBUG_MODE: raise error
            except Exception as exception:
                MessageDialog.message(self, title="Unexpected Exception", message=str(exception.args[0]), type="critical", width=700)
                if DEBUG_MODE: raise exception
            finally:
                dialog.accept()

        try:    QTimer.singleShot(100, _execute)
        except: pass

    # ----------------------------------------------------
    # PLOT METHODS

    def __plot_shot_image(self, h_coord, v_coord, image):
        data_2D = image
        hh      = h_coord
        vv      = v_coord[::-1]
        hh_orig = copy.deepcopy(h_coord)
        vv_orig = copy.deepcopy(v_coord[::-1])

        if self.plot_rebinning_factor > 1:
            height, width = data_2D.shape
            if height % self.plot_rebinning_factor != 0 or width % self.plot_rebinning_factor != 0:
                raise ValueError("Image dimensions must be divisible by the rebinning factor.")

            new_shape = (height // self.plot_rebinning_factor, self.plot_rebinning_factor, width // self.plot_rebinning_factor, self.plot_rebinning_factor)

            data_2D = data_2D.reshape(new_shape).mean(axis=(1, 3))

            hh = hh.reshape((width  // self.plot_rebinning_factor, self.plot_rebinning_factor)).mean(axis=1)
            vv = vv.reshape((height // self.plot_rebinning_factor, self.plot_rebinning_factor)).mean(axis=1)

        xrange = [np.min(hh), np.max(hh)]
        yrange = [np.min(vv), np.max(vv)]

        fig = self._image_figure.figure
        fig.clear()

        def custom_formatter(x, pos): return f'{x:.2f}'

        axis  = fig.gca()
        plotted_image = axis.pcolormesh(hh, vv, data_2D, cmap=cmm.sunburst_r, rasterized=True)
        axis.set_xlim(xrange[0], xrange[1])
        axis.set_ylim(yrange[0], yrange[1])
        axis.set_xticks(np.linspace(xrange[0], xrange[1], 11, endpoint=True))
        axis.set_yticks(np.linspace(yrange[0], yrange[1], 11, endpoint=True))
        axis.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
        axis.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
        axis.axhline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
        axis.axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
        axis.set_xlabel("Horizontal (mm)")
        axis.set_ylabel("Vertical (mm)")
        axis.set_aspect("equal")
        
        axis.set_title("Choose Roi: select and release (right click: reset)", fontsize=12, color='black', weight='bold')

        if sys.platform == 'darwin':  axis.set_position([-0.1, 0.15, 1.0, 0.8])
        else:                         axis.set_position([0.15, 0.15, 0.8, 0.8])
        
        cbar = fig.colorbar(mappable=plotted_image, ax=axis, pad=0.03, aspect=30, shrink=0.6)
        cbar.ax.text(0.5, 1.05, "Intensity", transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=10, color="black")

        def set_crop(crop_array): self.le_crop.setText(list_to_string(crop_array))

        def onselect(eclick, erelease):
            if eclick.button == 3:  # right click
                axis.set_xlim(xrange[0], xrange[1])
                axis.set_ylim(yrange[0], yrange[1])

                set_crop([0, -1, 0, -1])
            elif eclick.button == 1:

                if self.plot_rebinning_factor > 1:
                    dimensions = [data_2D.shape[0]*self.plot_rebinning_factor, data_2D.shape[1]*self.plot_rebinning_factor]
                    pixel_size = self.pixel_size*self.plot_rebinning_factor*1e3    # mm
                else:
                    dimensions = data_2D.shape
                    pixel_size = self.pixel_size * 1e3  # mm

                ROI_j_lim = np.sort([eclick.xdata, erelease.xdata]).tolist()
                ROI_i_lim = np.sort([eclick.ydata, erelease.ydata]).tolist()

                axis.set_xlim(ROI_j_lim[0] - pixel_size, ROI_j_lim[1] + pixel_size)
                axis.set_ylim(ROI_i_lim[0] - pixel_size, ROI_i_lim[1] + pixel_size)

                ROI_j_lim[0] = np.argmin(abs(vv_orig - ROI_j_lim[0]))
                ROI_j_lim[1] = np.argmin(abs(vv_orig - ROI_j_lim[1]))
                ROI_i_lim[0] = np.argmin(abs(hh_orig - ROI_i_lim[0]))
                ROI_i_lim[1] = np.argmin(abs(hh_orig - ROI_i_lim[1]))

                set_crop([
                          int(dimensions[1] - ROI_i_lim[1]),
                          int(dimensions[1] - ROI_i_lim[0]),
                          int(dimensions[0] - ROI_j_lim[0]),
                          int(dimensions[0] - ROI_j_lim[1]),
                ])

            self._image_figure_canvas.draw()

        def toggle_selector(event): pass

        toggle_selector.RS = RectangleSelector(axis, onselect,
                                               props=dict(facecolor='purple',
                                                          edgecolor='black',
                                                          alpha=0.2,
                                                          fill=True))
        toggle_selector.RS.set_active(True)

        fig.canvas.mpl_connect('key_press_event', toggle_selector)

        self._image_figure_canvas.draw()

        self._out_tab_widget.setCurrentIndex(0)
        
