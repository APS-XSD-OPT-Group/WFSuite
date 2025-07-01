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
import os
import sys

import numpy as np

from aps.common.plot import gui
from aps.common.plot.gui import MessageDialog
from aps.common.widgets.generic_widget import GenericWidget
from aps.common.widgets.congruence import *
from aps.common.scripts.script_data import ScriptData
from aps.common.utilities import list_to_string, string_to_list

from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from cmasher import cm as cmm

from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QHBoxLayout, QVBoxLayout, QScrollArea, QSlider
from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap

from aps.wavefront_analysis.absolute_phase.gui.absolute_phase_manager_initialization import get_data_from_int_to_string, get_data_from_string_to_int

DEBUG_MODE = int(os.environ.get("DEBUG_MODE", 0)) == 1

class AbsolutePhaseWidget(GenericWidget):
    wavefront_sensor_changed = pyqtSignal()
    profile_clicked          = pyqtSignal(str, int)

    def __init__(self, parent, application_name=None, **kwargs):
        self._log_stream_widget             = kwargs["log_stream_widget"]
        self._working_directory             = kwargs["working_directory"]
        self._initialization_parameters     = kwargs["initialization_parameters"]

        # METHODS
        self._connect_wavefront_sensor      = kwargs["connect_wavefront_sensor_method"]
        self._close                         = kwargs["close_method"]
        self._take_shot                     = kwargs["take_shot_method"]
        self._take_shot_as_flat_image       = kwargs["take_shot_as_flat_image_method"]
        self._take_shot_and_generate_mask   = kwargs["take_shot_and_generate_mask_method"]
        self._take_shot_and_process_image   = kwargs["take_shot_and_process_image_method"]
        self._take_shot_and_back_propagate  = kwargs["take_shot_and_back_propagate_method"]
        self._read_image_from_file          = kwargs["read_image_from_file_method"]
        self._generate_mask_from_file       = kwargs["generate_mask_from_file_method"]
        self._process_image_from_file       = kwargs["process_image_from_file_method"]
        self._back_propagate_from_file      = kwargs["back_propagate_from_file_method"]

        self._set_values_from_initialization_parameters()

        icons_path = os.path.join(os.path.dirname(__import__("aps.wavefront_analysis.absolute_phase.gui", fromlist=[""]).__file__), 'icons')
        self.__ws_pixmaps = {
            "red": QPixmap(os.path.join(icons_path, "red_light.png")).scaled(25, 25, Qt.KeepAspectRatio, Qt.SmoothTransformation),
            "green": QPixmap(os.path.join(icons_path, "green_light.png")).scaled(25, 25, Qt.KeepAspectRatio, Qt.SmoothTransformation),
            "orange" : QPixmap(os.path.join(icons_path, "orange_light.png")).scaled(25, 25, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        }

        self.__is_wavefront_sensor_initialized = False

        super(AbsolutePhaseWidget, self).__init__(parent=parent, application_name=application_name, **kwargs)

        self.wavefront_sensor_changed.connect(self._wavefront_sensor_changed)
        self.profile_clicked.connect(self._on_profile_clicked)


    def _set_values_from_initialization_parameters(self):
        self.working_directory = self._working_directory

        initialization_parameters: ScriptData = self._initialization_parameters

        self.wavefront_sensor_image_directory = initialization_parameters.get_parameter("wavefront_sensor_image_directory", os.path.join(os.path.abspath(os.curdir), "wf_images"))
        self.simulated_mask_directory         = initialization_parameters.get_parameter("simulated_mask_directory", os.path.join(self.wavefront_sensor_image_directory, "simulated_mask"))
        self.use_flat                         = initialization_parameters.get_parameter("use_flat", False)
        self.use_dark                         = initialization_parameters.get_parameter("use_dark", False)
        self.save_images                      = initialization_parameters.get_parameter("save_images", True)
        self.plot_raw_image                   = initialization_parameters.get_parameter("plot_raw_image", True)
        self.plot_rebinning_factor            = initialization_parameters.get_parameter("plot_rebinning_factor", 4)
        self.data_from                        = initialization_parameters.get_parameter("data_from", 1)
        self.bp_calibration_mode              = initialization_parameters.get_parameter("bp_calibration_mode", False)
        self.bp_plot_shift                    = initialization_parameters.get_parameter("bp_plot_shift", True)

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
        self.transpose_stream_image = wavefront_sensor_configuration["transpose_stream_ima"]
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

        wavefront_analyzer_configuration = initialization_parameters.get_parameter("wavefront_analyzer_configuration")
        data_analysis_configuration = wavefront_analyzer_configuration["data_analysis"]
        back_propagation_configuration = wavefront_analyzer_configuration["back_propagation"]
    
        self.pattern_size = data_analysis_configuration["pattern_size"]
        self.pattern_thickness = data_analysis_configuration["pattern_thickness"]
        self.pattern_transmission = data_analysis_configuration["pattern_transmission"]
        self.ran_mask = data_analysis_configuration["ran_mask"]
        self.propagation_distance = data_analysis_configuration["propagation_distance"]
        self.energy = data_analysis_configuration["energy"]
        self.source_v = data_analysis_configuration["source_v"]
        self.source_h = data_analysis_configuration["source_h"]
        self.source_distance_v = data_analysis_configuration["source_distance_v"]
        self.source_distance_h = data_analysis_configuration["source_distance_h"]
        self.d_source_recal = data_analysis_configuration["d_source_recal"]
        self.find_transfer_matrix = data_analysis_configuration["find_transfer_matrix"]
        self.crop = list_to_string(data_analysis_configuration["crop"])
        self.estimation_method = data_analysis_configuration["estimation_method"]
        self.propagator = data_analysis_configuration["propagator"]

        self._image_ops = data_analysis_configuration["image_ops"]

        self.image_ops = list_to_string(self._image_ops.get(get_data_from_int_to_string(self.data_from), []))
        self.calibration_path = data_analysis_configuration["calibration_path"]
        self.mode = data_analysis_configuration["mode"]
        self.line_width = data_analysis_configuration["line_width"]
        self.rebinning = data_analysis_configuration["rebinning"]
        self.down_sampling = data_analysis_configuration["down_sampling"]
        self.method = data_analysis_configuration["method"]
        self.use_gpu = data_analysis_configuration["use_gpu"]
        self.use_wavelet = data_analysis_configuration["use_wavelet"]
        self.wavelet_cut = data_analysis_configuration["wavelet_cut"]
        self.pyramid_level = data_analysis_configuration["pyramid_level"]
        self.n_iterations = data_analysis_configuration["n_iterations"]
        self.template_size = data_analysis_configuration["template_size"]
        self.window_search = data_analysis_configuration["window_search"]
        self.crop_boundary = data_analysis_configuration["crop_boundary"]
        self.n_cores = data_analysis_configuration["n_cores"]
        self.n_group = data_analysis_configuration["n_group"]
        self.image_transfer_matrix = list_to_string(data_analysis_configuration["image_transfer_matrix"])
        self.show_align_figure = data_analysis_configuration["show_align_figure"]
        self.correct_scale = data_analysis_configuration["correct_scale"]

        self._delta_f_v = back_propagation_configuration["delta_f_v"]
        self._delta_f_h = back_propagation_configuration["delta_f_h"]

        self.kind = back_propagation_configuration["kind"]
        self.rebinning_bp = back_propagation_configuration["rebinning_bp"]
        self.smooth_intensity = back_propagation_configuration["smooth_intensity"]
        self.sigma_intensity = back_propagation_configuration["sigma_intensity"]
        self.smooth_phase = back_propagation_configuration["smooth_phase"]
        self.sigma_phase = back_propagation_configuration["sigma_phase"]
        self.crop_v = back_propagation_configuration["crop_v"]
        self.crop_h = back_propagation_configuration["crop_h"]
        self.crop_shift_v = back_propagation_configuration["crop_shift_v"]
        self.crop_shift_h = back_propagation_configuration["crop_shift_h"]
        self.distance = back_propagation_configuration["distance"]
        self.distance_v = back_propagation_configuration["distance_v"]
        self.distance_h = back_propagation_configuration["distance_h"]
        self.delta_f_v = self._delta_f_v.get(self.method, 0.0)
        self.delta_f_h = self._delta_f_h.get(self.method, 0.0)
        self.rms_range_v = list_to_string(back_propagation_configuration["rms_range_v"])
        self.rms_range_h = list_to_string(back_propagation_configuration["rms_range_h"])
        self.magnification_v = back_propagation_configuration["magnification_v"]
        self.magnification_h = back_propagation_configuration["magnification_h"]
        self.shift_half_pixel = back_propagation_configuration["shift_half_pixel"]
        self.scan_best_focus = back_propagation_configuration["scan_best_focus"]
        self.use_fit = back_propagation_configuration["use_fit"]
        self.best_focus_from = back_propagation_configuration["best_focus_from"]
        self.best_focus_scan_range   = list_to_string(back_propagation_configuration["best_focus_scan_range"])
        self.best_focus_scan_range_v = list_to_string(back_propagation_configuration["best_focus_scan_range_v"])
        self.best_focus_scan_range_h = list_to_string(back_propagation_configuration["best_focus_scan_range_h"])

    def get_plot_tab_name(self): return "Wavefront Sensor Driver and Data Analysis"

    def build_widget(self, **kwargs):
        geom = QApplication.desktop().availableGeometry()

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

        button_box = gui.widgetBox(self._main_box, "", width=self._main_box.width(), orientation='horizontal')
        button_box.layout().setAlignment(Qt.AlignCenter)

        exit_button           = gui.button(button_box, None, "Exit GUI", callback=self._close_callback, width=self._main_box.width(), height=35)
        font = QFont(exit_button.font())
        font.setBold(True)
        font.setItalic(True)
        exit_button.setFont(font)
        palette = QPalette(exit_button.palette())
        palette.setColor(QPalette.ButtonText, QColor('Dark Blue'))
        exit_button.setPalette(palette)

        forms_box = gui.widgetBox(self._main_box, "", width=self._main_box.width(), orientation='horizontal')

        self._input_box   = gui.widgetBox(forms_box, "", width=input_box_width, height=self.height() - 60)
        self._command_box = gui.widgetBox(forms_box, "", width=command_box_width, height=self.height() - 60)

        self._input_tab_widget = gui.tabWidget(self._input_box)
        ws_tab     = gui.createTabPage(self._input_tab_widget, "Wavefront Sensor")
        wa_tab     = gui.createTabPage(self._input_tab_widget, "Wavefront Analysis")

        self._command_tab_widget = gui.tabWidget(self._command_box)
        ex_tab     = gui.createTabPage(self._command_tab_widget, "Execution")

        labels_width_1 = 300
        labels_width_2 = 150
        labels_width_3 = 100

        #########################################################################################
        # WAVEFRONT SENSOR

        self._ws_box  = gui.widgetBox(ws_tab, "", width=self._input_box.width()-10, height=self._input_box.height()-120)

        gui.separator(self._ws_box)

        self._wavefront_sensor_image_directory_box = gui.widgetBox(self._ws_box , "", width=self._ws_box.width(), orientation='horizontal', addSpace=False)
        self.le_wavefront_sensor_image_directory  = gui.lineEdit(self._wavefront_sensor_image_directory_box, self, "wavefront_sensor_image_directory", "Store image from detector at", orientation='vertical', valueType=str)
        gui.button(self._wavefront_sensor_image_directory_box, self, "...", width=30, callback=self._set_wavefront_sensor_image_directory)

        self._simulated_mask_directory_box = gui.widgetBox(self._ws_box , "", width=self._ws_box.width(), orientation='horizontal', addSpace=False)
        self.le_simulated_mask_directory  = gui.lineEdit(self._simulated_mask_directory_box, self, "simulated_mask_directory", "Simulated Mask at", orientation='vertical', valueType=str)
        gui.button(self._simulated_mask_directory_box, self, "...", width=30, callback=self._set_simulated_mask_directory)

        tab_widget = gui.tabWidget( self._ws_box)
        ws_tab_1     = gui.createTabPage(tab_widget, "Image Capture")
        ws_tab_2     = gui.createTabPage(tab_widget, "IOC")

        if sys.platform == 'darwin' : ws_box_1 = gui.widgetBox(ws_tab_1, "Execution", width=self._ws_box.width()-15, height=310)
        else:                         ws_box_1 = gui.widgetBox(ws_tab_1, "Execution", width=self._ws_box.width()-15, height=340)

        ws_send_stop_command      = gui.checkBox(ws_box_1, self, "send_stop_command",      "Send Stop Command")
        ws_send_save_command      = gui.checkBox(ws_box_1, self, "send_save_command",      "Send Save Command")
        ws_remove_image           = gui.checkBox(ws_box_1, self, "remove_image",           "Remove Image")
        ws_is_stream_available    = gui.checkBox(ws_box_1, self, "is_stream_available",    "Is Stream Available")
        ws_transpose_stream_image = gui.checkBox(ws_box_1, self, "transpose_stream_image", "Transpose Stream Image")

        gui.separator(ws_box_1)

        ws_wait_time        = gui.lineEdit(ws_box_1, self, "wait_time",     "Wait Time [s]",         labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        ws_exposure_time    = gui.lineEdit(ws_box_1, self, "exposure_time", "Exposure Time [s]",     labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        ws_pause_after_shot = gui.lineEdit(ws_box_1, self, "pause_after_shot", "Pause After Shot [s]", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        ws_pixel_format     = gui.lineEdit(ws_box_1, self, "pixel_format",  "Pixel Format",          labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        ws_index_digits     = gui.lineEdit(ws_box_1, self, "index_digits",  "Digits on Image Index", labelWidth=labels_width_1, orientation='horizontal', valueType=int)

        ws_box_2 = gui.widgetBox(ws_tab_1, "Detector", width=self._ws_box.width()-15, height=100)

        ws_pixel_size          = gui.lineEdit(ws_box_2, self, "pixel_size",          "Pixel Size [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        ws_detector_resolution = gui.lineEdit(ws_box_2, self, "detector_resolution", "Resolution [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float)

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
        ws_transpose_stream_image.stateChanged.connect(emit_wavefront_sensor_changed)
        ws_wait_time.textChanged.connect(emit_wavefront_sensor_changed)
        ws_exposure_time.textChanged.connect(emit_wavefront_sensor_changed)
        ws_pause_after_shot.textChanged.connect(emit_wavefront_sensor_changed)
        ws_pixel_format.textChanged.connect(emit_wavefront_sensor_changed)
        ws_index_digits.textChanged.connect(emit_wavefront_sensor_changed)
        ws_pixel_size.textChanged.connect(emit_wavefront_sensor_changed)
        ws_detector_resolution.textChanged.connect(emit_wavefront_sensor_changed)
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
        # WAVEFRONT ANALYSIS

        if sys.platform == 'darwin' : self._wa_box  = gui.widgetBox(wa_tab, "", width=self._input_box.width()-10, height=self._input_box.height()-40)
        else:                         self._wa_box  = gui.widgetBox(wa_tab, "", width=self._input_box.width()-10, height=self._input_box.height()-40)

        gui.separator(self._wa_box)

        self._wa_tab_widget = gui.tabWidget(self._wa_box)

        tab_1     = gui.createTabPage(self._wa_tab_widget, "Analysis")
        tab_2     = gui.createTabPage(self._wa_tab_widget, "Back-Propagation")

        self._wa_tab_widget_1 = gui.tabWidget(tab_1)
        self._wa_tab_widget_2 = gui.tabWidget(tab_2)

        wa_tab_1     = gui.createTabPage(self._wa_tab_widget_1, "Setup")
        wa_tab_2     = gui.createTabPage(self._wa_tab_widget_1, "Calculation")
        wa_tab_5     = gui.createTabPage(self._wa_tab_widget_1, "Runtime")
        wa_tab_3     = gui.createTabPage(self._wa_tab_widget_2, "Propagation")
        wa_tab_4     = gui.createTabPage(self._wa_tab_widget_2, "Best Focus")

        if sys.platform == 'darwin': wa_box_1 = gui.widgetBox(wa_tab_1, "Mask", width=self._wa_box.width()-25, height=170)
        else:                        wa_box_1 = gui.widgetBox(wa_tab_1, "Mask", width=self._wa_box.width()-25, height=190)

        gui.lineEdit(wa_box_1, self, "pattern_size",          "Pattern Size [m]",           labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_1, self, "pattern_thickness",     "Pattern Thickness [m]",      labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_1, self, "pattern_transmission",  "Pattern Transmission [0,1]", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_1, self, "ran_mask",              "Random Mask",                labelWidth=labels_width_3, orientation='horizontal', valueType=str)
        gui.lineEdit(wa_box_1, self, "propagation_distance",  "Propagation Distance [m]",   labelWidth=labels_width_1, orientation='horizontal', valueType=float)

        if sys.platform == 'darwin': wa_box_2 = gui.widgetBox(wa_tab_1, "Source", width=self._wa_box.width()-25, height=170)
        else:                        wa_box_2 = gui.widgetBox(wa_tab_1, "Source", width=self._wa_box.width()-25, height=190)

        le = gui.lineEdit(wa_box_2, self, "energy",            "Energy [eV]",           labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_2, self, "source_v",          "Source Size V [m]",      labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_2, self, "source_h",          "Source Size H [m]",      labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_2, self, "source_distance_v", "Source Distance V [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_2, self, "source_distance_h", "Source Distance H [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        font = QFont(le.font())
        font.setBold(True)
        font.setItalic(False)
        font.setPixelSize(14)
        le.setFont(font)
        le.setStyleSheet("QLineEdit {color : darkred}")

        if sys.platform == 'darwin' : wa_box_3 = gui.widgetBox(wa_tab_1, "Image", width=self._wa_box.width()-25, height=180)
        else:                         wa_box_3 = gui.widgetBox(wa_tab_1, "Image", width=self._wa_box.width()-25, height=210)

        gui.checkBox(wa_box_3, self, "plot_raw_image", "Plot Raw Image after Shot")
        gui.lineEdit(wa_box_3, self, "plot_rebinning_factor", label="Rebinning Factor for plotting", labelWidth=labels_width_1, orientation='horizontal', valueType=int)

        gui.comboBox(wa_box_3, self, "data_from", label="Data From", labelWidth=labels_width_1, orientation='horizontal', items=["stream", "file"], callback=self._set_data_from)
        self.le_image_ops = gui.lineEdit(wa_box_3, self, "image_ops", "Image Transformations (T, FV, FH)", labelWidth=labels_width_1, orientation='horizontal', valueType=str, callback=self._set_image_ops)
        self.le_crop = gui.lineEdit(wa_box_3, self, "crop", "Crop (-1: auto, n: pixels around center,\n            [b, t, l, r]: coordinates in pixels)", labelWidth=labels_width_1, orientation='horizontal', valueType=str)

        wa_box_7 = gui.widgetBox(wa_tab_5, "Processing", width=self._wa_box.width()-25, height=130)

        gui.lineEdit(wa_box_7, self, "n_cores", label="Number of Cores", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_7, self, "n_group", label="Number of Threads", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.checkBox(wa_box_7, self, "use_gpu",      "Use GPUs")

        wa_box_4 = gui.widgetBox(wa_tab_5, "Output", width=self._wa_box.width()-25, height=100)

        gui.checkBox(wa_box_4, self, "show_align_figure",  "Show Align Figure")
        gui.checkBox(wa_box_4, self, "correct_scale",      "Correct Scale")

        if sys.platform == 'darwin' : wa_box_5 = gui.widgetBox(wa_tab_2, "Simulated Mask", width=self._wa_box.width()-25, height=140)
        else:                         wa_box_5 = gui.widgetBox(wa_tab_2, "Simulated Mask", width=self._wa_box.width()-25, height=170)

        gui.checkBox(wa_box_5, self, "d_source_recal",  "Source Distance Recalculation", callback=self._set_d_source_recal)
        self.le_estimation_method = gui.lineEdit(wa_box_5, self, "estimation_method", "Method", labelWidth=labels_width_1, orientation='horizontal', valueType=str)
        gui.checkBox(wa_box_5, self, "find_transfer_matrix",  "Find Transfer Matrix")
        self._le_itm = gui.lineEdit(wa_box_5, self, "image_transfer_matrix", "Image Transfer Matrix", labelWidth=labels_width_1, orientation='horizontal', valueType=str)

        if sys.platform == 'darwin' : wa_box_6 = gui.widgetBox(wa_tab_2, "Reconstruction", width=self._wa_box.width()-25, height=400)
        else:                         wa_box_6 = gui.widgetBox(wa_tab_2, "Reconstruction", width=self._wa_box.width()-25, height=450)

        gui.checkBox(wa_box_6, self, "use_flat", "Use Flat Image")
        gui.checkBox(wa_box_6, self, "use_dark", "Use Dark Image")

        gui.lineEdit(wa_box_6, self, "mode", label="Mode (area, lineWidth)", labelWidth=labels_width_1, orientation='horizontal', valueType=str)
        gui.lineEdit(wa_box_6, self, "line_width", label="Line Width", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "rebinning", label="Image Rebinning Factor", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_6, self, "down_sampling", label="Down Sampling", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_6, self, "method", label="Method (WXST, SPINNet(SD), simple)", labelWidth=labels_width_1, orientation='horizontal', valueType=str, callback=self._set_method)
        gui.checkBox(wa_box_6, self, "use_wavelet",  "Use Wavelets")

        gui.lineEdit(wa_box_6, self, "wavelet_cut", label="Wavelet Cut", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "pyramid_level", label="Pyramid Level", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "n_iterations", label="Number of Iterations", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "template_size", label="Template Size", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "window_search", label="Window Search", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "crop_boundary", "Boundary Crop (-1: auto, 0: no, n: nr pixels)", labelWidth=labels_width_1, orientation='horizontal', valueType=int)

        #########################################################################################
        # Back-Propagation

        bp_box_1 = gui.widgetBox(wa_tab_3, "Propagation", width=self._wa_box.width()-25, height=260)

        self.le_kind  = gui.lineEdit(bp_box_1, self, "kind", label="Kind (1D, 2D)", labelWidth=labels_width_1, orientation='horizontal',  valueType=str, callback=self._set_kind)

        self.kind_box_1_1 = gui.widgetBox(bp_box_1, "", width=bp_box_1.width()-20, height=50)
        self.kind_box_2_1 = gui.widgetBox(bp_box_1, "", width=bp_box_1.width()-20, height=50)

        gui.lineEdit(self.kind_box_1_1, self, "distance",   label="Propagation Distance [m] (<0)", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(self.kind_box_2_1, self, "distance_h", label="Propagation Distance H  [m] (<0)",  labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(self.kind_box_2_1, self, "distance_v", label="Propagation Distance V  [m] (<0)",  labelWidth=labels_width_1, orientation='horizontal', valueType=float)

        self.le_delta_f_h = gui.lineEdit(bp_box_1, self, "delta_f_h", label="Phase Shift H [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float, callback=self._set_delta_f)
        self.le_delta_f_v = gui.lineEdit(bp_box_1, self, "delta_f_v", label="Phase Shift V [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float, callback=self._set_delta_f)

        gui.lineEdit(bp_box_1, self, "magnification_h", label="Magnification H", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(bp_box_1, self, "magnification_v", label="Magnification V", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.checkBox(bp_box_1, self, "shift_half_pixel",  "Shift Half Pixel")

        if sys.platform == 'darwin' : bp_box_2 = gui.widgetBox(wa_tab_3, "Image", width=self._wa_box.width()-25, height=260)
        else:                         bp_box_2 = gui.widgetBox(wa_tab_3, "Image", width=self._wa_box.width()-25, height=280)

        gui.lineEdit(bp_box_2, self, "rebinning_bp", label="Wavefront Rebinning Factor", labelWidth=labels_width_1, orientation='horizontal', valueType=float)

        box = gui.widgetBox(bp_box_2, "", orientation="horizontal")
        gui.checkBox(box, self, "smooth_intensity", "Smooth Intensity")
        gui.lineEdit(box, self, "sigma_intensity", label="\u03c3", labelWidth=20, orientation='horizontal', valueType=float)

        box = gui.widgetBox(bp_box_2, "", orientation="horizontal")
        gui.checkBox(box, self, "smooth_phase", "Smooth Phase    ")
        gui.lineEdit(box, self, "sigma_phase", label="\u03c3", labelWidth=20, orientation='horizontal', valueType=float)

        gui.lineEdit(bp_box_2, self, "crop_h",       label="Crop H", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(bp_box_2, self, "crop_shift_h", label="Crop Shift H", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(bp_box_2, self, "crop_v",       label="Crop V", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(bp_box_2, self, "crop_shift_v", label="Crop Shift V", labelWidth=labels_width_1, orientation='horizontal', valueType=int)

        gui.checkBox(bp_box_2, self, "bp_plot_shift", "Add shift on plots")

        bp_box_3 = gui.widgetBox(wa_tab_4, "Best Focus", width=self._wa_box.width()-25, height=270)

        gui.checkBox(bp_box_3, self, "scan_best_focus", "Scan Best Focus", callback=self._set_scan_best_focus)

        self._bp_box_3_1 = gui.widgetBox(bp_box_3, "", width=bp_box_3.width()-20, height=210)

        gui.checkBox(self._bp_box_3_1, self, "use_fit", "Use Polynomial Fit")
        gui.lineEdit(self._bp_box_3_1, self, "best_focus_from",   label="Besto Focus From (rms, fwhm)",   labelWidth=labels_width_1, orientation='horizontal', valueType=str)

        self.kind_box_1_2 = gui.widgetBox(self._bp_box_3_1, "", width=bp_box_3.width()-20, height=50)
        self.kind_box_2_2 = gui.widgetBox(self._bp_box_3_1, "", width=bp_box_3.width()-20, height=50)

        gui.lineEdit(self.kind_box_1_2, self, "best_focus_scan_range",   label="Range [m] (start, stop, step)",   labelWidth=200, orientation='horizontal', valueType=str)
        gui.lineEdit(self.kind_box_2_2, self, "best_focus_scan_range_h", label="Range H [m] (start, stop, step)", labelWidth=200, orientation='horizontal', valueType=str)
        gui.lineEdit(self.kind_box_2_2, self, "best_focus_scan_range_v", label="Range V [m] (start, stop, step)", labelWidth=200, orientation='horizontal', valueType=str)

        gui.lineEdit(self._bp_box_3_1, self, "rms_range_h", label="R.M.S. Range H [m] (start, stop)", labelWidth=220, orientation='horizontal', valueType=str)
        gui.lineEdit(self._bp_box_3_1, self, "rms_range_v", label="R.M.S. Range V [m] (start, stop)", labelWidth=220, orientation='horizontal', valueType=str)

        gui.checkBox(self._bp_box_3_1, self, "bp_calibration_mode", "Phase Shift Calibration")

        self._set_data_from()
        self._set_method()
        self._set_d_source_recal()
        self._set_kind()
        self._set_scan_best_focus()

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
        gui.button(ex_box_1, None, "Take Shot and Generate Mask",  callback=self._take_shot_and_generate_mask_callback, width=ex_box_1.width()-20, height=35)
        gui.button(ex_box_1, None, "Take Shot and Process Image",  callback=self._take_shot_and_process_image_callback, width=ex_box_1.width()-20, height=35)
        gui.button(ex_box_1, None, "Take Shot and Back-Propagate", callback=self._take_shot_and_back_propagate_callback, width=ex_box_1.width()-20, height=35)

        gui.button(ex_box_2, None, "Read Image From File",     callback=self._read_image_from_file_callback, width=ex_box_2.width()-20, height=35)
        gui.separator(ex_box_2)
        gui.button(ex_box_2, None, "Generate Mask From File",  callback=self._generate_mask_from_file_callback, width=ex_box_2.width()-20, height=35)
        gui.button(ex_box_2, None, "Process Image From File",  callback=self._process_image_from_file_callback, width=ex_box_2.width()-20, height=35)
        gui.button(ex_box_2, None, "Back-Propagate From File", callback=self._back_propagate_from_file_callback, width=ex_box_2.width()-20, height=35)

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

        tab_box    = gui.widgetBox(self._out_box, "", width=self._out_box.width(), height=self._out_box.height() - 55, orientation="vertical")
        self._out_tab_widget = gui.tabWidget(tab_box)

        self._out_tab_0 = gui.createTabPage(self._out_tab_widget, "Image")
        self._out_tab_1 = gui.createTabPage(self._out_tab_widget, "Wavefront")
        self._out_tab_2 = gui.createTabPage(self._out_tab_widget, "Log")

        self._image_box     = gui.widgetBox(self._out_tab_0, "")
        self._wavefront_box = gui.widgetBox(self._out_tab_1, "")
        self._log_box       = gui.widgetBox(self._out_tab_2, "Log", width=tab_box.width() - 20, height=tab_box.height() - 40)

        if sys.platform == 'darwin':  self._image_figure = Figure(figsize=(9.65, 5.9), constrained_layout=True)
        else:                         self._image_figure = Figure(figsize=(9.65, 6.9), constrained_layout=True)

        self._image_figure_canvas = FigureCanvas(self._image_figure)
        self._image_scroll = QScrollArea(self._image_box)
        self._image_scroll.setWidget(self._image_figure_canvas)
        self._image_box.layout().addWidget(NavigationToolbar(self._image_figure_canvas, self))
        self._image_box.layout().addWidget(self._image_scroll)

        self._wf_tab_widget = gui.tabWidget(self._wavefront_box)

        if sys.platform == 'darwin':  figsize = (9.4, 5.15)
        else:                         figsize = (9.4, 6.15) 

        self._wf_tab_0 = gui.createTabPage(self._wf_tab_widget, "At Detector")
        self._wf_tab_1 = gui.createTabPage(self._wf_tab_widget, "Back Propagated")
        self._wf_tab_2 = gui.createTabPage(self._wf_tab_widget, "Longitudinal Profiles")
        
        # ------------------------- WF DET
        
        self._wf_tab_0_widget = gui.tabWidget(self._wf_tab_0)

        self._wf_tab_0_0 = gui.createTabPage(self._wf_tab_0_widget, "Intensity")
        self._wf_tab_0_1 = gui.createTabPage(self._wf_tab_0_widget, "Phase")
        self._wf_tab_0_2 = gui.createTabPage(self._wf_tab_0_widget, "Displacement")
        self._wf_tab_0_3 = gui.createTabPage(self._wf_tab_0_widget, "Curvature")

        self._wf_box_0_0     = gui.widgetBox(self._wf_tab_0_0, "")
        self._wf_box_0_1     = gui.widgetBox(self._wf_tab_0_1, "")
        self._wf_box_0_2     = gui.widgetBox(self._wf_tab_0_2, "")
        self._wf_box_0_3     = gui.widgetBox(self._wf_tab_0_3, "")        

        self._wf_int_figure = Figure(figsize=figsize, constrained_layout=True)
        self._wf_int_figure_canvas = FigureCanvas(self._wf_int_figure)

        self._wf_int_scroll = QScrollArea(self._wf_box_0_0)
        self._wf_int_scroll.setWidget(self._wf_int_figure_canvas)
        self._wf_box_0_0.layout().addWidget(NavigationToolbar(self._wf_int_figure_canvas, self))
        self._wf_box_0_0.layout().addWidget(self._wf_int_scroll)

        self._wf_pha_figure = Figure(figsize=figsize, constrained_layout=True)
        self._wf_pha_figure_canvas = FigureCanvas(self._wf_pha_figure)
        self._wf_pha_scroll = QScrollArea(self._wf_box_0_1)
        self._wf_pha_scroll.setWidget(self._wf_pha_figure_canvas)
        self._wf_box_0_1.layout().addWidget(NavigationToolbar(self._wf_pha_figure_canvas, self))
        self._wf_box_0_1.layout().addWidget(self._wf_pha_scroll)
        
        self._wf_dis_figure = Figure(figsize=figsize, constrained_layout=True)
        self._wf_dis_figure_canvas = FigureCanvas(self._wf_dis_figure)
        self._wf_dis_scroll = QScrollArea(self._wf_box_0_2)
        self._wf_dis_scroll.setWidget(self._wf_dis_figure_canvas)
        self._wf_box_0_2.layout().addWidget(NavigationToolbar(self._wf_dis_figure_canvas, self))
        self._wf_box_0_2.layout().addWidget(self._wf_dis_scroll)
        
        self._wf_cur_figure = Figure(figsize=figsize, constrained_layout=True)
        self._wf_cur_figure_canvas = FigureCanvas(self._wf_cur_figure)
        self._wf_cur_scroll = QScrollArea(self._wf_box_0_3)
        self._wf_cur_scroll.setWidget(self._wf_cur_figure_canvas)
        self._wf_box_0_3.layout().addWidget(NavigationToolbar(self._wf_cur_figure_canvas, self))
        self._wf_box_0_3.layout().addWidget(self._wf_cur_scroll)

        # ------------------------- WF PROP
        
        self._wf_tab_1_widget = gui.tabWidget(self._wf_tab_1)

        self._wf_tab_1_0 = gui.createTabPage(self._wf_tab_1_widget, "Intensity (2D)")
        self._wf_tab_1_1 = gui.createTabPage(self._wf_tab_1_widget, "Projections (1D)")

        self._wf_box_1_0 = gui.widgetBox(self._wf_tab_1_0, "")
        self._wf_box_1_1 = gui.widgetBox(self._wf_tab_1_1, "")

        self._wf_int_prop_figure = Figure(figsize=figsize, constrained_layout=True)
        self._wf_int_prop_figure_canvas = FigureCanvas(self._wf_int_prop_figure)
        self._wf_int_prop_scroll = QScrollArea(self._wf_box_1_0)
        self._wf_int_prop_scroll.setWidget(self._wf_int_prop_figure_canvas)
        self._wf_box_1_0.layout().addWidget(NavigationToolbar(self._wf_int_prop_figure_canvas, self))
        self._wf_box_1_0.layout().addWidget(self._wf_int_prop_scroll)
        
        self._wf_ipr_prop_figure = Figure(figsize=figsize, constrained_layout=True)
        self._wf_ipr_prop_figure_canvas = FigureCanvas(self._wf_ipr_prop_figure)
        self._wf_ipr_prop_scroll = QScrollArea(self._wf_box_1_1)
        self._wf_ipr_prop_scroll.setWidget(self._wf_ipr_prop_figure_canvas)
        self._wf_box_1_1.layout().addWidget(NavigationToolbar(self._wf_ipr_prop_figure_canvas, self))
        self._wf_box_1_1.layout().addWidget(self._wf_ipr_prop_scroll)

        # ------------------------- WF PROILES
        
        self._wf_tab_2_widget = gui.tabWidget(self._wf_tab_2)

        self._wf_tab_2_0 = gui.createTabPage(self._wf_tab_2_widget, "Best Focus Search")
        self._wf_box_2_0 = gui.widgetBox(self._wf_tab_2_0, "")

        self._wf_prof_figure = Figure(figsize=figsize, constrained_layout=True)
        self._wf_prof_figure_canvas = FigureCanvas(self._wf_prof_figure)
        wf_prof_scroll = QScrollArea(self._wf_box_2_0)
        wf_prof_scroll.setWidget(self._wf_prof_figure_canvas)
        self._wf_box_2_0.layout().addWidget(NavigationToolbar(self._wf_prof_figure_canvas, self))
        self._wf_box_2_0.layout().addWidget(wf_prof_scroll)

        self._wf_tab_2_1 = gui.createTabPage(self._wf_tab_2_widget, "Best Focus Profiles")
        self._wf_box_2_1 = gui.widgetBox(self._wf_tab_2_1, "")

        self._wf_prof_figure_2 = Figure(figsize=figsize, constrained_layout=True)
        self._wf_prof_figure_2_canvas = FigureCanvas(self._wf_prof_figure_2)
        wf_prof_scroll = QScrollArea(self._wf_box_2_1)
        wf_prof_scroll.setWidget(self._wf_prof_figure_2_canvas)
        self._wf_box_2_1.layout().addWidget(NavigationToolbar(self._wf_prof_figure_2_canvas, self))
        self._wf_box_2_1.layout().addWidget(wf_prof_scroll)

        self._wf_tab_2_2 = gui.createTabPage(self._wf_tab_2_widget, "Best Focus Planes")
        self._wf_box_2_2 = gui.widgetBox(self._wf_tab_2_2, "")

        self._wf_prof_figure_3 = Figure(figsize=(figsize[0], figsize[1]-0.5), constrained_layout=True)
        self._wf_prof_figure_3_canvas = FigureCanvas(self._wf_prof_figure_3)
        wf_prof_scroll = QScrollArea(self._wf_box_2_2)
        wf_prof_scroll.setWidget(self._wf_prof_figure_3_canvas)
        self._wf_box_2_2.layout().addWidget(NavigationToolbar(self._wf_prof_figure_3_canvas, self))
        self._wf_box_2_2.layout().addWidget(wf_prof_scroll)

        slider_box = gui.widgetBox(self._wf_box_2_2, "", orientation="horizontal")

        self._slider_h = SliderWithButtons()
        self._slider_h.setMinimum(0)
        self._slider_h.setMaximum(100)
        self._slider_h.setValue(50)
        self._slider_h.setTickPosition(QSlider.TicksBelow)
        self._slider_h.setTickInterval(10)

        self._slider_v = SliderWithButtons()
        self._slider_v.setMinimum(0)
        self._slider_v.setMaximum(100)
        self._slider_v.setValue(50)
        self._slider_v.setTickPosition(QSlider.TicksBelow)
        self._slider_v.setTickInterval(10)

        gui.separator(slider_box, width=30)
        slider_box.layout().addWidget(self._slider_h)
        gui.separator(slider_box, width=30)
        slider_box.layout().addWidget(self._slider_v)

        self._wf_box_2_2.layout().addWidget(slider_box)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self._log_box.setLayout(layout)
        if not self._log_stream_widget is None:
            self._log_box.layout().addWidget(self._log_stream_widget.get_widget())
            self._log_stream_widget.set_widget_size(width=self._log_box.width() - 15, height=self._log_box.height() - 35)
        else:
            self._log_box.layout().addWidget(QLabel("Log on file only"))

        self._set_wavefront_sensor_icon()

    def _set_wavefront_sensor_image_directory(self):
        self.le_wavefront_sensor_image_directory.setText(
            gui.selectDirectoryFromDialog(self,
                                          previous_directory_path=self.wavefront_sensor_image_directory,
                                          start_directory=self.working_directory))

    def _set_simulated_mask_directory(self):
        self.le_simulated_mask_directory.setText(
            gui.selectDirectoryFromDialog(self,
                                          previous_directory_path=self.simulated_mask_directory,
                                          start_directory=self.wavefront_sensor_image_directory))

    def _set_d_source_recal(self):
        self.le_estimation_method.setEnabled(bool(self.d_source_recal))

    def _set_kind(self):
        if not self.kind in ["2D", "1D"]: MessageDialog.message(self, title="Input Error", message="Kind must be '2D' or '1D'", type="critical", width=500)
        else:
            self.kind_box_1_1.setVisible(self.kind=="2D")
            self.kind_box_1_2.setVisible(self.kind=="2D")
            self.kind_box_2_1.setVisible(self.kind=="1D")
            self.kind_box_2_2.setVisible(self.kind=="1D")

    def _set_scan_best_focus(self):
        self._bp_box_3_1.setEnabled(bool(self.scan_best_focus))

    def _set_data_from(self):
        data_from = get_data_from_int_to_string(self.data_from)

        self.image_ops = list_to_string(self._image_ops.get(data_from, []))
        self.le_image_ops.setText(str(self.image_ops))

    def _set_image_ops(self):
        data_from = get_data_from_int_to_string(self.data_from)

        self._image_ops[data_from] = string_to_list(self.image_ops, str)

    def _set_method(self):
        if not self.method in ["WXST", "SPINNet", "SPINNetSD", "simple"]: MessageDialog.message(self, title="Input Error", message="Method must be 'WXST', 'SPINNet', 'SPINNetSD' or 'simple'", type="critical", width=500)
        else:
            self.delta_f_h = self._delta_f_h.get(self.method, 0.0)
            self.delta_f_v = self._delta_f_v.get(self.method, 0.0)
            self.le_delta_f_h.setText(str(self.delta_f_h))
            self.le_delta_f_v.setText(str(self.delta_f_v))

    def _set_delta_f(self):
        if not self.method in ["WXST", "SPINNet", "SPINNetSD", "simple"]: MessageDialog.message(self, title="Input Error", message="Method must be 'WXST', 'SPINNet', 'SPINNetSD' or 'simple'", type="critical", width=500)
        else:
            self._delta_f_h[self.method] = self.delta_f_h
            self._delta_f_v[self.method] = self.delta_f_v

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
        wavefront_sensor_configuration["transpose_stream_ima"] = self.transpose_stream_image
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
    
        # -----------------------------------------------------
        # Wavefront Analyzer

        wavefront_analyzer_configuration = initialization_parameters.get_parameter("wavefront_analyzer_configuration")
        data_analysis_configuration      = wavefront_analyzer_configuration["data_analysis"]
        back_propagation_configuration   = wavefront_analyzer_configuration["back_propagation"]

        data_analysis_configuration["pattern_size"] = self.pattern_size
        data_analysis_configuration["pattern_thickness"] = self.pattern_thickness
        data_analysis_configuration["pattern_transmission"] = self.pattern_transmission
        data_analysis_configuration["ran_mask"] = self.ran_mask
        data_analysis_configuration["propagation_distance"] = self.propagation_distance
        data_analysis_configuration["energy"] = self.energy
        data_analysis_configuration["source_v"] = self.source_v
        data_analysis_configuration["source_h"] = self.source_h
        data_analysis_configuration["source_distance_v"] = self.source_distance_v
        data_analysis_configuration["source_distance_h"] = self.source_distance_h
        data_analysis_configuration["d_source_recal"] = self.d_source_recal
        data_analysis_configuration["find_transfer_matrix"] = self.find_transfer_matrix
        data_analysis_configuration["crop"] = string_to_list(self.crop, int)
        data_analysis_configuration["estimation_method"] = self.estimation_method
        data_analysis_configuration["propagator"] = self.propagator
        data_analysis_configuration["image_ops"] = self._image_ops
        data_analysis_configuration["calibration_path"] = self.calibration_path
        data_analysis_configuration["mode"] = self.mode
        data_analysis_configuration["line_width"] = self.line_width
        data_analysis_configuration["rebinning"] = self.rebinning
        data_analysis_configuration["down_sampling"] = self.down_sampling
        data_analysis_configuration["method"] = self.method
        data_analysis_configuration["use_gpu"] = self.use_gpu
        data_analysis_configuration["use_wavelet"] = self.use_wavelet
        data_analysis_configuration["wavelet_cut"] = self.wavelet_cut
        data_analysis_configuration["pyramid_level"] = self.pyramid_level
        data_analysis_configuration["n_iterations"] = self.n_iterations
        data_analysis_configuration["template_size"] = self.template_size
        data_analysis_configuration["window_search"] = self.window_search
        data_analysis_configuration["crop_boundary"] = self.crop_boundary
        data_analysis_configuration["n_cores"] = self.n_cores
        data_analysis_configuration["n_group"] = self.n_group
        data_analysis_configuration["image_transfer_matrix"] = string_to_list(self.image_transfer_matrix, int)
        data_analysis_configuration["show_align_figure"] = self.show_align_figure
        data_analysis_configuration["correct_scale"] = self.correct_scale

        back_propagation_configuration["kind"]         = self.kind
        back_propagation_configuration["rebinning_bp"] = self.rebinning_bp
        back_propagation_configuration["smooth_intensity"] = self.smooth_intensity
        back_propagation_configuration["sigma_intensity"] = self.sigma_intensity
        back_propagation_configuration["smooth_phase"] = self.smooth_phase
        back_propagation_configuration["sigma_phase"] = self.sigma_phase
        back_propagation_configuration["crop_v"] = self.crop_v
        back_propagation_configuration["crop_h"] = self.crop_h
        back_propagation_configuration["crop_shift_v"] = self.crop_shift_v
        back_propagation_configuration["crop_shift_h"] = self.crop_shift_h
        back_propagation_configuration["distance"] = self.distance
        back_propagation_configuration["distance_v"] = self.distance_v
        back_propagation_configuration["distance_h"] = self.distance_h
        back_propagation_configuration["delta_f_v"] = self._delta_f_v
        back_propagation_configuration["delta_f_h"] = self._delta_f_h
        back_propagation_configuration["rms_range_v"]      = string_to_list(self.rms_range_v, float)
        back_propagation_configuration["rms_range_h"]      = string_to_list(self.rms_range_h, float)
        back_propagation_configuration["magnification_v"]  = self.magnification_v
        back_propagation_configuration["magnification_h"]  = self.magnification_h
        back_propagation_configuration["shift_half_pixel"] = self.shift_half_pixel
        back_propagation_configuration["scan_best_focus"]  = self.scan_best_focus
        back_propagation_configuration["use_fit"]          = self.use_fit
        back_propagation_configuration["best_focus_from"]  = self.best_focus_from
        back_propagation_configuration["best_focus_scan_range"]   = string_to_list(self.best_focus_scan_range, float)
        back_propagation_configuration["best_focus_scan_range_v"] = string_to_list(self.best_focus_scan_range_v, float)
        back_propagation_configuration["best_focus_scan_range_h"] = string_to_list(self.best_focus_scan_range_h, float)

        # Widget ini

        initialization_parameters.set_parameter("wavefront_sensor_image_directory", self.wavefront_sensor_image_directory)
        initialization_parameters.set_parameter("simulated_mask_directory",         self.simulated_mask_directory)
        initialization_parameters.set_parameter("use_dark",                         bool(self.use_dark))
        initialization_parameters.set_parameter("use_flat",                         bool(self.use_flat))
        initialization_parameters.set_parameter("save_images",                      bool(self.save_images))
        initialization_parameters.set_parameter("plot_raw_image",                   bool(self.plot_raw_image))
        initialization_parameters.set_parameter("plot_rebinning_factor",            self.plot_rebinning_factor)
        initialization_parameters.set_parameter("data_from",                        self.data_from)
        initialization_parameters.set_parameter("bp_calibration_mode",              bool(self.bp_calibration_mode))
        initialization_parameters.set_parameter("bp_plot_shift",                    bool(self.bp_plot_shift))

    def _close_callback(self):
        if ConfirmDialog.confirmed(self, "Confirm Exit?"):
            self._collect_initialization_parameters(raise_errors=False)
            self._close(self._initialization_parameters)

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

    def _on_profile_clicked(self, direction, index):
        if direction == "x":   self._slider_h.setValue(index)
        elif direction == "y": self._slider_v.setValue(index)

    # Online -------------------------------------------

    def _take_shot_callback(self):
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

    def _take_shot_as_flat_image_callback(self):
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

    def _take_shot_and_generate_mask_callback(self):
        try:
            self._collect_initialization_parameters(raise_errors=True)
            h_coord, v_coord, image, image_transfer_matrix = self._take_shot_and_generate_mask(self._initialization_parameters)
            if self.plot_raw_image: self.__plot_shot_image(h_coord, v_coord, image)
            self._manage_generate_mask_result(image_transfer_matrix)
        except ValueError as error:
            MessageDialog.message(self, title="Input Error", message=str(error.args[0]), type="critical", width=500)
            if DEBUG_MODE: raise error
        except Exception as exception:
            MessageDialog.message(self, title="Unexpected Exception", message=str(exception.args[0]), type="critical", width=700)
            if DEBUG_MODE: raise exception

    def _take_shot_and_process_image_callback(self):
        try:
            self._collect_initialization_parameters(raise_errors=True)
            h_coord, v_coord, image, wavefront_at_detector_data = self._take_shot_and_process_image(self._initialization_parameters)
            if self.plot_raw_image: self.__plot_shot_image(h_coord, v_coord, image)
            self.__plot_wavefront_at_detector(wavefront_at_detector_data)
        except ValueError as error:
            MessageDialog.message(self, title="Input Error", message=str(error.args[0]), type="critical", width=500)
            if DEBUG_MODE: raise error
        except Exception as exception:
            MessageDialog.message(self, title="Unexpected Exception", message=str(exception.args[0]), type="critical", width=700)
            if DEBUG_MODE: raise exception

    def _take_shot_and_back_propagate_callback(self):
        try:
            self._collect_initialization_parameters(raise_errors=True)
            h_coord, v_coord, image, wavefront_at_detector_data, propagated_wavefront_data = self._take_shot_and_back_propagate(self._initialization_parameters)
            if bool(self.plot_raw_image): self.__plot_shot_image(h_coord, v_coord, image)
            self.__plot_wavefront_at_detector(wavefront_at_detector_data)
            self._manage_back_propagate_result(propagated_wavefront_data)
        except ValueError as error:
            MessageDialog.message(self, title="Input Error", message=str(error.args[0]), type="critical", width=500)
            if DEBUG_MODE: raise error
        except Exception as exception:
            MessageDialog.message(self, title="Unexpected Exception", message=str(exception.args[0]), type="critical", width=700)
            if DEBUG_MODE: raise exception

    # Offline -------------------------------------------

    def _read_image_from_file_callback(self):
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

    def _generate_mask_from_file_callback(self):
        try:
            self._collect_initialization_parameters(raise_errors=True)
            image_transfer_matrix = self._generate_mask_from_file(self._initialization_parameters)
            self._manage_generate_mask_result(image_transfer_matrix)
        except ValueError as error:
            MessageDialog.message(self, title="Input Error", message=str(error.args[0]), type="critical", width=500)
            if DEBUG_MODE: raise error
        except Exception as exception:
            MessageDialog.message(self, title="Unexpected Exception", message=str(exception.args[0]), type="critical", width=700)
            if DEBUG_MODE: raise exception

    def _process_image_from_file_callback(self):
        try:
            self._collect_initialization_parameters(raise_errors=True)
            wavefront_at_detector_data = self._process_image_from_file(self._initialization_parameters)
            self.__plot_wavefront_at_detector(wavefront_at_detector_data)
        except ValueError as error:
            MessageDialog.message(self, title="Input Error", message=str(error), type="critical", width=500)
            if DEBUG_MODE: raise error
        except Exception as exception:
            MessageDialog.message(self, title="Unexpected Exception", message=str(exception), type="critical", height=400, width=700)
            if DEBUG_MODE: raise exception

    def _back_propagate_from_file_callback(self):
        try:
            self._collect_initialization_parameters(raise_errors=True)
            propagated_wavefront_data = self._back_propagate_from_file(self._initialization_parameters)
            self._manage_back_propagate_result(propagated_wavefront_data)
        except ValueError as error:
            MessageDialog.message(self, title="Input Error", message=str(error.args[0]), type="critical", width=500)
            if DEBUG_MODE: raise error
        except Exception as exception:
            MessageDialog.message(self, title="Unexpected Exception", message=str(exception.args[0]), type="critical", width=700)
            if DEBUG_MODE: raise exception

    def _manage_generate_mask_result(self, image_transfer_matrix):
        MessageDialog.message(self, title="Mask Generation", message=f"Image Transfer Matrix: {image_transfer_matrix}", type="information", width=500)

        self.image_transfer_matrix = list_to_string(image_transfer_matrix)
        self._le_itm.setText(self.image_transfer_matrix)

    def _manage_back_propagate_result(self, propagated_wavefront_data):
        self.__plot_back_propagated_wavefront(propagated_wavefront_data)

        if bool(self.scan_best_focus):
            self.__plot_longitudinal_profiles(propagated_wavefront_data)

            if bool(self.bp_calibration_mode):
                focus_z_position_x = propagated_wavefront_data["focus_z_position_x"]
                focus_z_position_y = propagated_wavefront_data["focus_z_position_y"]

                message = "Scan Best Focus Results:\n\n" + \
                          f"Best Focus Position x: {focus_z_position_x}\n" + \
                          f"Best Focus Position y: {focus_z_position_y}\n" + \
                          f"\n\nDo you want to use these data as permanent phase shift for the method {self.method}?"

                if ConfirmDialog.confirmed(self,
                                           title="Scan Best Focus",
                                           message=message,
                                           height=250):
                    self.delta_f_h = -round(focus_z_position_x, 6)
                    self.delta_f_v = -round(focus_z_position_y, 6)
                    self._set_delta_f()
                    self.le_delta_f_h.setText(str(self.delta_f_h))
                    self.le_delta_f_v.setText(str(self.delta_f_v))
                    self._input_tab_widget.setCurrentIndex(1)
                    self._wa_tab_widget.setCurrentIndex(1)
                    self._wa_tab_widget_2.setCurrentIndex(0)

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
        
    def __plot_wavefront_at_detector(self, wavefront_data):
        p_x = self.pixel_size*self.rebinning

        if wavefront_data['mode'] == 'area':
            intensity     = wavefront_data['intensity']
            phase         = wavefront_data['phase']
            line_displace = wavefront_data['line_displace']
            line_curve    = wavefront_data['line_curve']

            plot_2D(self._wf_int_figure.figure, intensity, "[counts]", p_x)
            self._wf_int_figure_canvas.draw()

            plot_2D(self._wf_pha_figure.figure, phase, "[rad]", p_x)
            self._wf_pha_figure_canvas.draw()

            plot_1D(self._wf_dis_figure.figure, line_displace[0], line_displace[1], "[px]", p_x)
            self._wf_dis_figure_canvas.draw()

            plot_1D(self._wf_cur_figure.figure, line_curve[0], line_curve[1], "[1/m]", p_x)
            self._wf_cur_figure_canvas.draw()

        elif wavefront_data['mode'] == 'lineWidth':
            intensity     = wavefront_data['intensity']
            phase         = wavefront_data['line_phase']
            line_displace = wavefront_data['line_displace']
            line_curve    = wavefront_data['line_curve']

            plot_1D(self._wf_int_figure.figure, intensity[0], intensity[1], "[counts]", p_x)
            self._wf_int_figure_canvas.draw()

            plot_1D(self._wf_pha_figure.figure, phase[0], phase[1], "[rad]", p_x)
            self._wf_pha_figure_canvas.draw()

            plot_1D(self._wf_dis_figure.figure, line_displace[0], line_displace[1], "[px]", p_x)
            self._wf_dis_figure_canvas.draw()

            plot_1D(self._wf_cur_figure.figure, line_curve[0], line_curve[1], "[1/m]", p_x)
            self._wf_cur_figure_canvas.draw()
        else:
            MessageDialog.message(self, title="Unexpected Error", message=f"Data not plottable, mode not recognized {wavefront_data['mode']}", type="critical", width=500)

        self._out_tab_widget.setCurrentIndex(1)
        self._wf_tab_widget.setCurrentIndex(0)
        self._wf_tab_0_widget.setCurrentIndex(0)

    def __plot_back_propagated_wavefront(self, wavefront_data):
        def add_text_2D(ax):
            text = "Wavefront Properties:\n"
            for prop, label in zip(["fwhm_x", "fwhm_y", "sigma_x", "sigma_y", "wf_position_x", "wf_position_y"],
                                   ["fwhm(x)", "fwhm(y)", "rms(x)", "rms(y)", "shift(x)", "shift(y)"]):
                text += "\n" + rf"{label:<8}: {wavefront_data[prop]*1e6 : 3.3f} $\mu$m"

            if sys.platform == 'darwin': ax.text(1.5, 0.55, text, color="black", alpha=0.9, fontsize=12, fontname="Courier",
                                                 bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7), transform=ax.transAxes)
            else:                        ax.text(1.25, 0.55, text, color="black", alpha=0.9, fontsize=12, fontname="DejaVu Sans",
                                                 bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7), transform=ax.transAxes)

        def add_text_1D(ax, dir):
            text = f"Direction {dir}:\n"
            for prop, label in zip([f"fwhm_{dir}", f"sigma_{dir}", f"wf_position_{dir}"],
                                   ["fwhm", "rms", "shift"]):
                text += "\n" + rf"{label:<5}: {wavefront_data[prop] * 1e6 : 3.3f} $\mu$m"

            ax.text(0.65, 0.8, text, color="black", alpha=0.9, fontsize=9, fontname=("Courier" if sys.platform == 'darwin' else "DejaVu Sans"),
                    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7), transform=ax.transAxes)

        if wavefront_data['kind'] == '2D':
            intensity     = wavefront_data['intensity']
            intensity_x   = wavefront_data['integrated_intensity_x']
            intensity_y   = wavefront_data['integrated_intensity_y']
            wf_position_x = wavefront_data['wf_position_x']
            wf_position_y = wavefront_data['wf_position_y']
            x_coordinates = wavefront_data['coordinates_x']
            y_coordinates = wavefront_data['coordinates_y']

            coords_orig = [(x_coordinates)*1e6, (y_coordinates)*1e6]
            coords      = [(x_coordinates + wf_position_x)*1e6, (y_coordinates + wf_position_y)*1e6]

            fig = self._wf_int_prop_figure.figure
            fig.clear()
            def custom_formatter(x, pos): return f'{x:.2f}'
            ax = fig.gca()
            if self.bp_plot_shift: image = ax.pcolormesh(coords[0], coords[1], intensity.T, cmap=cmm.sunburst_r, rasterized=True)
            else:                  image = ax.pcolormesh(coords_orig[0], coords_orig[1], intensity.T, cmap=cmm.sunburst_r, rasterized=True)
            ax.set_xlim(coords_orig[0][0], coords_orig[0][-1])
            ax.set_ylim(coords_orig[1][0], coords_orig[1][-1])
            ax.set_xticks(np.linspace(coords_orig[0][0], coords_orig[0][-1], 6, endpoint=True))
            ax.set_yticks(np.linspace(coords_orig[1][0], coords_orig[1][-1], 6, endpoint=True))

            ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
            ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
            ax.axhline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
            ax.axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
            ax.set_xlabel('x ($\mu$m)')
            ax.set_ylabel('y ($\mu$m)')
            ax.set_aspect("equal")
            if sys.platform == 'darwin': ax.set_position([-0.375, 0.15, 1.0, 0.8])
            else:                        ax.set_position([-0.02, 0.15, 0.8, 0.8])
            add_text_2D(ax)
            cbar = fig.colorbar(mappable=image, ax=ax, pad=0.04, aspect=30, shrink=0.6)
            cbar.ax.text(0.5, 1.05, "Intensity", transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=10, color="black")
            self._wf_int_prop_figure_canvas.draw()

            if self.bp_plot_shift: axes = plot_1D(self._wf_ipr_prop_figure.figure, intensity_x, intensity_y, "[counts]", None, coords=coords)
            else:                  axes = plot_1D(self._wf_ipr_prop_figure.figure, intensity_x, intensity_y, "[counts]", None, coords=coords_orig)
            axes[0].set_xlim(coords_orig[0][0], coords_orig[0][-1])
            axes[1].set_xlim(coords_orig[1][0], coords_orig[1][-1])
            axes[0].axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
            axes[1].axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
            add_text_1D(axes[0], "x")
            add_text_1D(axes[1], "y")
            self._wf_ipr_prop_figure_canvas.draw()
        elif wavefront_data['kind'] == '1D':
            intensity_x   = wavefront_data['intensity_x']
            intensity_y   = wavefront_data['intensity_y']
            wf_position_x = wavefront_data['wf_position_x']
            wf_position_y = wavefront_data['wf_position_y']
            x_coordinates = wavefront_data['coordinates_x']
            y_coordinates = wavefront_data['coordinates_y']

            coords_orig = [(x_coordinates)*1e6, (y_coordinates)*1e6]
            coords      = [(x_coordinates + wf_position_x)*1e6, (y_coordinates + wf_position_y)*1e6]

            self._wf_int_prop_figure.figure.clear()
            self._wf_int_prop_figure_canvas.draw()

            if self.bp_plot_shift: axes = plot_1D(self._wf_ipr_prop_figure.figure, intensity_x, intensity_y, "[counts]", None, coords=coords)
            else:                  axes = plot_1D(self._wf_ipr_prop_figure.figure, intensity_x, intensity_y, "[counts]", None, coords=coords_orig)
            axes[0].set_xlim(coords_orig[0][0], coords_orig[0][-1])
            axes[1].set_xlim(coords_orig[1][0], coords_orig[1][-1])
            axes[0].axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
            axes[1].axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
            add_text_1D(axes[0], "x")
            add_text_1D(axes[1], "y")
            self._wf_ipr_prop_figure_canvas.draw()

        self._out_tab_widget.setCurrentIndex(1)
        self._wf_tab_widget.setCurrentIndex(1)
        self._wf_tab_1_widget.setCurrentIndex(0)

    def __plot_longitudinal_profiles(self, profiles_data):
        x_coordinates = 1e6 * profiles_data['coordinates_x']
        y_coordinates = 1e6 * profiles_data['coordinates_y']

        bf_size_values_x     = 1e6 * profiles_data['bf_size_values_x']
        bf_size_values_fit_x = profiles_data.get('bf_size_values_fit_x', None)
        bf_size_values_fit_x = 1e6 * bf_size_values_fit_x if not bf_size_values_fit_x is None else None
        bf_size_values_y     = 1e6 * profiles_data['bf_size_values_y']
        bf_size_values_fit_y = profiles_data.get('bf_size_values_fit_y', None)
        bf_size_values_fit_y = 1e6 * bf_size_values_fit_y if not bf_size_values_fit_y is None else None

        focus_z_position_x = profiles_data["bf_propagation_distance_x"]
        focus_z_position_y = profiles_data["bf_propagation_distance_y"]
        best_size_value_x  = 1e6 * profiles_data["bf_size_value_x"]
        best_size_value_y  = 1e6 * profiles_data["bf_size_value_y"]
        best_focus_from    = profiles_data["scan_best_focus_from"]

        if profiles_data['kind'] == '2D':
            bf_propagation_distances  = profiles_data['bf_propagation_distances']
            bf_propagation_distances_x = bf_propagation_distances
            bf_propagation_distances_y = bf_propagation_distances
            coords                    = [bf_propagation_distances, bf_propagation_distances]
        elif profiles_data['kind'] == '1D':
            bf_propagation_distances_x  = profiles_data['bf_propagation_distances_x']
            bf_propagation_distances_y  = profiles_data['bf_propagation_distances_y']
            coords                      = [bf_propagation_distances_x, bf_propagation_distances_y]

        def plot_ax(ax, dir, coord, size, size_fit, best_size, focus):
            ax.plot(coord, size, marker='o', label=f"Size {dir}")
            if not size_fit is None:ax.plot(coord, size_fit, label=f"Size {dir} - FIT")
            ax.set_xlabel(f'p. distance {dir} (m)', fontsize=22)
            ax.set_ylabel(f"Size {dir} ($\mu$m)", fontsize=22)
            ax.legend()
            ax.grid(True)
            ax.axvline(focus, color="gray", ls="--", linewidth=2, alpha=0.9)
            ax.text(0.53, 0.85, f"{best_focus_from} {round(best_size, 3)} $\mu$m\nat {round(focus, 5)} m", color="blue", alpha=0.9, fontsize=11, fontname=("Courier" if sys.platform == 'darwin' else "DejaVu Sans"),
                    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7), transform=ax.transAxes)

        fig = self._wf_prof_figure
        fig.clear()
        axes = fig.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
        plot_ax(axes[0], "x", coords[0], bf_size_values_x, bf_size_values_fit_x, best_size_value_x, focus_z_position_x)
        plot_ax(axes[1], "y", coords[1], bf_size_values_y, bf_size_values_fit_y, best_size_value_y, focus_z_position_y)
        fig.tight_layout()

        self._wf_prof_figure_canvas.draw()

        # BF Profiles
        def add_text_1D(ax, dir, size, focus, vpos=0.8):
            text = f"Direction {dir}:\n"
            text += "\n" + rf"{best_focus_from:<5}: {size: 3.3f} $\mu$m"
            text += "\n" + rf"{'at':<5}: {round(focus, 5)} m"
            ax.text(0.65, vpos, text, color="black", alpha=0.9, fontsize=9, fontname=("Courier" if sys.platform == 'darwin' else "DejaVu Sans"),
                    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7), transform=ax.transAxes)

        if profiles_data['kind'] == '2D':
            intensity_x   = profiles_data['bf_integrated_intensity_x']
            intensity_y   = profiles_data['bf_integrated_intensity_y']
            intensities_x = profiles_data['bf_integrated_intensities_x']
            intensities_y = profiles_data['bf_integrated_intensities_y']
        elif profiles_data['kind'] == '1D':
            intensity_x   = profiles_data['bf_intensity_x']
            intensity_y   = profiles_data['bf_intensity_y']
            intensities_x = profiles_data['bf_intensities_x']
            intensities_y = profiles_data['bf_intensities_y']

        planes_x      = np.zeros((len(intensity_x), len(bf_propagation_distances_x)))
        planes_y      = np.zeros((len(intensity_y), len(bf_propagation_distances_y)))

        axes = plot_1D(self._wf_prof_figure_2.figure, intensity_x, intensity_y, "[counts]", None, coords=[x_coordinates, y_coordinates])
        axes[0].set_xlim(x_coordinates[0], x_coordinates[-1])
        axes[1].set_xlim(y_coordinates[0], y_coordinates[-1])
        axes[0].axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
        axes[1].axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
        add_text_1D(axes[0], "x", best_size_value_x, focus_z_position_x)
        add_text_1D(axes[1], "y", best_size_value_y, focus_z_position_y)
        self._wf_prof_figure_2_canvas.draw()

        # Propagation planes
        for i in range(planes_x.shape[1]): planes_x[:, i] = intensities_x[i]
        for i in range(planes_y.shape[1]): planes_y[:, i] = intensities_y[i]

        self._wf_prof_figure_3.clear()

        def plot_ax_plane(ax, ax_prof, dir, planes, extent_data, best_size, focus, sizes, distances, coords, profiles):
            ax.imshow(planes, interpolation='bilinear', extent=extent_data)
            ax.set_xlabel(f"p. distance {dir} (m)", fontsize=22)
            ax.set_ylabel(f"{dir} ($\mu$m)", fontsize=22)
            ax.set_aspect('auto')
            ax.axvline(focus, color="gray", ls="--", linewidth=2, alpha=0.9)
            ax.text(0.53, 0.81, f"{best_focus_from} {round(best_size, 3)} $\mu$m\nat {round(focus, 5)} m", color="blue", alpha=0.9, fontsize=11, fontname=("Courier" if sys.platform == 'darwin' else "DejaVu Sans"),
                         bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7), transform=ax.transAxes)

            index = np.abs(distances - focus).argmin()
            ax_prof.plot(coords, profiles[index], 'k')
            ax_prof.set_xlim(coords[0], coords[-1])
            add_text_1D(ax_prof, dir, best_size, focus, vpos=0.7)

            line = ax.axvline(focus, color="gray", ls="--", linewidth=1, alpha=0.9, visible=False)
            text = ax.text(0.5, 0.6, f"{best_focus_from} {round(best_size, 3)} $\mu$m\nat {round(focus, 5)} m", color="darkred", alpha=0.9, fontsize=9, fontname=("Courier" if sys.platform == 'darwin' else "DejaVu Sans"),
                           bbox=dict(facecolor="yellow", edgecolor="darkred", alpha=0.7), transform=ax.transAxes, visible=False)

            def onclick(event):
                # Check if the click is inside the axes
                if event.inaxes == ax and event.xdata is not None:
                    x = event.xdata
                    index = np.abs(distances - x).argmin()

                    line.set_xdata([distances[index]])
                    text.set_text(f"{best_focus_from} {round(sizes[index], 3)} $\mu$m\nat {round(distances[index], 5)} m")
                    line.set_visible(True)
                    text.set_visible(True)
                    text.set_position((min((index + 1)/len(sizes), 0.7), 0.6))

                    ax_prof.clear()
                    ax_prof.plot(coords, profiles[index], 'k')
                    ax_prof.set_xlim(coords[0], coords[-1])
                    add_text_1D(ax_prof, dir, sizes[index], distances[index], vpos=0.7)

                    self._wf_prof_figure_3.canvas.draw_idle()

                    self.profile_clicked.emit(dir, index)

            self._wf_prof_figure_3.canvas.mpl_connect('button_press_event', onclick)

            return line, text

        axes = self._wf_prof_figure_3.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
        extent_data_x = np.array([
            bf_propagation_distances_x[0],
            bf_propagation_distances_x[-1],
            x_coordinates[0],
            x_coordinates[-1]])
        extent_data_y = np.array([
            bf_propagation_distances_y[0],
            bf_propagation_distances_y[-1],
            y_coordinates[0],
            y_coordinates[-1]])
        line_h, text_h = plot_ax_plane(axes[1][0], axes[0][0], "x", planes_x, extent_data_x, best_size_value_x, focus_z_position_x, bf_size_values_x, bf_propagation_distances_x, x_coordinates, intensities_x)
        line_v, text_v = plot_ax_plane(axes[1][1], axes[0][1], "y", planes_y, extent_data_y, best_size_value_y, focus_z_position_y, bf_size_values_y, bf_propagation_distances_y, y_coordinates, intensities_y)

        def on_value_changed(index, line, text, ax_prof, sizes, distances, coords, profiles, dir):
            line.set_xdata([distances[index]])
            text.set_text(f"{best_focus_from} {round(sizes[index], 3)} $\mu$m\nat {round(distances[index], 5)} m")
            line.set_visible(True)
            text.set_visible(True)
            text.set_position((min((index + 1) / len(sizes), 0.7), 0.6))

            ax_prof.clear()
            ax_prof.plot(coords, profiles[index], 'k')
            ax_prof.set_xlim(coords[0], coords[-1])
            add_text_1D(ax_prof, dir, sizes[index], distances[index], vpos=0.7)

            self._wf_prof_figure_3.canvas.draw_idle()

        def on_value_changed_h(index_h):
            on_value_changed(index_h,
                             line_h,
                             text_h,
                             axes[0][0],
                             bf_size_values_x,
                             bf_propagation_distances_x,
                             x_coordinates,
                             intensities_x,
                             "x")

        def on_value_changed_v(index_v):
            on_value_changed(index_v,
                             line_v,
                             text_v,
                             axes[0][1],
                             bf_size_values_y,
                             bf_propagation_distances_y,
                             y_coordinates,
                             intensities_y,
                             "y")

        self._slider_h.setMaximum(len(bf_propagation_distances_x)-1)
        self._slider_h.setTickInterval(int(len(bf_propagation_distances_x)/10))
        self._slider_h.setValue(0)
        self._slider_h.value_changed().connect(on_value_changed_h)

        self._slider_v.setMaximum(len(bf_propagation_distances_y)-1)
        self._slider_v.setTickInterval(int(len(bf_propagation_distances_y)/10))
        self._slider_v.setValue(0)
        self._slider_v.value_changed().connect(on_value_changed_v)

        self._wf_prof_figure_3.tight_layout()
        self._wf_prof_figure_3_canvas.draw()


def plot_2D(fig, image, label, p_x, extent_data=None):
    extent_data = np.array([
        -image.shape[1] / 2 * p_x * 1e6,
        image.shape[1] / 2 * p_x * 1e6,
        -image.shape[0] / 2 * p_x * 1e6,
        image.shape[0] / 2 * p_x * 1e6]) if extent_data is None else extent_data

    fig.clear()
    im = fig.gca().imshow(image, interpolation='bilinear', extent=extent_data)
    if sys.platform == 'darwin':  fig.gca().set_position([-0.175, 0.15, 1.0, 0.8])
    else:                         fig.gca().set_position([0.1, 0.15, 0.8, 0.8])
    fig.gca().set_xlabel('x ($\mu$m)', fontsize=22)
    fig.gca().set_ylabel('y ($\mu$m)', fontsize=22)
    cbar = fig.colorbar(mappable=im, ax=fig.gca())
    cbar.set_label(label, rotation=90, fontsize=20)
    fig.gca().set_aspect('equal')

def plot_1D(fig, line_x, line_y, label, p_x, coords=None):
    coords = [(np.arange(len(line_x)) - len(line_x) / 2) * p_x * 1e6,
            (np.arange(len(line_y)) - len(line_y) / 2) * p_x * 1e6] if coords is None else coords

    fig.clear()
    axes = fig.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    axes[0].plot(coords[0], line_x, 'k')
    axes[0].set_xlabel('x ($\mu$m)', fontsize=22)
    axes[0].set_ylabel(label, fontsize=22)
    axes[1].plot(coords[1], line_y, 'k')
    axes[1].set_xlabel('y ($\mu$m)', fontsize=22)
    axes[1].set_ylabel(label, fontsize=22)
    fig.tight_layout()

    return axes

class SliderWithButtons(QWidget):
    def __init__(self):
        super().__init__()

        main_layout = QHBoxLayout()

        # Slider
        self.slider = QSlider(Qt.Horizontal)

        # Buttons layout
        button_layout_left  = QHBoxLayout()
        button_layout_right = QHBoxLayout()
        self.btn_minus = QPushButton("-")
        self.btn_plus  = QPushButton("+")
        self.btn_min   = QPushButton("Min")
        self.btn_max   = QPushButton("Max")

        self.btn_minus.setFixedWidth(20)
        self.btn_plus.setFixedWidth(20)
        self.btn_min.setFixedWidth(30)
        self.btn_max.setFixedWidth(30)

        self.btn_minus.clicked.connect(self.decrease_value)
        self.btn_plus.clicked.connect(self.increase_value)
        self.btn_min.clicked.connect(lambda: self.slider.setValue(self.slider.minimum()))
        self.btn_max.clicked.connect(lambda: self.slider.setValue(self.slider.maximum()))

        button_layout_left.addWidget(self.btn_min)
        button_layout_left.addWidget(self.btn_minus)
        button_layout_right.addWidget(self.btn_plus)
        button_layout_right.addWidget(self.btn_max)

        main_layout.addLayout(button_layout_left)
        main_layout.addWidget(self.slider)
        main_layout.addLayout(button_layout_right)

        self.setLayout(main_layout)

    def setMinimum(self, value=0):   self.slider.setMinimum(value)
    def setMaximum(self, value=100): self.slider.setMaximum(value)
    def setValue(self, value=50):    self.slider.setValue(value)
    def setTickPosition(self, tick_position=QSlider.TicksBelow): self.slider.setTickPosition(tick_position)
    def setTickInterval(self, value=10): self.slider.setTickInterval(value)

    def increase_value(self):
        self.slider.setValue(self.slider.value() + self.slider.singleStep())

    def decrease_value(self):
        self.slider.setValue(self.slider.value() - self.slider.singleStep())

    def value_changed(self): return self.slider.valueChanged