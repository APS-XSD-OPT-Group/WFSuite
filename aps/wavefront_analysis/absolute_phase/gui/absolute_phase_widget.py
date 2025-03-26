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

import numpy as np

from aps.common.plot import gui
from aps.common.plot.gui import MessageDialog
from aps.common.widgets.generic_widget import GenericWidget
from aps.common.widgets.congruence import *
from aps.common.singleton import synchronized_method
from aps.common.scripts.script_data import ScriptData
from aps.common.utilities import list_to_string, string_to_list

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QScrollArea
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QFont, QPalette, QColor

class AbsolutePhaseWidget(GenericWidget):
    def __init__(self, parent, application_name=None, **kwargs):
        self._log_stream_widget             = kwargs["log_stream_widget"]
        self._working_directory             = kwargs["working_directory"]
        self._initialization_parameters     = kwargs["initialization_parameters"]

        # METHODS
        self._connect_wavefront_sensor      = kwargs["connect_wavefront_sensor_method"]
        self._close                         = kwargs["close_method"]

        self._set_values_from_initialization_parameters()

        super(AbsolutePhaseWidget, self).__init__(parent=parent, application_name=application_name, **kwargs)

        self.__is_init = True

    def _set_values_from_initialization_parameters(self):
        self.working_directory = self._working_directory

        initialization_parameters: ScriptData = self._initialization_parameters

        self.wavefront_sensor_image_directory = initialization_parameters.get_parameter("wavefront_sensor_image_directory", os.path.join(os.path.abspath(os.curdir), "wf_images"))
        self.save_result                      = initialization_parameters.get_parameter("save_result", True)
        self.plot_raw_image                   = initialization_parameters.get_parameter("plot_raw_image", True)
        self.data_from                        = initialization_parameters.get_parameter("data_from", True)

        # -----------------------------------------------------
        # Wavefront Sensor

        wavefront_sensor_configuration = initialization_parameters.get_parameter("wavefront_sensor_configuration")
    
        self.send_stop_command = wavefront_sensor_configuration["send_stop_command"]
        self.send_save_command = wavefront_sensor_configuration["send_save_command"]
        self.remove_image = wavefront_sensor_configuration["remove_image"]
        self.wait_time = wavefront_sensor_configuration["wait_time"]
        self.exposure_time = wavefront_sensor_configuration["exposure_time"]
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
        self.crop = list_to_string(data_analysis_configuration["crop"])
        self.estimation_method = data_analysis_configuration["estimation_method"]
        self.propagator = data_analysis_configuration["propagator"]
        self.image_ops = list_to_string(data_analysis_configuration["image_ops"])
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
        self.crop_boundary = list_to_string(data_analysis_configuration["crop_boundary"])
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
        except: widget_width = min(1450, geom.width()*0.98)
        try:    widget_height = kwargs["widget_height"]
        except:
            if sys.platform == 'darwin' : widget_height = min(750, geom.height()*0.95)
            else:                         widget_height = min(850, geom.height()*0.95)
        self.setGeometry(QRect(10, 10, widget_width, widget_height))
        self.setFixedWidth(int(widget_width))
        self.setFixedHeight(int(widget_height))

        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignLeft)
        self.setLayout(layout)

        main_box_width = 450

        self._main_box = gui.widgetBox(self, "", width=main_box_width, height=self.height() - 20)

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

        tab_widget = gui.tabWidget( self._main_box)
        ws_tab     = gui.createTabPage(tab_widget, "Wavefront Sensor")
        wa_tab     = gui.createTabPage(tab_widget, "Wavefront Analysis")
        ex_tab     = gui.createTabPage(tab_widget, "Execution")

        labels_width_1 = 300
        labels_width_2 = 150
        labels_width_3 = 100

        #########################################################################################
        # WAVEFRONT SENSOR

        self._ws_box  = gui.widgetBox(ws_tab, "", width=self._main_box.width()-10, height=self._main_box.height()-185)

        gui.separator(self._ws_box)

        self._wavefront_sensor_image_directory_box = gui.widgetBox(self._ws_box , "", width=self._ws_box.width(), orientation='horizontal', addSpace=False)
        self.le_wavefront_sensor_image_directory  = gui.lineEdit(self._wavefront_sensor_image_directory_box, self, "wavefront_sensor_image_directory", "Store image from detector at", orientation='vertical', valueType=str)
        gui.button(self._wavefront_sensor_image_directory_box, self, "...", width=30, callback=self._set_wavefront_sensor_image_directory)

        tab_widget = gui.tabWidget( self._ws_box)
        ws_tab_1     = gui.createTabPage(tab_widget, "Image Capture")
        ws_tab_2     = gui.createTabPage(tab_widget, "IOC")

        ws_box_1 = gui.widgetBox(ws_tab_1, "Execution", width=self._ws_box.width()-15, height=280)

        gui.checkBox(ws_box_1, self, "send_stop_command",      "Send Stop Command")
        gui.checkBox(ws_box_1, self, "send_save_command",      "Send Save Command")
        gui.checkBox(ws_box_1, self, "remove_image",           "Remove Image")
        gui.checkBox(ws_box_1, self, "is_stream_available",    "Is Stream Available")
        gui.checkBox(ws_box_1, self, "transpose_stream_image", "Transpose Stream Image")
        gui.separator(ws_box_1)
        gui.lineEdit(ws_box_1, self, "wait_time",     "Wait Time [s]",         labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(ws_box_1, self, "exposure_time", "Exposure Time [s]",     labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(ws_box_1, self, "pixel_format",  "Pixel Format",          labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(ws_box_1, self, "index_digits",  "Digits on Image Index", labelWidth=labels_width_1, orientation='horizontal', valueType=int)

        ws_box_2 = gui.widgetBox(ws_tab_1, "Detector", width=self._ws_box.width()-15, height=100)

        gui.lineEdit(ws_box_2, self, "pixel_size",          "Pixel Size [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(ws_box_2, self, "detector_resolution", "Resolution [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float)

        ws_box_3 = gui.widgetBox(ws_tab_2, "Epics", width=self._ws_box.width()-15, height=350)

        gui.lineEdit(ws_box_3, self, "cam_pixel_format",      "Cam: Pixel Format",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        gui.lineEdit(ws_box_3, self, "cam_acquire",           "Cam: Acquire",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        gui.lineEdit(ws_box_3, self, "cam_exposure_time",     "Cam: Acquire Time",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        gui.lineEdit(ws_box_3, self, "cam_image_mode",        "Cam: Image Mode",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        gui.lineEdit(ws_box_3, self, "tiff_enable_callbacks", "Tiff: Enable Callbacks",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        gui.lineEdit(ws_box_3, self, "tiff_filename",         "Tiff: File Name",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        gui.lineEdit(ws_box_3, self, "tiff_filepath",         "Tiff: File Path",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        gui.lineEdit(ws_box_3, self, "tiff_filenumber",       "Tiff: File Number",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        gui.lineEdit(ws_box_3, self, "tiff_autosave",         "Tiff: Auto-Save",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        gui.lineEdit(ws_box_3, self, "tiff_savefile",         "Tiff: Write File",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        gui.lineEdit(ws_box_3, self, "tiff_autoincrement",    "Tiff: Auto-Increment",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)
        gui.lineEdit(ws_box_3, self, "pva_image",             "Pva Image",  labelWidth=labels_width_2, orientation='horizontal', valueType=str)

        ws_button = gui.button(self._ws_box, None, "Connect Wavefront Sensor", callback=self._connect_wavefront_sensor_callback, width=self._ws_box.width(), height=60)

        font = QFont(ws_button.font())
        font.setBold(True)
        font.setItalic(False)
        font.setPixelSize(18)
        ws_button.setFont(font)
        palette = QPalette(ws_button.palette())
        palette.setColor(QPalette.ButtonText, QColor('Dark Red'))
        ws_button.setPalette(palette)

        #########################################################################################
        # WAVEFRONT ANALYSIS

        self._wa_box  = gui.widgetBox(wa_tab, "", width=self._main_box.width()-10, height=self._main_box.height()-85)

        gui.separator(self._wa_box)

        tab_widget = gui.tabWidget(self._wa_box)

        tab_1     = gui.createTabPage(tab_widget, "Analysis")
        tab_2     = gui.createTabPage(tab_widget, "Back-Propagation ")

        tab_widget_1 = gui.tabWidget(tab_1)
        tab_widget_2 = gui.tabWidget(tab_2)

        wa_tab_1     = gui.createTabPage(tab_widget_1, "Analysis (1)")
        wa_tab_2     = gui.createTabPage(tab_widget_1, "Analysis (2)")
        wa_tab_5     = gui.createTabPage(tab_widget_1, "Analysis (3)")
        wa_tab_3       = gui.createTabPage(tab_widget_2, "Back-Propagation (1)")
        wa_tab_4       = gui.createTabPage(tab_widget_2, "Back-Propagation (2)")

        wa_box_1 = gui.widgetBox(wa_tab_1, "Mask", width=self._wa_box.width()-25, height=170)

        gui.lineEdit(wa_box_1, self, "pattern_size",          "Pattern Size [m]",           labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_1, self, "pattern_thickness",     "Pattern Thickness [m]",      labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_1, self, "pattern_transmission",  "Pattern Transmission [0,1]", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_1, self, "ran_mask",              "Random Mask",                labelWidth=labels_width_3, orientation='horizontal', valueType=str)
        gui.lineEdit(wa_box_1, self, "propagation_distance",  "Propagation Distance [m]",   labelWidth=labels_width_1, orientation='horizontal', valueType=float)

        wa_box_2 = gui.widgetBox(wa_tab_1, "Source", width=self._wa_box.width()-25, height=170)

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

        wa_box_3 = gui.widgetBox(wa_tab_1, "Image", width=self._wa_box.width()-25, height=130)

        gui.comboBox(wa_box_3, self, "data_from", label="Data From", labelWidth=labels_width_1, orientation='horizontal', items=["stream", "file"])
        gui.lineEdit(wa_box_3, self, "image_ops", "Image Transformations (T, FV, FH)", labelWidth=labels_width_1, orientation='horizontal', valueType=str)
        gui.lineEdit(wa_box_3, self, "crop", "Crop (-1: auto, n: pixels around center,\n            x1, x2, y2, y2: coordinates in pixels)", labelWidth=labels_width_1, orientation='horizontal', valueType=str)

        wa_box_7 = gui.widgetBox(wa_tab_5, "Processing", width=self._wa_box.width()-25, height=120)

        gui.lineEdit(wa_box_7, self, "n_cores", label="Number of Cores", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_7, self, "n_group", label="Number of Threads", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.checkBox(wa_box_7, self, "use_gpu",      "Use GPUs")

        wa_box_4 = gui.widgetBox(wa_tab_5, "Output", width=self._wa_box.width()-25, height=120)

        gui.checkBox(wa_box_4, self, "show_align_figure",  "Show Align Figure")
        gui.checkBox(wa_box_4, self, "correct_scale",      "Correct Scale")
        le = gui.lineEdit(wa_box_4, self, "image_transfer_matrix", "Image Transfer Matrix", labelWidth=labels_width_1, orientation='horizontal', valueType=str)
        le.setReadOnly(True)
        font = QFont(le.font())
        font.setBold(True)
        font.setItalic(False)
        le.setFont(font)
        le.setStyleSheet("QLineEdit {color : darkgreen; background : rgb(243, 240, 160)}")

        wa_box_5 = gui.widgetBox(wa_tab_2, "Simulated Mask", width=self._wa_box.width()-25, height=90)

        gui.checkBox(wa_box_5, self, "d_source_recal",  "Source Distance Recalculation", callback=self._set_d_source_recal)
        self.le_estimation_method = gui.lineEdit(wa_box_5, self, "estimation_method", "Method", labelWidth=labels_width_1, orientation='horizontal', valueType=str)

        wa_box_6 = gui.widgetBox(wa_tab_2, "Reconstruction", width=self._wa_box.width()-25, height=380)

        gui.lineEdit(wa_box_6, self, "mode", label="Mode (area, lineWidth)", labelWidth=labels_width_1, orientation='horizontal', valueType=str)
        gui.lineEdit(wa_box_6, self, "line_width", label="Line Width", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "rebinning", label="Image Rebinning Factor", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_6, self, "down_sampling", label="Down Sampling", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_6, self, "method", label="Method (WXST, SPINNet)", labelWidth=labels_width_1, orientation='horizontal', valueType=str, callback=self._set_method)
        gui.checkBox(wa_box_6, self, "use_wavelet",  "Use Wavelets")

        gui.lineEdit(wa_box_6, self, "wavelet_cut", label="Wavelet Cut", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "pyramid_level", label="Pyramid Level", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "n_iterations", label="Number of Iterations", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "template_size", label="Template Size", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "window_search", label="Window Search", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "crop_boundary", "Boundary Crop (same format as Crop)", labelWidth=labels_width_1, orientation='horizontal', valueType=str)


        #########################################################################################
        # Back-Propagation

        bp_box_1 = gui.widgetBox(wa_tab_3, "Propagation", width=self._wa_box.width()-25, height=260)

        self.le_kind                   = gui.lineEdit(bp_box_1, self, "kind", label="Kind (1D, 2D)", labelWidth=labels_width_1, orientation='horizontal',  valueType=str, callback=self._set_kind)

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

        bp_box_2 = gui.widgetBox(wa_tab_3, "Image", width=self._wa_box.width()-25, height=240)

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

        bp_box_3 = gui.widgetBox(wa_tab_4, "Best Focus", width=self._wa_box.width()-25, height=270)

        gui.checkBox(bp_box_3, self, "scan_best_focus", "Scan Best Focus")
        gui.checkBox(bp_box_3, self, "use_fit",         "Use Polynomial Fit")
        gui.lineEdit(bp_box_3, self, "best_focus_from",   label="Besto Focus From (rms, fwhm)",   labelWidth=labels_width_1, orientation='horizontal', valueType=str)

        self.kind_box_1_2 = gui.widgetBox(bp_box_3, "", width=bp_box_1.width()-20, height=50)
        self.kind_box_2_2 = gui.widgetBox(bp_box_3, "", width=bp_box_1.width()-20, height=50)

        gui.lineEdit(self.kind_box_1_2, self, "best_focus_scan_range",   label="Range [m] (start, stop, step)",   labelWidth=200, orientation='horizontal', valueType=str)
        gui.lineEdit(self.kind_box_2_2, self, "best_focus_scan_range_h", label="Range H [m] (start, stop, step)", labelWidth=200, orientation='horizontal', valueType=str)
        gui.lineEdit(self.kind_box_2_2, self, "best_focus_scan_range_v", label="Range V [m] (start, stop, step)", labelWidth=200, orientation='horizontal', valueType=str)

        gui.lineEdit(bp_box_3, self, "rms_range_h", label="R.M.S. Range H [m] (start, stop)", labelWidth=220, orientation='horizontal', valueType=str)
        gui.lineEdit(bp_box_3, self, "rms_range_v", label="R.M.S. Range V [m] (start, stop)", labelWidth=220, orientation='horizontal', valueType=str)

        self._set_d_source_recal()
        self._set_kind()

        #########################################################################################
        # BACK-PROPAGATION


        #########################################################################################
        #########################################################################################
        # output
        #########################################################################################
        #########################################################################################

        self._out_box     = gui.widgetBox(self, "", width=self.width() - main_box_width - 20, height=self.height() - 20, orientation="vertical")
        self._ws_dir_box  = gui.widgetBox(self._out_box, "", width=self._out_box.width(), height=50)

        self.le_working_directory = gui.lineEdit(self._ws_dir_box, self, "working_directory", "  Working Directory", labelWidth=150, orientation='horizontal', valueType=str)
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

        self._image_figure = Figure(figsize=(9.65, 5.9), constrained_layout=True)
        self._image_figure_canvas = FigureCanvas(self._image_figure)
        self._image_scroll = QScrollArea(self._image_box)
        self._image_scroll.setWidget(self._image_figure_canvas)
        self._image_box.layout().addWidget(NavigationToolbar(self._image_figure_canvas, self))
        self._image_box.layout().addWidget(self._image_scroll)

        self._wf_tab_widget = gui.tabWidget(self._wavefront_box)

        self._wf_tab_0 = gui.createTabPage(self._wf_tab_widget, "At Detector")
        self._wf_tab_1 = gui.createTabPage(self._wf_tab_widget, "Back Propagated")
        self._wf_tab_2 = gui.createTabPage(self._wf_tab_widget, "Longitudinal Profiles")

        self._wf_box_0     = gui.widgetBox(self._wf_tab_0, "")
        self._wf_box_1     = gui.widgetBox(self._wf_tab_1, "")
        self._wf_box_2     = gui.widgetBox(self._wf_tab_2, "")

        self._wf_det_figure = Figure(figsize=(9.65, 5.9), constrained_layout=True)
        self._wf_det_figure_canvas = FigureCanvas(self._wf_det_figure)
        self._wf_det_scroll = QScrollArea(self._wf_box_0)
        self._wf_det_scroll.setWidget(self._wf_det_figure_canvas)
        self._wf_box_0.layout().addWidget(NavigationToolbar(self._wf_det_figure_canvas, self))
        self._wf_box_0.layout().addWidget(self._wf_det_scroll)

        self._wf_prop_figure = Figure(figsize=(9.65, 5.9), constrained_layout=True)
        self._wf_prop_figure_canvas = FigureCanvas(self._wf_prop_figure)
        self._wf_prop_scroll = QScrollArea(self._wf_box_1)
        self._wf_prop_scroll.setWidget(self._wf_prop_figure_canvas)
        self._wf_box_1.layout().addWidget(NavigationToolbar(self._wf_prop_figure_canvas, self))
        self._wf_box_1.layout().addWidget(self._wf_prop_scroll)

        self._wf_prof_figure = Figure(figsize=(9.65, 5.9), constrained_layout=True)
        self._wf_prof_figure_canvas = FigureCanvas(self._wf_prof_figure)
        self._wf_prof_scroll = QScrollArea(self._wf_box_2)
        self._wf_prof_scroll.setWidget(self._wf_prof_figure_canvas)
        self._wf_box_2.layout().addWidget(NavigationToolbar(self._wf_prof_figure_canvas, self))
        self._wf_box_2.layout().addWidget(self._wf_prof_scroll)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self._log_box.setLayout(layout)
        if not self._log_stream_widget is None:
            self._log_box.layout().addWidget(self._log_stream_widget.get_widget())
            self._log_stream_widget.set_widget_size(width=self._log_box.width() - 15, height=self._log_box.height() - 35)
        else:
            self._log_box.layout().addWidget(QLabel("Log on file only"))

    def _set_wavefront_sensor_image_directory(self):
        self.le_wavefront_sensor_image_directory.setText(
            gui.selectDirectoryFromDialog(self,
                                          previous_directory_path=self.wavefront_sensor_image_directory,
                                          start_directory=self.working_directory))

    def _set_d_source_recal(self):
        self.le_estimation_method.setEnabled(bool(self.d_source_recal))

    def _set_kind(self):
        if not self.kind in ["2D", "1D"]: MessageDialog.message(self, title="Input Error", message="Kind must be '2D' or '1D'", type="critical", width=500)
        else:
            self.kind_box_1_1.setVisible(self.kind=="2D")
            self.kind_box_1_2.setVisible(self.kind=="2D")
            self.kind_box_2_1.setVisible(self.kind=="1D")
            self.kind_box_2_2.setVisible(self.kind=="1D")

    def _set_method(self):
        if not self.method in ["WXST", "SPINNet"]: MessageDialog.message(self, title="Input Error", message="Method must be 'WXST' or 'SPINNet'", type="critical", width=500)
        else:
            self.delta_f_h = self._delta_f_h[self.method]
            self.delta_f_v = self._delta_f_v[self.method]
            self.le_delta_f_h.setText(str(self.delta_f_h))
            self.le_delta_f_v.setText(str(self.delta_f_v))

    def _set_delta_f(self):
        if not self.method in ["WXST", "SPINNet"]: MessageDialog.message(self, title="Input Error", message="Method must be 'WXST' or 'SPINNet'", type="critical", width=500)
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
        data_analysis_configuration["crop"] = string_to_list(self.crop, int)
        data_analysis_configuration["estimation_method"] = self.estimation_method
        data_analysis_configuration["propagator"] = self.propagator
        data_analysis_configuration["image_ops"] = string_to_list(self.image_ops, str)
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
        data_analysis_configuration["crop_boundary"] = string_to_list(self.crop_boundary, int)
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
        initialization_parameters.set_parameter("save_result",                      bool(self.save_result))
        initialization_parameters.set_parameter("plot_raw_image",                   bool(self.plot_raw_image))
        initialization_parameters.set_parameter("data_from",                        self.data_from)

    def _close_callback(self):
        if ConfirmDialog.confirmed(self, "Confirm Exit?"):
            self._collect_initialization_parameters(raise_errors=False)
            self._close(self._initialization_parameters)

    def _connect_wavefront_sensor_callback(self):
        try:
            self._collect_initialization_parameters(raise_errors=True)
            self._connect_wavefront_sensor(self._initialization_parameters)

            MessageDialog.message(self, title="Wavefront Sensor", message="Wavefront Sensor is connected", type="information", width=500)
        except ValueError as error:
            MessageDialog.message(self, title="Input Error", message=error.args[0], type="critical", width=500)
        except Exception as exception:
            MessageDialog.message(self, title="Unexpected Exception", message=exception.args[0], type="critical", width=700)

    @synchronized_method
    def analysis_completed(self):
        pass
