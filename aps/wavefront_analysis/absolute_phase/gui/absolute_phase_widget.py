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
from aps.common.plot.splitter import ToggleSplitter, ToggleDirection
from aps.common.widgets.generic_widget import GenericWidget
from aps.common.widgets.congruence import *
from aps.common.scripts.script_data import ScriptData
from aps.common.utilities import list_to_string, string_to_list

from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.gridspec import GridSpec
from cmasher import cm as cmm

from AnyQt.QtWidgets import QHBoxLayout, QVBoxLayout, QScrollArea, QSlider, QWidget
from AnyQt.QtCore import QRect, Qt, pyqtSignal, QTimer
from AnyQt.QtGui import QFont, QPalette, QColor

from aps.wavefront_analysis.common.gui.util import ShowWaitDialog, SliderWithButtons, plot_1D, plot_2D
import aps.wavefront_analysis.driver.wavefront_sensor as ws

import warnings
warnings.filterwarnings("ignore")

DEBUG_MODE = int(os.environ.get("DEBUG_MODE", 0)) == 1

class AbsolutePhaseWidget(GenericWidget):
    synchronize_wavefront_sensor = pyqtSignal()
    profile_clicked              = pyqtSignal(str, int)
    crop_changed_offline         = pyqtSignal(list)

    def __init__(self, parent, application_name=None, **kwargs):
        self._log_stream_widget             = kwargs["log_stream_widget"]
        self._working_directory             = kwargs["working_directory"]
        self._initialization_parameters     = kwargs["initialization_parameters"]

        # METHODS
        self._close                          = kwargs["close_method"]
        self._image_files_parameters_changed = kwargs["image_files_parameters_changed_method"]
        self._take_shot                      = kwargs["take_shot_method"]
        self._take_shot_as_flat_image        = kwargs["take_shot_as_flat_image_method"]
        self._read_image_from_file           = kwargs["read_image_from_file_method"]
        self._generate_mask                  = kwargs["generate_mask_method"]
        self._process_image                  = kwargs["process_image_method"]
        self._back_propagate                 = kwargs["back_propagate_method"]

        #SIGNALS
        crop_changed_online = kwargs["crop_changed_signal"]
        close_application   = kwargs["close_application_signal"]

        self._set_values_from_initialization_parameters()

        super(AbsolutePhaseWidget, self).__init__(parent=parent, application_name=application_name, **kwargs)

        self.profile_clicked.connect(self._on_profile_clicked)
        self.synchronize_wavefront_sensor.connect(self._on_synchronize_wavefront_sensor)
        self.crop_changed_offline.connect(self._on_crop_changed_offline)
        crop_changed_online.connect(self._on_crop_changed_online)
        close_application.connect(self._close_application_callback)

    def _set_values_from_initialization_parameters(self):
        self.working_directory = self._working_directory

        initialization_parameters: ScriptData = self._initialization_parameters

        self.wavefront_sensor_mode     = initialization_parameters.get_parameter("wavefront_sensor_mode", 0)
        self.plot_rebinning_factor     = initialization_parameters.get_parameter("plot_rebinning_factor", 4)

        self.image_index               = initialization_parameters.get_parameter("image_index", 1)
        self.file_name_type            = initialization_parameters.get_parameter("file_name_type", 0)
        self.index_digits_custom       = initialization_parameters.get_parameter("index_digits_custom", 5)
        self.pixel_size_type           = initialization_parameters.get_parameter("pixel_size_type", 0)
        self.pixel_size_custom         = initialization_parameters.get_parameter("pixel_size_custom", ws.PIXEL_SIZE)
        self.file_name_prefix_custom   = initialization_parameters.get_parameter("file_name_prefix_custom", "custom_file_prefix")
        self.image_directory           = initialization_parameters.get_parameter("image_directory", os.path.join(os.path.abspath(os.curdir), "wf_images"))
        self.simulated_mask_directory  = initialization_parameters.get_parameter("simulated_mask_directory", os.path.join(self.image_directory, "simulated_mask"))
        self.use_flat                  = initialization_parameters.get_parameter("use_flat", False)
        self.use_dark                  = initialization_parameters.get_parameter("use_dark", False)
        self.save_images               = initialization_parameters.get_parameter("save_images", True)
        self.bp_calibration_mode       = initialization_parameters.get_parameter("bp_calibration_mode", False)
        self.bp_plot_shift             = initialization_parameters.get_parameter("bp_plot_shift", True)

        absolute_phase_analyzer_configuration = initialization_parameters.get_parameter("absolute_phase_analyzer_configuration")
        data_analysis_configuration = absolute_phase_analyzer_configuration["data_analysis"]
        back_propagation_configuration = absolute_phase_analyzer_configuration["back_propagation"]
    
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
        self.filter_intensity = back_propagation_configuration["filter_intensity"]
        self.sigma_intensity = back_propagation_configuration["sigma_intensity"]
        self.smooth_phase = back_propagation_configuration["smooth_phase"]
        self.filter_phase = back_propagation_configuration["filter_phase"]
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

        self.engine = back_propagation_configuration["engine"]

        # WOFRY
        self.magnification_v = back_propagation_configuration["magnification_v"]
        self.magnification_h = back_propagation_configuration["magnification_h"]
        self.shift_half_pixel = back_propagation_configuration["shift_half_pixel"]

        # SRW
        self.auto_resize_before_propagation                         = back_propagation_configuration["auto_resize_before_propagation"]
        self.auto_resize_after_propagation                          = back_propagation_configuration["auto_resize_after_propagation"]
        self.relative_precision_for_propagation_with_autoresizing   = back_propagation_configuration["relative_precision_for_propagation_with_autoresizing"]
        self.allow_semianalytical_treatment_of_quadratic_phase_term = back_propagation_configuration["allow_semianalytical_treatment_of_quadratic_phase_term"]
        self.do_any_resizing_on_fourier_side_using_fft              = back_propagation_configuration["do_any_resizing_on_fourier_side_using_fft"]
        self.horizontal_range_modification_factor_at_resizing       = back_propagation_configuration["horizontal_range_modification_factor_at_resizing"]
        self.horizontal_resolution_modification_factor_at_resizing  = back_propagation_configuration["horizontal_resolution_modification_factor_at_resizing"]
        self.vertical_range_modification_factor_at_resizing         = back_propagation_configuration["vertical_range_modification_factor_at_resizing"]
        self.vertical_resolution_modification_factor_at_resizing    = back_propagation_configuration["vertical_resolution_modification_factor_at_resizing"]

        self.scan_best_focus = back_propagation_configuration["scan_best_focus"]
        self.use_fit = back_propagation_configuration["use_fit"]
        self.best_focus_from = back_propagation_configuration["best_focus_from"]
        self.best_focus_scan_range   = list_to_string(back_propagation_configuration["best_focus_scan_range"])
        self.best_focus_scan_range_v = list_to_string(back_propagation_configuration["best_focus_scan_range_v"])
        self.best_focus_scan_range_h = list_to_string(back_propagation_configuration["best_focus_scan_range_h"])

    def get_plot_tab_name(self): return "Wavefront Data Analysis"

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
        wa_tab     = gui.createTabPage(self._input_tab_widget, "Wavefront Analysis")

        self._command_tab_widget = gui.tabWidget(self._command_box)
        ex_tab     = gui.createTabPage(self._command_tab_widget, "Execution")

        labels_width_1 = 300
        labels_width_2 = 150
        labels_width_3 = 100

        #########################################################################################
        # WAVEFRONT ANALYSIS

        def emit_synchronize_wavefront_sensor(): self.synchronize_wavefront_sensor.emit()

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

        if sys.platform == 'darwin' : wa_box_3 = gui.widgetBox(wa_tab_1, "Files", width=self._wa_box.width()-25, height=210)
        else:                         wa_box_3 = gui.widgetBox(wa_tab_1, "Files", width=self._wa_box.width()-25, height=240)

        self._image_directory_box = gui.widgetBox(wa_box_3, "", width=wa_box_3.width() - 20, orientation='horizontal', addSpace=False)
        self._le_image_directory = gui.lineEdit(self._image_directory_box, self, "image_directory", "Image At", orientation='horizontal', valueType=str)
        gui.button(self._image_directory_box, self, "...", width=30, callback=self._set_image_directory)
        self._le_image_directory.textChanged.connect(emit_synchronize_wavefront_sensor)

        self._le_image_index                = gui.lineEdit(wa_box_3, self, "image_index",  "Image Index", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        self._cb_ws_file_name_type          = gui.comboBox(wa_box_3, self, "file_name_type", label="File Name From", labelWidth=labels_width_1, orientation='horizontal', items=["W.S. Configuration", "Custom"], callback=self._set_file_name_type)
        self._le_ws_index_digits_custom     = gui.lineEdit(wa_box_3, self, "index_digits_custom",  "Digits on Image Index", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        self._le_ws_file_name_prefix_custom = gui.lineEdit(wa_box_3, self, "file_name_prefix_custom", "Custom Prefix", labelWidth=120, orientation='horizontal', valueType=str)

        self._cb_ws_file_name_type.currentIndexChanged.connect(emit_synchronize_wavefront_sensor)
        self._le_ws_index_digits_custom.textChanged.connect(emit_synchronize_wavefront_sensor)
        self._le_ws_file_name_prefix_custom.textChanged.connect(emit_synchronize_wavefront_sensor)

        self._simulated_mask_directory_box = gui.widgetBox(wa_box_3, "", width=wa_box_3.width() - 20, orientation='horizontal', addSpace=False)
        self._le_simulated_mask_directory = gui.lineEdit(self._simulated_mask_directory_box, self, "simulated_mask_directory", "Sim. Mask at", orientation='horizontal', valueType=str)
        gui.button(self._simulated_mask_directory_box, self, "...", width=30, callback=self._set_simulated_mask_directory)

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

        if sys.platform == 'darwin' : wa_box_6 = gui.widgetBox(wa_tab_2, "Reconstruction", width=self._wa_box.width()-25, height=450)
        else:                         wa_box_6 = gui.widgetBox(wa_tab_2, "Reconstruction", width=self._wa_box.width()-25, height=500)

        gui.checkBox(wa_box_6, self, "use_flat", "Use Flat Image")
        gui.checkBox(wa_box_6, self, "use_dark", "Use Dark Image")

        self._crop_box = gui.widgetBox(wa_box_6, "", width=wa_box_6.width() - 20, height=30, orientation='horizontal', addSpace=False)

        self.le_crop = gui.lineEdit(self._crop_box, self, "crop", "Crop (-1: auto, n: pixels around center,\n            [b, t, l, r]: coordinates in pixels)",
                                    labelWidth=labels_width_1, orientation='horizontal', valueType=str)

        gui.lineEdit(wa_box_6, self, "mode", label="Mode (area, centralLine)", labelWidth=labels_width_1, orientation='horizontal', valueType=str)
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

        if sys.platform == 'darwin' : bp_box_1 = gui.widgetBox(wa_tab_3, "Propagation", width=self._wa_box.width()-25, height=350)
        else:                         bp_box_1 = gui.widgetBox(wa_tab_3, "Propagation", width=self._wa_box.width()-25, height=380)

        self.le_kind  = gui.lineEdit(bp_box_1, self, "kind", label="Kind (1D, 2D)", labelWidth=labels_width_1, orientation='horizontal',  valueType=str, callback=self._set_kind)

        self.kind_box_1_1 = gui.widgetBox(bp_box_1, "", width=bp_box_1.width()-20, height=30)
        self.kind_box_2_1 = gui.widgetBox(bp_box_1, "", width=bp_box_1.width()-20, height=30, orientation='horizontal' )

        gui.lineEdit(self.kind_box_1_1, self, "distance",   label="Propagation Distance [m] (<0)", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(self.kind_box_2_1, self, "distance_h", label="Propagation Distance H  [m] (<0)",  labelWidth=labels_width_2, orientation='horizontal', valueType=float)
        gui.lineEdit(self.kind_box_2_1, self, "distance_v", label="V  [m] (<0)",  labelWidth=labels_width_3, orientation='horizontal', valueType=float)

        self.le_delta_f_h = gui.lineEdit(bp_box_1, self, "delta_f_h", label="Phase Shift H [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float, callback=self._set_delta_f)
        self.le_delta_f_v = gui.lineEdit(bp_box_1, self, "delta_f_v", label="Phase Shift V [m]",  labelWidth=labels_width_1, orientation='horizontal', valueType=float, callback=self._set_delta_f)

        self.le_engine = gui.lineEdit(bp_box_1, self, "engine", label="Engine (WOFRY, SRW)", labelWidth=labels_width_1, orientation='horizontal', valueType=str, callback=self._set_engine)

        self.bp_box_1_1 = gui.widgetBox(bp_box_1, "", width=bp_box_1.width() - 20)
        self.bp_box_1_2 = gui.widgetBox(bp_box_1, "", width=bp_box_1.width() - 20)

        # WOFRY
        gui.lineEdit(self.bp_box_1_1, self, "magnification_h", label="Magnification H", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(self.bp_box_1_1, self, "magnification_v", label="Magnification V", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.checkBox(self.bp_box_1_1, self, "shift_half_pixel",  "Shift Half Pixel")

        # SRW
        box = gui.widgetBox(self.bp_box_1_2, "", orientation="horizontal")
        gui.comboBox(box, self, "auto_resize_before_propagation", label="Auto Resize Before",
                     items=["No", "Yes"], labelWidth=labels_width_2, sendSelectedValue=False, orientation="horizontal")

        gui.comboBox(box, self, "auto_resize_after_propagation", label="After Propagation",
                     items=["No", "Yes"], labelWidth=labels_width_2, sendSelectedValue=False, orientation="horizontal")

        gui.lineEdit(self.bp_box_1_2, self, "relative_precision_for_propagation_with_autoresizing", "Autoresizing relative precision (1.0 nominal)", labelWidth=labels_width_1, valueType=float, orientation="horizontal")

        gui.comboBox(self.bp_box_1_2, self, "allow_semianalytical_treatment_of_quadratic_phase_term", label="Propagator",
                     items=["Standard", "Quadratic Term", "Quadratic Term Special", "From Waist", "To Waist"], labelWidth=labels_width_2,
                     sendSelectedValue=False, orientation="horizontal")

        gui.comboBox(self.bp_box_1_2, self, "do_any_resizing_on_fourier_side_using_fft", label="Do any resizing on fourier side using fft",
                     items=["No", "Yes"], labelWidth=labels_width_1, sendSelectedValue=False, orientation="horizontal")

        box = gui.widgetBox(self.bp_box_1_2, "", orientation="horizontal")
        gui.lineEdit(box, self, "horizontal_range_modification_factor_at_resizing", "H modification factor: range", labelWidth=labels_width_2+20, valueType=float, orientation="horizontal")
        gui.lineEdit(box, self, "horizontal_resolution_modification_factor_at_resizing", "resolution", labelWidth=labels_width_3-20, valueType=float, orientation="horizontal")
        box = gui.widgetBox(self.bp_box_1_2, "", orientation="horizontal")
        gui.lineEdit(box, self, "vertical_range_modification_factor_at_resizing", "V modification factor: range", labelWidth=labels_width_2+20, valueType=float, orientation="horizontal")
        gui.lineEdit(box, self, "vertical_resolution_modification_factor_at_resizing", "resolution", labelWidth=labels_width_3-20, valueType=float, orientation="horizontal")

        if sys.platform == 'darwin' : bp_box_2 = gui.widgetBox(wa_tab_3, "Image", width=self._wa_box.width()-25, height=200)
        else:                         bp_box_2 = gui.widgetBox(wa_tab_3, "Image", width=self._wa_box.width()-25, height=220)

        gui.lineEdit(bp_box_2, self, "rebinning_bp", label="Wavefront Rebinning Factor", labelWidth=labels_width_1, orientation='horizontal', valueType=float)

        box = gui.widgetBox(bp_box_2, "", orientation="horizontal")
        gui.checkBox(box, self, "smooth_intensity", "Smooth Intensity")
        gui.lineEdit(box, self, "filter_intensity", label="filter", labelWidth=50, orientation='horizontal', valueType=str)
        gui.lineEdit(box, self, "sigma_intensity", label="\u03c3", labelWidth=20, orientation='horizontal', valueType=float)

        box = gui.widgetBox(bp_box_2, "", orientation="horizontal")
        gui.checkBox(box, self, "smooth_phase", "Smooth Phase    ")
        gui.lineEdit(box, self, "filter_phase", label="filter", labelWidth=50, orientation='horizontal', valueType=str)
        gui.lineEdit(box, self, "sigma_phase", label="\u03c3", labelWidth=20, orientation='horizontal', valueType=float)

        box = gui.widgetBox(bp_box_2, "", orientation="horizontal")

        gui.lineEdit(box, self, "crop_h",       label="Crop H", labelWidth=labels_width_3, orientation='horizontal', valueType=int)
        gui.lineEdit(box, self, "crop_shift_h", label="Shift H", labelWidth=labels_width_3, orientation='horizontal', valueType=int)

        box = gui.widgetBox(bp_box_2, "", orientation="horizontal")

        gui.lineEdit(box, self, "crop_v",       label="Crop V", labelWidth=labels_width_3, orientation='horizontal', valueType=int)
        gui.lineEdit(box, self, "crop_shift_v", label="Shift V", labelWidth=labels_width_3, orientation='horizontal', valueType=int)

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

        self._set_file_name_type()
        self._set_method()
        self._set_d_source_recal()
        self._set_kind()
        self._set_engine()
        self._set_scan_best_focus()

        #########################################################################################
        # Execution

        self._ex_box = gui.widgetBox(ex_tab, "", width=self._command_box.width() - 10, height=self._command_box.height() - 85)

        gui.separator(self._ex_box)

        ex_box_0 = gui.widgetBox(self._ex_box , "Application",       width=self._ex_box.width()-5, orientation='vertical', addSpace=False)
        ex_box_1 = gui.widgetBox(self._ex_box , "Wavefront Sensor",  width=self._ex_box.width()-5, orientation='vertical', addSpace=False)
        ex_box_2 = gui.widgetBox(self._ex_box , "Data Analysis",     width=self._ex_box.width()-5, orientation='vertical', addSpace=False)

        exit_button = gui.button(ex_box_0, None, "Exit GUI", callback=self._close_callback, width=ex_box_0.width()-20, height=35)
        font = QFont(exit_button.font())
        font.setBold(True)
        font.setItalic(True)
        exit_button.setFont(font)
        palette = QPalette(exit_button.palette())
        palette.setColor(QPalette.ButtonText, QColor('Dark Blue'))
        exit_button.setPalette(palette)

        self._cb_mode = gui.comboBox(ex_box_1, self, "wavefront_sensor_mode", label="Mode",
                                     items=["Connected to W.S.", "Standalone"], sendSelectedValue=False, orientation="horizontal",
                                     callback=self._set_wavefront_sensor_mode)

        gui.lineEdit(ex_box_1, self, "plot_rebinning_factor", label="Rebinning Factor", orientation='horizontal', valueType=int)
        gui.separator(ex_box_1)
        gui.widgetLabel(ex_box_1, "For Tiff File only:")
        self._cb_pixel_size_type = gui.comboBox(ex_box_1, self, "pixel_size_type", label="Pixel Size From", labelWidth=labels_width_1, orientation='horizontal', items=["W.S.", "Custom"], callback=self._set_pixel_size_type)
        self._le_pixel_size      = gui.lineEdit(ex_box_1, self, "pixel_size_custom", label="Pixel Size [m]", orientation='horizontal', valueType=float)

        gui.separator(ex_box_1, 15)

        self._btn_take_shot = gui.button(ex_box_1, None, "Take Shot", callback=self._take_shot_callback, width=ex_box_1.width()-20, height=35)
        gui.separator(ex_box_1)
        self._btn_take_shot_as_flat_image = gui.button(ex_box_1, None, "Take Shot As Flat Image", callback=self._take_shot_as_flat_image_callback, width=ex_box_1.width()-20, height=35)
        gui.separator(ex_box_1)
        gui.button(ex_box_1, None, "Read Image From File", callback=self._read_image_from_file_callback, width=ex_box_1.width()-20, height=35)

        gui.button(ex_box_2, None, "Generate Mask", callback=self._generate_mask_callback, width=ex_box_2.width() - 20, height=35)
        gui.separator(ex_box_2)
        gui.button(ex_box_2, None, "Process Image", callback=self._process_image_callback, width=ex_box_2.width() - 20, height=35)
        gui.separator(ex_box_2)
        gui.button(ex_box_2, None, "Back-Propagate", callback=self._back_propagate_callback, width=ex_box_2.width() - 20, height=35)

        self._set_wavefront_sensor_mode()

        #########################################################################################
        #########################################################################################
        # output
        #########################################################################################
        #########################################################################################

        self._out_box     = gui.widgetBox(self, "", width=self.width() - main_box_width - 20, height=self.height() - 20, orientation="vertical")
        self._ws_dir_box  = gui.widgetBox(self._out_box, "", width=self._out_box.width(), height=50, orientation="horizontal")

        self.le_working_directory = gui.lineEdit(self._ws_dir_box, self, "working_directory", "  Configuration Directory", labelWidth=120, orientation='horizontal', valueType=str)
        self.le_working_directory.setReadOnly(True)
        font = QFont(self.le_working_directory.font())
        font.setBold(True)
        font.setItalic(False)
        self.le_working_directory.setFont(font)
        self.le_working_directory.setStyleSheet("QLineEdit {color : darkgreen; background : rgb(243, 240, 160)}")

        self._tab_box    = gui.widgetBox(self._out_box, "", width=self._out_box.width(), height=self._out_box.height() - 55, orientation="vertical")

        self._out_tab_widget = gui.tabWidget(self._tab_box)

        self._out_tab_1 = gui.createTabPage(self._out_tab_widget, "Wavefront")
        self._out_tab_2 = gui.createTabPage(self._out_tab_widget, "Log")

        self._wavefront_box = gui.widgetBox(self._out_tab_1, "")
        self._log_box       = gui.widgetBox(self._out_tab_2, "Log", width=self._tab_box.width() - 20, height=self._tab_box.height() - 40)

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

    def _on_synchronize_wavefront_sensor(self):
        self.file_name_type = self._cb_ws_file_name_type.currentIndex()

        self._collect_initialization_parameters(raise_errors=False)
        self._image_files_parameters_changed(self._initialization_parameters)

    def _set_pixel_size_type(self):
        self._le_pixel_size.setEnabled(self.pixel_size_type==1)

    def _set_wavefront_sensor_mode(self):
        default_bg = QColor("grey").name()
        default_fg = QColor("white").name()

        if self.wavefront_sensor_mode == 0: # online
            bg = QColor("darkgreen").name()

            self._btn_take_shot.setEnabled(True)
            self._btn_take_shot_as_flat_image.setEnabled(True)
            self._le_image_index.setEnabled(False)

            self.pixel_size_type = 0 # from WS
            self._cb_pixel_size_type.setCurrentIndex(0)
            self._cb_pixel_size_type.setEnabled(False)
            self._set_pixel_size_type()

            self._collect_initialization_parameters(raise_errors=False)
            self._image_files_parameters_changed(self._initialization_parameters)
        else:
            bg = QColor("darkred").name()

            self._btn_take_shot.setEnabled(False)
            self._btn_take_shot_as_flat_image.setEnabled(False)
            self._le_image_index.setEnabled(True)
            self._cb_pixel_size_type.setEnabled(True)
            self._set_pixel_size_type()

        self._cb_mode.setStyleSheet(f"""
            QComboBox {{
                background-color: {bg};
                color: {default_fg};
                font-weight: bold;
            }}
            QComboBox QAbstractItemView {{
                background: {default_bg};
                color: {default_fg};
            }}
        """)



    def _set_file_name_type(self):
        self._le_ws_index_digits_custom.setEnabled(self.file_name_type == 1)
        self._le_ws_file_name_prefix_custom.setEnabled(self.file_name_type == 1)

    def _set_image_directory(self):
        self._le_image_directory.setText(
            gui.selectDirectoryFromDialog(self,
                                          previous_directory_path=self.image_directory,
                                          start_directory=self.working_directory))

    def _set_simulated_mask_directory(self):
        self._le_simulated_mask_directory.setText(
            gui.selectDirectoryFromDialog(self,
                                          previous_directory_path=self.simulated_mask_directory,
                                          start_directory=self.image_directory))

    def _set_d_source_recal(self):
        self.le_estimation_method.setEnabled(bool(self.d_source_recal))

    def _set_kind(self):
        if not self.kind in ["2D", "1D"]: MessageDialog.message(self, title="Input Error", message="Kind must be '2D' or '1D'", type="critical", width=500)
        else:
            self.kind_box_1_1.setVisible(self.kind=="2D")
            self.kind_box_1_2.setVisible(self.kind=="2D")
            self.kind_box_2_1.setVisible(self.kind=="1D")
            self.kind_box_2_2.setVisible(self.kind=="1D")

    def _set_engine(self):
        if not self.engine.lower() in ["wofry", "srw"]: MessageDialog.message(self, title="Input Error", message="Engine be 'WOFRY' or 'SRW'", type="critical", width=500)
        else:
            self.bp_box_1_1.setVisible(self.engine.lower()=="wofry")
            self.bp_box_1_2.setVisible(self.engine.lower()=="srw")

    def _set_scan_best_focus(self):
        self._bp_box_3_1.setEnabled(bool(self.scan_best_focus))

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
        # Wavefront Analyzer

        absolute_phase_analyzer_configuration = initialization_parameters.get_parameter("absolute_phase_analyzer_configuration")
        data_analysis_configuration      = absolute_phase_analyzer_configuration["data_analysis"]
        back_propagation_configuration   = absolute_phase_analyzer_configuration["back_propagation"]

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
        back_propagation_configuration["filter_intensity"] = self.filter_intensity
        back_propagation_configuration["sigma_intensity"] = self.sigma_intensity
        back_propagation_configuration["smooth_phase"] = self.smooth_phase
        back_propagation_configuration["filter_phase"] = self.filter_phase
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
        back_propagation_configuration["engine"]           = self.engine
        back_propagation_configuration["rms_range_v"]      = string_to_list(self.rms_range_v, float)
        back_propagation_configuration["rms_range_h"]      = string_to_list(self.rms_range_h, float)
        back_propagation_configuration["magnification_v"]  = self.magnification_v
        back_propagation_configuration["magnification_h"]  = self.magnification_h
        back_propagation_configuration["shift_half_pixel"] = self.shift_half_pixel

        back_propagation_configuration["auto_resize_before_propagation"] = self.auto_resize_before_propagation
        back_propagation_configuration["auto_resize_after_propagation"] = self.auto_resize_after_propagation
        back_propagation_configuration["relative_precision_for_propagation_with_autoresizing"] = self.relative_precision_for_propagation_with_autoresizing
        back_propagation_configuration["allow_semianalytical_treatment_of_quadratic_phase_term"] = self.allow_semianalytical_treatment_of_quadratic_phase_term
        back_propagation_configuration["do_any_resizing_on_fourier_side_using_fft"] = self.do_any_resizing_on_fourier_side_using_fft
        back_propagation_configuration["horizontal_range_modification_factor_at_resizing"] = self.horizontal_range_modification_factor_at_resizing
        back_propagation_configuration["horizontal_resolution_modification_factor_at_resizing"] = self.horizontal_resolution_modification_factor_at_resizing
        back_propagation_configuration["vertical_range_modification_factor_at_resizing"] = self.vertical_range_modification_factor_at_resizing
        back_propagation_configuration["vertical_resolution_modification_factor_at_resizing"] = self.vertical_resolution_modification_factor_at_resizing

        back_propagation_configuration["scan_best_focus"]  = self.scan_best_focus
        back_propagation_configuration["use_fit"]          = self.use_fit
        back_propagation_configuration["best_focus_from"]  = self.best_focus_from
        back_propagation_configuration["best_focus_scan_range"]   = string_to_list(self.best_focus_scan_range, float)
        back_propagation_configuration["best_focus_scan_range_v"] = string_to_list(self.best_focus_scan_range_v, float)
        back_propagation_configuration["best_focus_scan_range_h"] = string_to_list(self.best_focus_scan_range_h, float)

        # Widget ini

        initialization_parameters.set_parameter("wavefront_sensor_mode",    self.wavefront_sensor_mode)
        initialization_parameters.set_parameter("plot_rebinning_factor",    self.plot_rebinning_factor)

        initialization_parameters.set_parameter("image_index",              self.image_index)
        initialization_parameters.set_parameter("file_name_type",           self.file_name_type)
        initialization_parameters.set_parameter("index_digits_custom",      self.index_digits_custom)
        initialization_parameters.set_parameter("pixel_size_type",          self.pixel_size_type)
        initialization_parameters.set_parameter("pixel_size_custom",        self.pixel_size_custom)
        initialization_parameters.set_parameter("file_name_prefix_custom",  self.file_name_prefix_custom)
        initialization_parameters.set_parameter("image_directory",          self.image_directory)
        initialization_parameters.set_parameter("simulated_mask_directory", self.simulated_mask_directory)
        initialization_parameters.set_parameter("use_dark",                 bool(self.use_dark))
        initialization_parameters.set_parameter("use_flat",                 bool(self.use_flat))
        initialization_parameters.set_parameter("save_images",              bool(self.save_images))
        initialization_parameters.set_parameter("bp_calibration_mode",      bool(self.bp_calibration_mode))
        initialization_parameters.set_parameter("bp_plot_shift",            bool(self.bp_plot_shift))

    def _close_application_callback(self):
        self._collect_initialization_parameters(raise_errors=False)
        self._close(self._initialization_parameters)

    def _close_callback(self):
        if ConfirmDialog.confirmed(self, "Confirm Exit?"):
            self._collect_initialization_parameters(raise_errors=False)
            self._close(self._initialization_parameters)

    def _on_crop_changed_offline(self, crop_array):
        if self.wavefront_sensor_mode == 1: self._on_crop_changed(crop_array)

    def _on_crop_changed_online(self, crop_array):
        if self.wavefront_sensor_mode == 0: self._on_crop_changed(crop_array)

    def _on_crop_changed(self, crop_array):
        self.crop = list_to_string(crop_array)
        self.le_crop.setText(list_to_string(crop_array))

    def _on_profile_clicked(self, direction, index):
        if direction == "x":   self._slider_h.setValue(index)
        elif direction == "y": self._slider_v.setValue(index)

    # Delegated -------------------------------------------

    def _take_shot_callback(self):
        self._take_shot()

    def _take_shot_as_flat_image_callback(self):
        self._take_shot_as_flat_image()

    def _read_image_from_file_callback(self):
        if self.wavefront_sensor_mode == 0: # online
            self._read_image_from_file(None)
        else:
            dialog = ShowWaitDialog(title="Operation in Progress", text="Reading Image From File", parent=self._tab_box)
            dialog.show()

            def _execute():
                try:
                    self._collect_initialization_parameters(raise_errors=True)
                    self._read_image_from_file(self._initialization_parameters, **{"calling_widget" : self})
                except ValueError as error:
                    MessageDialog.message(self, title="Input Error", message=str(error.args[0]), type="critical", width=500)
                    if DEBUG_MODE: raise error
                except Exception as exception:
                    MessageDialog.message(self, title="Unexpected Exception", message=str(exception.args[0]), type="critical", width=700)
                    if DEBUG_MODE: raise exception
                finally:
                    dialog.accept()
            try:
                QTimer.singleShot(100, _execute)
            except:
                pass

    # Offline -------------------------------------------

    def _generate_mask_callback(self):
        dialog = ShowWaitDialog(title="Operation in Progress", text="Generating Mask", parent=self._tab_box)
        dialog.show()

        def _execute():
            try:
                self._collect_initialization_parameters(raise_errors=True)
                image_transfer_matrix = self._generate_mask(self._initialization_parameters)
                self._manage_generate_mask_result(image_transfer_matrix)
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

    def _process_image_callback(self):
        dialog = ShowWaitDialog(title="Operation in Progress", text="Processing Image", parent=self._tab_box)
        dialog.show()

        def _execute():
            try:
                self._collect_initialization_parameters(raise_errors=True)
                wavefront_at_detector_data = self._process_image(self._initialization_parameters)
                self.__plot_wavefront_at_detector(wavefront_at_detector_data)
            except ValueError as error:
                MessageDialog.message(self, title="Input Error", message=str(error), type="critical", width=500)
                if DEBUG_MODE: raise error
            except Exception as exception:
                MessageDialog.message(self, title="Unexpected Exception", message=str(exception), type="critical", height=400, width=700)
                if DEBUG_MODE: raise exception
            finally:
                dialog.accept()

        try:    QTimer.singleShot(100, _execute)
        except: pass

    def _back_propagate_callback(self):
        dialog = ShowWaitDialog(title="Operation in Progress", text="Back-Propagating", parent=self._tab_box)
        dialog.show()

        def _execute():
            try:
                self._collect_initialization_parameters(raise_errors=True)
                propagated_wavefront_data = self._back_propagate(self._initialization_parameters)
                self._manage_back_propagate_result(propagated_wavefront_data)

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

    def __plot_wavefront_at_detector(self, wavefront_data):
        if self.wavefront_sensor_mode == 1 and self.pixel_size_type == 1: pixel_size = self.pixel_size_custom
        else:                                                             pixel_size = ws.PIXEL_SIZE
        p_x = pixel_size*self.rebinning

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

        elif wavefront_data['mode'] == 'centralLine':
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

        self._out_tab_widget.setCurrentIndex(0)
        self._wf_tab_widget.setCurrentIndex(0)
        self._wf_tab_0_widget.setCurrentIndex(0)

    def __plot_back_propagated_wavefront(self, wavefront_data):
        def add_text_2D(ax):
            text = "Wavefront Properties:\n"
            for prop, label in zip(["fwhm_x", "fwhm_y", "sigma_x", "sigma_y", "wf_position_x", "wf_position_y"],
                                   ["fwhm(x)", "fwhm(y)", "rms(x)", "rms(y)", "shift(x)", "shift(y)"]):
                text += "\n" + rf"{label:<8}: {wavefront_data[prop]*1e6 : 3.3f} $\mu$m"

            if sys.platform == 'darwin': ax.text(0.05, 0.55, text, color="black", alpha=0.9, fontsize=12, fontname="Courier",
                                                 bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7), transform=ax.transAxes)
            else:                        ax.text(0.05, 0.55, text, color="black", alpha=0.9, fontsize=12, fontname="DejaVu Sans",
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

            gs = GridSpec(nrows=1, ncols=2, width_ratios=[3, 1], wspace=0.05)

            ax     = fig.add_subplot(gs[0, 0])
            ax_txt = fig.add_subplot(gs[0, 1], frame_on=False)
            ax_txt.axis('off')

            def custom_formatter(x, pos): return f'{x:.2f}'

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

            cbar = fig.colorbar(mappable=image, ax=ax, pad=0.1, aspect=30, shrink=0.6)
            cbar.ax.text(0.5, 1.05, "Intensity", transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=10, color="black")

            add_text_2D(ax_txt)

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

            self._wf_int_prop_figure.figure.clear() # left empty
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

        self._out_tab_widget.setCurrentIndex(0)
        self._wf_tab_widget.setCurrentIndex(1)
        self._wf_tab_1_widget.setCurrentIndex(0)

    def __plot_longitudinal_profiles(self, profiles_data):
        best_x_coordinates = 1e6 * profiles_data['bf_x_coordinate']
        best_y_coordinates = 1e6 * profiles_data['bf_y_coordinate']

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

        x_coordinates = [1e6 * coord for coord in profiles_data['bf_x_coordinates']]
        y_coordinates = [1e6 * coord for coord in profiles_data['bf_y_coordinates']]

        axes = plot_1D(self._wf_prof_figure_2.figure, intensity_x, intensity_y, "[counts]", None, coords=[best_x_coordinates, best_y_coordinates])
        axes[0].set_xlim(best_x_coordinates[0], best_x_coordinates[-1])
        axes[1].set_xlim(best_y_coordinates[0], best_y_coordinates[-1])
        axes[0].axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
        axes[1].axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
        add_text_1D(axes[0], "x", best_size_value_x, focus_z_position_x)
        add_text_1D(axes[1], "y", best_size_value_y, focus_z_position_y)
        self._wf_prof_figure_2_canvas.draw()

        # Propagation planes

        # TODO: find the maximum extent, then create a space where to accomodate the same resolution of the smallest.
        #       fill each line with interpolation

        extension_x = abs(np.max(x_coordinates) - np.min(x_coordinates))
        extension_y = abs(np.max(y_coordinates) - np.min(y_coordinates))

        best_extension_x = abs(np.max(best_x_coordinates) - np.min(best_x_coordinates))
        best_extension_y = abs(np.max(best_y_coordinates) - np.min(best_y_coordinates))

        factor_x = extension_x/best_extension_x
        factor_y = extension_y/best_extension_y

        new_x_coordinates = np.linspace(np.min(x_coordinates), np.max(x_coordinates), int(len(best_x_coordinates) * factor_x))
        new_y_coordinates = np.linspace(np.min(y_coordinates), np.max(y_coordinates), int(len(best_y_coordinates) * factor_y))

        planes_x      = np.zeros((len(new_x_coordinates), len(bf_propagation_distances_x)))
        planes_y      = np.zeros((len(new_y_coordinates), len(bf_propagation_distances_y)))

        from scipy.interpolate import interp1d

        for i in range(planes_x.shape[1]): planes_x[:, i] = interp1d(x_coordinates[i], intensities_x[i], kind="cubic", bounds_error=False, fill_value=[0.0])(new_x_coordinates)
        for i in range(planes_y.shape[1]): planes_y[:, i] = interp1d(y_coordinates[i], intensities_y[i], kind="cubic", bounds_error=False, fill_value=[0.0])(new_y_coordinates)

        self._wf_prof_figure_3.clear()

        def plot_ax_plane(ax, ax_prof, dir, planes, extent_data, best_size, focus, sizes, distances, coords):
            ax.imshow(planes, interpolation='bilinear', extent=extent_data)
            ax.set_xlabel(f"p. distance {dir} (m)", fontsize=22)
            ax.set_ylabel(f"{dir} ($\mu$m)", fontsize=22)
            ax.set_aspect('auto')
            ax.axvline(focus, color="gray", ls="--", linewidth=2, alpha=0.9)
            ax.text(0.53, 0.81, f"{best_focus_from} {round(best_size, 3)} $\mu$m\nat {round(focus, 5)} m", color="blue", alpha=0.9, fontsize=11, fontname=("Courier" if sys.platform == 'darwin' else "DejaVu Sans"),
                         bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7), transform=ax.transAxes)

            index = np.abs(distances - focus).argmin()
            ax_prof.plot(coords, planes[:, index], 'k')
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
                    ax_prof.plot(coords, planes[:, index], 'k')
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
            new_x_coordinates[0],
            new_x_coordinates[-1]])
        extent_data_y = np.array([
            bf_propagation_distances_y[0],
            bf_propagation_distances_y[-1],
            new_y_coordinates[0],
            new_y_coordinates[-1]])
        line_h, text_h = plot_ax_plane(axes[1][0], axes[0][0], "x", planes_x, extent_data_x, best_size_value_x, focus_z_position_x, bf_size_values_x, bf_propagation_distances_x, new_x_coordinates)
        line_v, text_v = plot_ax_plane(axes[1][1], axes[0][1], "y", planes_y, extent_data_y, best_size_value_y, focus_z_position_y, bf_size_values_y, bf_propagation_distances_y, new_y_coordinates)

        def plot_slider(slider, ax_prof, dir, sizes, distances, coords, planes, line, text):
            slider.setMaximum(len(bf_propagation_distances_x) - 1)
            slider.setTickInterval(int(len(bf_propagation_distances_x) / 10))
            slider.setValue(0)

            def on_value_changed(index):
                line.set_xdata([distances[index]])
                text.set_text(f"{best_focus_from} {round(sizes[index], 3)} $\mu$m\nat {round(distances[index], 5)} m")
                line.set_visible(True)
                text.set_visible(True)
                text.set_position((min((index + 1) / len(sizes), 0.7), 0.6))

                ax_prof.clear()
                ax_prof.plot(coords, planes[:, index], 'k')
                ax_prof.set_xlim(coords[0], coords[-1])
                add_text_1D(ax_prof, dir, sizes[index], distances[index], vpos=0.7)

                self._wf_prof_figure_3.canvas.draw_idle()

            try:    slider.value_changed().disconnect()
            except: pass
            slider.value_changed().connect(on_value_changed)

        plot_slider(self._slider_h, axes[0][0], "x", bf_size_values_x, bf_propagation_distances_x, new_x_coordinates, planes_x, line_h, text_h)
        plot_slider(self._slider_v, axes[0][1], "y", bf_size_values_y, bf_propagation_distances_y, new_y_coordinates, planes_y, line_v, text_v)

        self._wf_prof_figure_3.tight_layout()
        self._wf_prof_figure_3_canvas.draw()