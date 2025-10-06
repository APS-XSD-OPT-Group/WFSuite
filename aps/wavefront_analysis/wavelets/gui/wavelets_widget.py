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
import sys

from aps.common.plot import gui
from aps.common.plot.gui import MessageDialog
from aps.common.plot.splitter import ToggleSplitter, ToggleDirection
from aps.common.widgets.generic_widget import GenericWidget
from aps.common.widgets.congruence import *
from aps.common.scripts.script_data import ScriptData
from aps.common.utilities import list_to_string, string_to_list

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QScrollArea, QWidget
from PyQt5.QtCore import QRect, Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

from aps.wavefront_analysis.common.gui.util import ShowWaitDialog, plot_3D, plot_2D
import aps.wavefront_analysis.driver.wavefront_sensor as ws

import warnings
warnings.filterwarnings("ignore")

DEBUG_MODE = int(os.environ.get("DEBUG_MODE", 0)) == 1

class WaveletsWidget(GenericWidget):
    crop_changed_offline  = pyqtSignal(list)

    def __init__(self, parent, application_name=None, **kwargs):
        self._log_stream_widget         = kwargs["log_stream_widget"]
        self._working_directory         = kwargs["working_directory"]
        self._initialization_parameters = kwargs["initialization_parameters"]

        # METHODS
        self._close                     = kwargs["close_method"]
        self._process_image_WXST        = kwargs["process_image_WXST_method"]
        self._process_images_WSVT       = kwargs["process_images_WSVT_method"]
        self._recrop_from_file          = kwargs["recrop_from_file_method"]

        #SIGNALS
        close_application   = kwargs["close_application_signal"]

        self._set_values_from_initialization_parameters()

        super(WaveletsWidget, self).__init__(parent=parent, application_name=application_name, **kwargs)

        self.crop_changed_offline.connect(self._on_crop_changed)
        close_application.connect(self._close_application_callback)

    def _set_values_from_initialization_parameters(self):
        self.working_directory = self._working_directory

        initialization_parameters: ScriptData = self._initialization_parameters

        self.calculation_type      = initialization_parameters.get_parameter("calculation_type", 0)
        self.plot_rebinning_factor = initialization_parameters.get_parameter("plot_rebinning_factor", 4)
        self.use_flat              = initialization_parameters.get_parameter("use_flat", False)
        self.use_dark              = initialization_parameters.get_parameter("use_dark", False)

        wavelets_analyzer_configuration = initialization_parameters.get_parameter("wavelets_analyzer_configuration")
        common_configuration            = wavelets_analyzer_configuration["common"]
        WXST_configuration              = wavelets_analyzer_configuration["WXST"]
        WSVT_configuration              = wavelets_analyzer_configuration["WSVT"]

        self.distance      = common_configuration["distance"]
        self.energy        = common_configuration["energy"]
        self.pixel_size    = common_configuration["pixel_size"]
        self.scaling_v     = common_configuration["scaling_v"]
        self.scaling_h     = common_configuration["scaling_h"]
        self.use_gpu       = common_configuration["use_gpu"]
        self.use_wavelet   = common_configuration["use_wavelet"]
        self.wavelet_cut   = common_configuration["wavelet_cut"]
        self.pyramid_level = common_configuration["pyramid_level"]
        self.n_iterations  = common_configuration["n_iterations"]
        self.template_size = common_configuration["template_size"]
        self.window_search = common_configuration["window_search"]
        self.crop          = list_to_string(common_configuration["crop"])
        self.down_sampling = common_configuration["down_sampling"]
        self.rebinning     = common_configuration["rebinning"]
        self.n_cores       = common_configuration["n_cores"]
        self.n_group       = common_configuration["n_group"]
        self.save_images   = common_configuration["save_images"]

        self.WXST_image_file_name     = WXST_configuration["WXST_image_file_name"]
        self.WXST_reference_file_name = WXST_configuration["WXST_reference_file_name"]
        self.WXST_dark_file_name      = WXST_configuration["WXST_dark_file_name"]
        self.WXST_flat_file_name      = WXST_configuration["WXST_flat_file_name"]
        self.WXST_result_folder       = WXST_configuration["WXST_result_folder"]

        self.WSVT_image_folder            = WSVT_configuration["WSVT_image_folder"]
        self.WSVT_reference_folder        = WSVT_configuration["WSVT_reference_folder"]
        self.WSVT_result_folder           = WSVT_configuration["WSVT_result_folder"]
        self.WSVT_calculation_half_window = WSVT_configuration["WSVT_calculation_half_window"]
        self.WSVT_n_scan                  = WSVT_configuration["WSVT_n_scan"]

    def get_plot_tab_name(self): return "Wavelets Data Analysis"

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
        wa_tab     = gui.createTabPage(self._input_tab_widget, "Wavelets Analysis")

        self._command_tab_widget = gui.tabWidget(self._command_box)
        ex_tab     = gui.createTabPage(self._command_tab_widget, "Execution")

        labels_width_1 = 300

        #########################################################################################
        # WAVEFRONT ANALYSIS

        if sys.platform == 'darwin' : self._wa_box  = gui.widgetBox(wa_tab, "", width=self._input_box.width()-10, height=self._input_box.height()-40)
        else:                         self._wa_box  = gui.widgetBox(wa_tab, "", width=self._input_box.width()-10, height=self._input_box.height()-40)

        gui.separator(self._wa_box)

        self._wa_tab_widget = gui.tabWidget(self._wa_box)

        tab_common = gui.createTabPage(self._wa_tab_widget, "Setup")
        self._tab_wxst   = gui.createTabPage(self._wa_tab_widget, "WXST")
        self._tab_wsvt   = gui.createTabPage(self._wa_tab_widget, "WXST")

        wa_box_1 = gui.widgetBox(tab_common, "Device", width=self._wa_box.width() - 25, height=90)

        le = gui.lineEdit(wa_box_1, self, "energy", "Energy [eV]", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        font = QFont(le.font())
        font.setBold(True)
        font.setItalic(False)
        font.setPixelSize(14)
        le.setFont(font)
        le.setStyleSheet("QLineEdit {color : darkred}")

        gui.lineEdit(wa_box_1, self, "distance", "Mask-Detector Distance [m]", labelWidth=labels_width_1, orientation='horizontal', valueType=float)

        if sys.platform == 'darwin' : wa_box_2 = gui.widgetBox(tab_common, "Analysis", width=self._wa_box.width()-25, height=400)
        else:                         wa_box_2 = gui.widgetBox(tab_common, "Analysis", width=self._wa_box.width()-25, height=430)

        gui.lineEdit(wa_box_2, self, "pixel_size", label="Pixel Size [m]",      labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_2, self, "scaling_v",  label="Pixel Scaling V [m]", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_2, self, "scaling_h",  label="Pixel Scaling H [m]", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_2, self, "rebinning", label="Image Rebinning Factor", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.lineEdit(wa_box_2, self, "down_sampling", label="Down Sampling", labelWidth=labels_width_1, orientation='horizontal', valueType=float)
        gui.checkBox(wa_box_2, self, "use_wavelet",  "Use Wavelets")
        gui.lineEdit(wa_box_2, self, "wavelet_cut", label="Wavelet Cut", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_2, self, "pyramid_level", label="Pyramid Level", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_2, self, "n_iterations", label="Number of Iterations", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_2, self, "template_size", label="Template Size", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_2, self, "window_search", label="Window Search", labelWidth=labels_width_1, orientation='horizontal', valueType=int)

        self._crop_box = gui.widgetBox(wa_box_2, "", width=wa_box_2.width() - 20, height=30, orientation='horizontal', addSpace=False)

        self.le_crop = gui.lineEdit(self._crop_box, self, "crop", "Crop (-1: auto, n: pixels around center,\n            [b, t, l, r]: coordinates in pixels)",
                                    labelWidth=labels_width_1 - 65, orientation='horizontal', valueType=str)
        gui.button(self._crop_box, self, "Recrop", width=50, callback=self._recrop_from_file_callback)

        gui.lineEdit(wa_box_2, self, "plot_rebinning_factor", label="Rebinning Factor for Crop Image", labelWidth=labels_width_1, orientation='horizontal', valueType=int)

        wa_box_3 = gui.widgetBox(tab_common, "Processing", width=self._wa_box.width()-25, height=120)

        gui.checkBox(wa_box_3, self, "use_gpu",      "Use GPUs")
        gui.lineEdit(wa_box_3, self, "n_cores", label="Number of Cores", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_3, self, "n_group", label="Number of Threads", labelWidth=labels_width_1, orientation='horizontal', valueType=int)

        # WXST
        wa_box_4 = gui.widgetBox(self._tab_wxst, "Input", width=self._wa_box.width() - 25, height=300)

        self._WXST_image_file_name_box = gui.widgetBox(wa_box_4, "", width=wa_box_4.width() - 20, orientation='horizontal', addSpace=False)
        self._le_WXST_image_file_name = gui.lineEdit(self._WXST_image_file_name_box, self, "WXST_image_file_name", "Sample Image At", orientation='vertical', valueType=str)
        gui.button(self._WXST_image_file_name_box, self, "...", width=30, callback=self._set_WXST_image_file_name)

        self._WXST_reference_file_name_box = gui.widgetBox(wa_box_4, "", width=wa_box_4.width() - 20, orientation='horizontal', addSpace=False)
        self._le_WXST_reference_file_name = gui.lineEdit(self._WXST_reference_file_name_box, self, "WXST_reference_file_name", "Reference Image At", orientation='vertical', valueType=str)
        gui.button(self._WXST_reference_file_name_box, self, "...", width=30, callback=self._set_WXST_reference_file_name)

        gui.checkBox(wa_box_4, self, "use_flat", "Use Flat Image", callback=self._set_use_flat)

        self._WXST_flat_file_name_box = gui.widgetBox(wa_box_4, "", width=wa_box_4.width() - 20, orientation='horizontal', addSpace=False)
        self._le_WXST_flat_file_name = gui.lineEdit(self._WXST_flat_file_name_box, self, "WXST_flat_file_name", "Flat Image At", orientation='vertical', valueType=str)
        gui.button(self._WXST_flat_file_name_box, self, "...", width=30, callback=self._set_WXST_flat_file_name)

        gui.checkBox(wa_box_4, self, "use_dark", "Use Dark Image", callback=self._set_use_dark)

        self._WXST_dark_file_name_box = gui.widgetBox(wa_box_4, "", width=wa_box_4.width() - 20, orientation='horizontal', addSpace=False)
        self._le_WXST_dark_file_name = gui.lineEdit(self._WXST_dark_file_name_box, self, "WXST_dark_file_name", "Dark Image At", orientation='vertical', valueType=str)
        gui.button(self._WXST_dark_file_name_box, self, "...", width=30, callback=self._set_WXST_dark_file_name)

        wa_box_5 = gui.widgetBox(self._tab_wxst, "Output", width=self._wa_box.width() - 25, height=80)

        self._WXST_result_folder_box = gui.widgetBox(wa_box_5, "", width=wa_box_5.width() - 20, orientation='horizontal', addSpace=False)
        self._le_WXST_result_folder = gui.lineEdit(self._WXST_result_folder_box, self, "WXST_result_folder", "Result Directory At", orientation='vertical', valueType=str)
        gui.button(self._WXST_result_folder_box, self, "...", width=30, callback=self._set_WXST_result_folder)


        # WSVT
        wa_box_6 = gui.widgetBox(self._tab_wsvt, "Input", width=self._wa_box.width() - 25, height=200)

        self._WSVT_image_folder_box = gui.widgetBox(wa_box_6, "", width=wa_box_6.width() - 20, orientation='horizontal', addSpace=False)
        self._le_WSVT_image_folder = gui.lineEdit(self._WSVT_image_folder_box, self, "WSVT_image_folder", "Sample Images At", orientation='vertical', valueType=str)
        gui.button(self._WSVT_image_folder_box, self, "...", width=30, callback=self._set_WSVT_image_folder)

        self._WSVT_reference_folder_box = gui.widgetBox(wa_box_6, "", width=wa_box_6.width() - 20, orientation='horizontal', addSpace=False)
        self._le_WSVT_reference_folder = gui.lineEdit(self._WSVT_reference_folder_box, self, "WSVT_reference_folder", "Reference Images At", orientation='vertical', valueType=str)
        gui.button(self._WSVT_reference_folder_box, self, "...", width=30, callback=self._set_WSVT_reference_folder)

        gui.lineEdit(wa_box_6, self, "WSVT_calculation_half_window", label="Calculation Half Window", labelWidth=labels_width_1, orientation='horizontal', valueType=int)
        gui.lineEdit(wa_box_6, self, "WSVT_n_scan", label="Number of Scans", labelWidth=labels_width_1, orientation='horizontal', valueType=int)

        wa_box_7 = gui.widgetBox(self._tab_wsvt, "Output", width=self._wa_box.width() - 25, height=80)

        self._WSVT_result_folder_box = gui.widgetBox(wa_box_7, "", width=wa_box_7.width() - 20, orientation='horizontal', addSpace=False)
        self._le_WSVT_result_folder = gui.lineEdit(self._WSVT_result_folder_box, self, "WSVT_result_folder", "Result Directory At", orientation='vertical', valueType=str)
        gui.button(self._WSVT_result_folder_box, self, "...", width=30, callback=self._set_WSVT_result_folder)

        #########################################################################################
        # Execution

        self._ex_box = gui.widgetBox(ex_tab, "", width=self._command_box.width() - 10, height=self._command_box.height() - 85)

        gui.separator(self._ex_box)

        ex_box_0 = gui.widgetBox(self._ex_box , "Application",       width=self._ex_box.width()-5, orientation='vertical', addSpace=False)
        ex_box_2 = gui.widgetBox(self._ex_box , "Data Analysis",     width=self._ex_box.width()-5, orientation='vertical', addSpace=False)

        exit_button = gui.button(ex_box_0, None, "Exit GUI", callback=self._close_callback, width=ex_box_0.width()-20, height=35)
        font = QFont(exit_button.font())
        font.setBold(True)
        font.setItalic(True)
        exit_button.setFont(font)
        palette = QPalette(exit_button.palette())
        palette.setColor(QPalette.ButtonText, QColor('Dark Blue'))
        exit_button.setPalette(palette)

        self._cb_calculation_type = gui.comboBox(ex_box_2, self, "calculation_type",
                                                 label="Calculation Type", orientation='horizontal',
                                                 items=["WXST", "WSVT"], callback=self._set_calculation_type)
        gui.checkBox(ex_box_2, self, "save_images",  "Save Images")

        gui.separator(ex_box_2, 15)
        self._btn_WXST = gui.button(ex_box_2, None, "Process Image WXST", callback=self._process_image_WXST_callback, width=ex_box_2.width() - 20, height=35)
        gui.separator(ex_box_2)
        self._btn_WSVT = gui.button(ex_box_2, None, "Process Images WSVT", callback=self._process_images_WSVT_callback, width=ex_box_2.width() - 20, height=35)

        self._set_calculation_type(init=True)
        self._set_use_flat()
        self._set_use_dark()

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

        self._out_tab_1 = gui.createTabPage(self._out_tab_widget, "Result")
        self._out_tab_2 = gui.createTabPage(self._out_tab_widget, "Log")

        self._wavefront_box = gui.widgetBox(self._out_tab_1, "")
        self._log_box       = gui.widgetBox(self._out_tab_2, "Log", width=self._tab_box.width() - 20, height=self._tab_box.height() - 40)

        self._wf_tab_widget = gui.tabWidget(self._wavefront_box)

        if sys.platform == 'darwin':
            figsize   = (9.52, 5.52)
            figsize_2 = (9.40, 5.15)
        else:
            figsize   = (9.52, 6.52)
            figsize_2 = (9.40, 6.15)

        self._wf_tab_0    = gui.createTabPage(self._wf_tab_widget, "Displacement H")
        self._wf_tab_1    = gui.createTabPage(self._wf_tab_widget, "Displacement V")
        self._wf_tab_2    = gui.createTabPage(self._wf_tab_widget, "Phase")
        self._wf_tab_3    = gui.createTabPage(self._wf_tab_widget, "DPC H")
        self._wf_tab_4    = gui.createTabPage(self._wf_tab_widget, "DPC V")

        self._phase_tab_widget = gui.tabWidget(self._wf_tab_2)

        self._phase_tab_1 = gui.createTabPage(self._phase_tab_widget, "2D")
        self._phase_tab_2 = gui.createTabPage(self._phase_tab_widget, "3D")

        self._wf_box_0     = gui.widgetBox(self._wf_tab_0, "")
        self._wf_box_1     = gui.widgetBox(self._wf_tab_1, "")
        self._phase_box_1  = gui.widgetBox(self._phase_tab_1, "")
        self._phase_box_2  = gui.widgetBox(self._phase_tab_2, "")
        self._wf_box_3     = gui.widgetBox(self._wf_tab_3, "")
        self._wf_box_4     = gui.widgetBox(self._wf_tab_4, "")

        self._wf_dis_x_figure = Figure(figsize=figsize, constrained_layout=True)
        self._wf_dis_x_figure_canvas = FigureCanvas(self._wf_dis_x_figure)
        self._wf_dis_x_scroll = QScrollArea(self._wf_box_0)
        self._wf_dis_x_scroll.setWidget(self._wf_dis_x_figure_canvas)
        self._wf_box_0.layout().addWidget(NavigationToolbar(self._wf_dis_x_figure_canvas, self))
        self._wf_box_0.layout().addWidget(self._wf_dis_x_scroll)

        self._wf_dis_y_figure = Figure(figsize=figsize, constrained_layout=True)
        self._wf_dis_y_figure_canvas = FigureCanvas(self._wf_dis_y_figure)
        self._wf_dis_y_scroll = QScrollArea(self._wf_box_1)
        self._wf_dis_y_scroll.setWidget(self._wf_dis_y_figure_canvas)
        self._wf_box_1.layout().addWidget(NavigationToolbar(self._wf_dis_y_figure_canvas, self))
        self._wf_box_1.layout().addWidget(self._wf_dis_y_scroll)

        self._wf_pha_figure = Figure(figsize=figsize_2, constrained_layout=True)
        self._wf_pha_figure_canvas = FigureCanvas(self._wf_pha_figure)
        self._wf_pha_scroll = QScrollArea(self._phase_box_1)
        self._wf_pha_scroll.setWidget(self._wf_pha_figure_canvas)
        self._phase_box_1.layout().addWidget(NavigationToolbar(self._wf_pha_figure_canvas, self))
        self._phase_box_1.layout().addWidget(self._wf_pha_scroll)
        
        self._wf_pha_3D_figure = Figure(figsize=figsize_2, constrained_layout=True)
        self._wf_pha_3D_figure_canvas = FigureCanvas(self._wf_pha_3D_figure)
        self._wf_pha_3D_scroll = QScrollArea(self._phase_box_2)
        self._wf_pha_3D_scroll.setWidget(self._wf_pha_3D_figure_canvas)
        self._phase_box_2.layout().addWidget(NavigationToolbar(self._wf_pha_3D_figure_canvas, self))
        self._phase_box_2.layout().addWidget(self._wf_pha_3D_scroll)

        self._wf_dpc_x_figure = Figure(figsize=figsize, constrained_layout=True)
        self._wf_dpc_x_figure_canvas = FigureCanvas(self._wf_dpc_x_figure)
        self._wf_dpc_x_scroll = QScrollArea(self._wf_box_4)
        self._wf_dpc_x_scroll.setWidget(self._wf_dpc_x_figure_canvas)
        self._wf_box_3.layout().addWidget(NavigationToolbar(self._wf_dpc_x_figure_canvas, self))
        self._wf_box_3.layout().addWidget(self._wf_dpc_x_scroll)
        
        self._wf_dpc_y_figure = Figure(figsize=figsize, constrained_layout=True)
        self._wf_dpc_y_figure_canvas = FigureCanvas(self._wf_dpc_y_figure)
        self._wf_dpc_y_scroll = QScrollArea(self._wf_box_4)
        self._wf_dpc_y_scroll.setWidget(self._wf_dpc_y_figure_canvas)
        self._wf_box_4.layout().addWidget(NavigationToolbar(self._wf_dpc_y_figure_canvas, self))
        self._wf_box_4.layout().addWidget(self._wf_dpc_y_scroll)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self._log_box.setLayout(layout)
        if not self._log_stream_widget is None:
            self._log_box.layout().addWidget(self._log_stream_widget.get_widget())
            self._log_stream_widget.set_widget_size(width=self._log_box.width() - 15, height=self._log_box.height() - 35)
        else:
            self._log_box.layout().addWidget(QLabel("Log on file only"))

    def _set_use_flat(self):
        self._WXST_flat_file_name_box.setEnabled(bool(self.use_flat))

    def _set_use_dark(self):
        self._WXST_dark_file_name_box.setEnabled(bool(self.use_dark))

    def _set_calculation_type(self, init=False):
        if init: self._wa_tab_widget.removeTab(2)
        self._wa_tab_widget.removeTab(1)

        if self.calculation_type == 0:   self._wa_tab_widget.addTab(self._tab_wxst, "WXST")
        elif self.calculation_type == 1: self._wa_tab_widget.addTab(self._tab_wsvt, "WSVT")
        
        if init: self._wa_tab_widget.setCurrentIndex(0)
        else:    self._wa_tab_widget.setCurrentIndex(1)
        
        self._btn_WXST.setEnabled(self.calculation_type == 0)
        self._btn_WSVT.setEnabled(self.calculation_type == 1)

    def _set_WXST_image_file_name(self):
        self._le_WXST_image_file_name.setText(
            gui.selectFileFromDialog(self,
                                     previous_file_path=self.WXST_image_file_name,
                                     start_directory=self.working_directory,
                                     file_extension_filter="Data Files (*.tif *.hdf5)"))

    def _set_WXST_reference_file_name(self):
        self._le_WXST_reference_file_name.setText(
            gui.selectFileFromDialog(self,
                                     previous_file_path=self.WXST_reference_file_name,
                                     start_directory=self.working_directory,
                                     file_extension_filter="Data Files (*.tif *.hdf5)"))

    def _set_WXST_dark_file_name(self):
        self._le_WXST_dark_file_name.setText(
            gui.selectFileFromDialog(self,
                                     previous_file_path=self.WXST_dark_file_name,
                                     start_directory=self.working_directory,
                                     file_extension_filter="Data Files (*.tif *.hdf5)"))

    def _set_WXST_flat_file_name(self):
        self._le_WXST_flat_file_name.setText(
            gui.selectFileFromDialog(self,
                                     previous_file_path=self.WXST_flat_file_name,
                                     start_directory=self.working_directory,
                                     file_extension_filter="Data Files (*.tif *.hdf5)"))

    def _set_WXST_result_folder(self):
        self._le_WXST_result_folder.setText(
            gui.selectDirectoryFromDialog(self,
                                          previous_directory_path=self.WXST_result_folder,
                                          start_directory=self.working_directory))

    def _set_WSVT_image_folder(self):
        self._le_WSVT_image_folder.setText(
            gui.selectDirectoryFromDialog(self,
                                          previous_directory_path=self.WSVT_image_folder,
                                          start_directory=self.working_directory))

    def _set_WSVT_reference_folder(self):
        self._le_WSVT_reference_folder.setText(
            gui.selectDirectoryFromDialog(self,
                                          previous_directory_path=self.WSVT_reference_folder,
                                          start_directory=self.working_directory))

    def _set_WSVT_result_folder(self):
        self._le_WSVT_result_folder.setText(
            gui.selectDirectoryFromDialog(self,
                                          previous_directory_path=self.WSVT_result_folder,
                                          start_directory=self.working_directory))


    def _check_fields(self, raise_errors=True):
        pass

    def _collect_initialization_parameters(self, raise_errors=True):
        initialization_parameters: ScriptData = self._initialization_parameters

        self._check_fields(raise_errors)

        # -----------------------------------------------------
        # Wavelets Analyzer

        wavelets_analyzer_configuration = initialization_parameters.get_parameter("wavelets_analyzer_configuration")
        common_configuration = wavelets_analyzer_configuration["common"]
        WXST_configuration   = wavelets_analyzer_configuration["WXST"]
        WSVT_configuration   = wavelets_analyzer_configuration["WSVT"]

        common_configuration["distance"]      = self.distance
        common_configuration["energy"]        = self.energy
        common_configuration["pixel_size"]    = self.pixel_size
        common_configuration["scaling_v"]     = self.scaling_v
        common_configuration["scaling_h"]     = self.scaling_h
        common_configuration["use_gpu"]       = self.use_gpu
        common_configuration["use_wavelet"]   = self.use_wavelet
        common_configuration["wavelet_cut"]   = self.wavelet_cut
        common_configuration["pyramid_level"] = self.pyramid_level
        common_configuration["n_iterations"]  = self.n_iterations
        common_configuration["template_size"] = self.template_size
        common_configuration["window_search"] = self.window_search
        common_configuration["crop"]          = string_to_list(self.crop, int)
        common_configuration["down_sampling"] = self.down_sampling
        common_configuration["rebinning"]     = self.rebinning
        common_configuration["n_cores"]       = self.n_cores
        common_configuration["n_group"]       = self.n_group
        common_configuration["save_images"]   = self.save_images

        WXST_configuration["WXST_image_file_name"]     = self.WXST_image_file_name
        WXST_configuration["WXST_reference_file_name"] = self.WXST_reference_file_name
        WXST_configuration["WXST_dark_file_name"]      = self.WXST_dark_file_name
        WXST_configuration["WXST_flat_file_name"]      = self.WXST_flat_file_name
        WXST_configuration["WXST_result_folder"]       = self.WXST_result_folder

        WSVT_configuration["WSVT_image_folder"]            = self.WSVT_image_folder
        WSVT_configuration["WSVT_reference_folder"]        = self.WSVT_reference_folder
        WSVT_configuration["WSVT_result_folder"]           = self.WSVT_result_folder
        WSVT_configuration["WSVT_calculation_half_window"] = self.WSVT_calculation_half_window
        WSVT_configuration["WSVT_n_scan"]                  = self.WSVT_n_scan

        # Widget ini

        initialization_parameters.set_parameter("calculation_type",      self.calculation_type)
        initialization_parameters.set_parameter("plot_rebinning_factor", self.plot_rebinning_factor)
        initialization_parameters.set_parameter("use_dark",              bool(self.use_dark))
        initialization_parameters.set_parameter("use_flat",              bool(self.use_flat))

    def _close_application_callback(self):
        self._collect_initialization_parameters(raise_errors=False)
        self._close(self._initialization_parameters)

    def _close_callback(self):
        if ConfirmDialog.confirmed(self, "Confirm Exit?"):
            self._collect_initialization_parameters(raise_errors=False)
            self._close(self._initialization_parameters)

    def _on_crop_changed(self, crop_array):
        self.crop = list_to_string(crop_array)
        self.le_crop.setText(list_to_string(crop_array))

    # Delegated -------------------------------------------

    def _recrop_from_file_callback(self):
        if self.calculation_type == 0:
            crop_file_name = gui.selectFileFromDialog(self,
                                                      previous_file_path=pathlib.Path(self.WXST_image_file_name).with_suffix('').name,
                                                      start_directory=pathlib.Path(self.WXST_image_file_name).with_suffix('').name,
                                                      file_extension_filter="Data Files (*.tif *.hdf5)")
        else:
            crop_file_name = gui.selectFileFromDialog(self,
                                                     previous_file_path=self.WSVT_image_folder,
                                                     start_directory=self.WSVT_image_folder,
                                                     file_extension_filter="Data Files (*.tif *.hdf5)")


        if not crop_file_name is None:
            dialog = ShowWaitDialog(title="Operation in Progress", text="Reading Image From File", parent=self._tab_box)
            dialog.show()

            def _execute():
                try:
                    self._collect_initialization_parameters(raise_errors=True)
                    self._recrop_from_file(self._initialization_parameters, **{"calling_widget" : self, "crop_file_name" : crop_file_name})
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
        else:
            MessageDialog.message(self, title="Input Error", message="Crop File not selected", type="critical", width=300)

    # -------------------------------------------

    def _process_image_WXST_callback(self):
        dialog = ShowWaitDialog(title="Operation in Progress", text="WXST Processing Image", parent=self._tab_box)
        dialog.show()

        def _execute():
            try:
                self._collect_initialization_parameters(raise_errors=True)
                result_data = self._process_image_WXST(self._initialization_parameters)
                self.__plot_result_data(result_data)
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

    # -------------------------------------------

    def _process_images_WSVT_callback(self):
        dialog = ShowWaitDialog(title="Operation in Progress", text="WSVT Processing Images", parent=self._tab_box)
        dialog.show()

        def _execute():
            try:
                self._collect_initialization_parameters(raise_errors=True)
                result_data = self._process_images_WSVT(self._initialization_parameters)
                self.__plot_result_data(result_data)
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

    # ----------------------------------------------------
    # PLOT METHODS

    def __plot_result_data(self, wavefront_data):
        p_x = ws.PIXEL_SIZE*self.rebinning

        displace  = wavefront_data['displace']
        DPC_x     = wavefront_data['DPC_x']
        DPC_y     = wavefront_data['DPC_y']
        phase     = wavefront_data['phase']

        plot_2D(self._wf_dis_x_figure.figure, displace[0], "[pixels]", p_x)
        self._wf_dis_x_figure_canvas.draw()

        plot_2D(self._wf_dis_y_figure.figure, displace[1], "[pixels]", p_x)
        self._wf_dis_y_figure_canvas.draw()

        plot_2D(self._wf_pha_figure.figure, phase, "[rad]", p_x)
        self._wf_pha_figure_canvas.draw()

        plot_3D(self._wf_pha_3D_figure.figure, phase, "[rad]", p_x, scaling=[self.scaling_h, self.scaling_v])
        self._wf_pha_3D_figure_canvas.draw()

        plot_2D(self._wf_dpc_x_figure.figure, DPC_x, "[rad]", p_x)
        self._wf_dpc_x_figure_canvas.draw()

        plot_2D(self._wf_dpc_y_figure.figure, DPC_y, "[rad]", p_x)
        self._wf_dpc_y_figure_canvas.draw()

        self._out_tab_widget.setCurrentIndex(0)
        self._wf_tab_widget.setCurrentIndex(0)
