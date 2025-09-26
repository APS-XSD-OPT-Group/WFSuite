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
from aps.common.plot import gui
from aps.common.plot.gui import MessageDialog
from aps.common.widgets.generic_widget import GenericWidget
from aps.common.widgets.congruence import *

from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QFont, QPalette, QColor

import warnings
warnings.filterwarnings("ignore")

DEBUG_MODE = int(os.environ.get("DEBUG_MODE", 0)) == 1

class LauncherWidget(GenericWidget):
    def __init__(self, parent, application_name: str, **kwargs):
        super(LauncherWidget, self).__init__(parent=parent, application_name=application_name, **kwargs)

        self._open_absolute_phase         = kwargs["open_absolute_phase_method"]
        self._open_wavelets               = kwargs["open_wavelets_method"]
        self._close                       = kwargs["close_method"]
        self.__initialization_parameters  = kwargs["initialization_parameters"]

        self.set_values_from_initialization_parameters(self.__initialization_parameters)

    def set_values_from_initialization_parameters(self, initialization_parameters):
        pass

    def get_plot_tab_name(self): return "Wavefront Sensor Main Menu"

    def build_widget(self, **kwargs):
        try:    widget_width = kwargs["widget_width"]
        except: widget_width = 320
        try:    widget_height = kwargs["widget_height"]
        except: widget_height = 280

        self.setGeometry(QRect(10,
                               10,
                               widget_width,
                               widget_height))

        self.setFixedWidth(widget_width)
        self.setFixedHeight(widget_height)

        current = self
        while current is not None:

            parent = current.parent()

            if not parent is None: print(parent.size())

            current = parent

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignLeft)
        self.setLayout(layout)

        button_width      = self.width() - 23
        button_height    = 50
        font_size        = 20
        separator        = 10
        tab_box_width    = self.width() - 23
        close_box_height = 60
        main_box_height  = self.height() - close_box_height

        def set_button(button, italic=False, bold=False, color=None):
            font = QFont(button.font())
            font.setBold(bold)
            font.setItalic(italic)
            font.setPixelSize(font_size)
            button.setFont(font)
            if not color is None:
                palette = QPalette(button.palette())
                palette.setColor(QPalette.ButtonText, QColor(color))
                button.setPalette(palette)

        main_box = gui.widgetBox(self, "", width=self.width(), height=main_box_height, orientation="vertical")
        close_box = gui.widgetBox(self, "", width=self.width(), height=close_box_height, orientation="vertical")

        button = gui.button(close_box, None, "Close Application", callback=self.__close_callback, width=button_width + 13, height=button_height)
        set_button(button, True, True, 'Red')

        tabs_collection = gui.tabWidget(main_box, width=self.width() - 10, height=main_box_height - 40)
        gui.separator(main_box, height=10)

        collection_box = gui.widgetBox(gui.createTabPage(tabs_collection, "Operations"), "", width=tab_box_width, orientation="vertical")

        gui.separator(collection_box, height=separator)

        button = gui.button(collection_box, None, "Absolute Phase", callback=self.__open_absolute_phase_callback, width=button_width, height=button_height)
        set_button(button)

        gui.separator(collection_box, height=separator)

        button = gui.button(collection_box, None, "Wavelets", callback=self.__open_wavelets_callback, width=button_width, height=button_height)
        set_button(button)

    def __open_absolute_phase_callback(self):
        try: self._open_absolute_phase(self.__initialization_parameters)
        except Exception as e:
            MessageDialog.message(self, title="Error", message=str(e.args[0]), type="critical", width=500)
            if DEBUG_MODE: raise e

    def __open_wavelets_callback(self):
        try: self._open_wavelets(initialization_parameters=self.__initialization_parameters)
        except Exception as e:
            MessageDialog.message(self, title="Error", message=str(e.args[0]), type="critical", width=500)
            if DEBUG_MODE: raise e

    def __close_callback(self):
        if ConfirmDialog.confirmed(self, "Confirm Exit?"):
            self._close(self.__initialization_parameters)
