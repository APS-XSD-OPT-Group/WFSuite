#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2025, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2025. UChicago Argonne, LLC. This software was produced       #
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
import numpy as np
import sys

from PyQt5.QtWidgets import QDialog, QLabel, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QSlider
from PyQt5.QtCore import QRect, Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap


class ShowWaitDialog(QDialog):
    def __init__(self, title="", text="", width=500, height=80, parent=None, color_string="139, 0, 0"):
        QDialog.__init__(self, parent)

        self.setModal(True)
        self.setWindowTitle(title)
        self.setFixedWidth(int(width))
        self.setFixedHeight(int(height))

        layout = QVBoxLayout(self)

        label = QLabel()
        label.setFixedWidth(int(width * 0.95))
        label.setText(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font: 14px")
        layout.addWidget(label)

        label = QLabel()
        label.setFixedWidth(int(width * 0.95))
        label.setText("Please wait....")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"font: bold italic 16px; color: rgb({color_string});")
        layout.addWidget(label)


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