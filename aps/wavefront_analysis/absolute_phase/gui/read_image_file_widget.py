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
import sys
import numpy as np
import copy

from PyQt5.QtCore import pyqtSignal

from aps.common.widgets.generic_widget import GenericWidget

from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import RectangleSelector
from cmasher import cm as cmm

from warnings import filterwarnings
filterwarnings("ignore")

class PlotImageFile(GenericWidget):
    def get_plot_tab_name(self): return "Read Image File"

    def build_widget(self, **kwargs):
        self.__calling_widget = kwargs.get("calling_widget", None)

        super(PlotImageFile, self).build_widget(**kwargs)

    def build_mpl_figure(self, **kwargs):
        data_2D = kwargs["image"]
        hh      = kwargs["h_coord"]
        vv      = kwargs["v_coord"][::-1]
        hh_orig = copy.deepcopy(hh)
        vv_orig = copy.deepcopy(vv)

        self.__plot_rebinning_factor = kwargs["plot_rebinning_factor"]
        self.__pixel_size            = kwargs["pixel_size"]

        if self.__plot_rebinning_factor > 1:
            height, width = data_2D.shape
            if height % self.__plot_rebinning_factor != 0 or width % self.__plot_rebinning_factor != 0:
                raise ValueError("Image dimensions must be divisible by the rebinning factor.")

            new_shape = (height // self.__plot_rebinning_factor, self.__plot_rebinning_factor, width // self.__plot_rebinning_factor, self.__plot_rebinning_factor)
            data_2D   = data_2D.reshape(new_shape).mean(axis=(1, 3))
            hh        = hh.reshape((width // self.__plot_rebinning_factor, self.__plot_rebinning_factor)).mean(axis=1)
            vv        = vv.reshape((height // self.__plot_rebinning_factor, self.__plot_rebinning_factor)).mean(axis=1)

        xrange = [np.min(hh), np.max(hh)]
        yrange = [np.min(vv), np.max(vv)]

        if sys.platform == 'darwin':  figure = Figure(figsize=(9.65, 5.9), constrained_layout=True)
        else:                         figure = Figure(figsize=(9.65, 6.9), constrained_layout=True)
        figure.clear()

        def custom_formatter(x, pos): return f'{x:.2f}'

        axis = figure.gca()
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

        if sys.platform == 'darwin': axis.set_position([-0.1, 0.15, 1.0, 0.8])
        else:                        axis.set_position([0.15, 0.15, 0.8, 0.8])

        cbar = figure.colorbar(mappable=plotted_image, ax=axis, pad=0.03, aspect=30, shrink=0.6)
        cbar.ax.text(0.5, 1.05, "Intensity", transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=10, color="black")

        def set_crop(crop_array):
            if not self.__calling_widget is None:
                self.__calling_widget.crop_changed_offline.emit(crop_array)

        def onselect(eclick, erelease):
            if eclick.button == 3:  # right click
                axis.set_xlim(xrange[0], xrange[1])
                axis.set_ylim(yrange[0], yrange[1])

                set_crop([0, -1, 0, -1])
            elif eclick.button == 1:
                if self.__plot_rebinning_factor > 1:
                    dimensions = [data_2D.shape[0] * self.__plot_rebinning_factor, data_2D.shape[1] * self.__plot_rebinning_factor]
                    pixel_size = self.__pixel_size * self.__plot_rebinning_factor * 1e3  # mm
                else:
                    dimensions = data_2D.shape
                    pixel_size = self.__pixel_size * 1e3  # mm

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

        def toggle_selector(event): pass

        toggle_selector.RS = RectangleSelector(axis, onselect,
                                               props=dict(facecolor='purple',
                                                          edgecolor='black',
                                                          alpha=0.2,
                                                          fill=True))
        toggle_selector.RS.set_active(True)

        figure.canvas.mpl_connect('key_press_event', toggle_selector)

        return figure
