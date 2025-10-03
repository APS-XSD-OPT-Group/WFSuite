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

import os
import glob
import random
import time
import pathlib
from threading import Thread

import numpy as np
import json

from aps.wavefront_analysis.wavelets.legacy.WSVT_executor import execute_process_images
from aps.wavefront_analysis.wavelets.legacy.WXST_executor import execute_process_image
from aps.wavefront_analysis.wavelets.facade import IWaveletsAnalyzer, ProcessingMode, MAX_THREADS
import aps.wavefront_analysis.driver.wavefront_sensor as ws

from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance

APPLICATION_NAME = "WAVELETS-ANALYSIS"

register_ini_instance(IniMode.LOCAL_JSON_FILE,
                      ini_file_name=".wavelets_analysis.json",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)


ENERGY        = ini_file.get_float_from_ini(  section="Common", key="Energy",        default=12398.0)
DISTANCE      = ini_file.get_float_from_ini(  section="Common", key="Distance",      default=500e-3)
PIXEL_SIZE    = ini_file.get_float_from_ini(  section="Common", key="Pixel-Size",    default=ws.PIXEL_SIZE)
SCALING_H     = ini_file.get_float_from_ini(  section="Common", key="Scaling-V",     default=1.0)
SCALING_V     = ini_file.get_float_from_ini(  section="Common", key="Scaling-H",     default=1.0)
USE_GPU       = ini_file.get_boolean_from_ini(section="Common", key="Use-Gpu",       default=False)
USE_WAVELET   = ini_file.get_boolean_from_ini(section="Common", key="Use-Wavelet",   default=False)
WAVELET_CUT   = ini_file.get_int_from_ini(    section="Common", key="Wavelet-Cut",   default=2)
PYRAMID_LEVEL = ini_file.get_int_from_ini(    section="Common", key="Pyramid-Level", default=1)
N_ITERATIONS  = ini_file.get_int_from_ini(    section="Common", key="N-Iterations",  default=1)
TEMPLATE_SIZE = ini_file.get_int_from_ini(    section="Common", key="Template-Size", default=21)
WINDOW_SEARCH = ini_file.get_int_from_ini(    section="Common", key="Window-Search", default=20)
CROP          = ini_file.get_list_from_ini(   section="Common", key="Crop",          default=[-1], type=int)
DOWN_SAMPLING = ini_file.get_float_from_ini(  section="Common", key="Down-Sampling", default=1.0)
REBINNING     = ini_file.get_float_from_ini(  section="Common", key="Rebinning",     default=1.0)
N_CORES       = ini_file.get_int_from_ini(    section="Common", key="N-Cores",       default=16)
N_GROUP       = ini_file.get_int_from_ini(    section="Common", key="N-Group",       default=1)
SAVE_IMAGES   = ini_file.get_boolean_from_ini(section="Common", key="Save-Images",   default=False)
VERBOSE       = ini_file.get_boolean_from_ini(section="Common", key="Verbose",       default=False)

# WXST --------------------------------------------
WXST_IMAGE_FILE_NAME     = ini_file.get_string_from_ini( section="WXST", key="Image-File-Name",     default=os.path.join(os.path.abspath(os.curdir), "sample_00001.tif"))
WXST_REFERENCE_FILE_NAME = ini_file.get_string_from_ini( section="WXST", key="Reference-File-Name", default=os.path.join(os.path.abspath(os.curdir), "ref_00001.tif"))
WXST_DARK_FILE_NAME      = ini_file.get_string_from_ini( section="WXST", key="Dark-File-Name",      default=None)
WXST_FLAT_FILE_NAME      = ini_file.get_string_from_ini( section="WXST", key="Flat-File-Name",      default=None)
WXST_RESULT_FOLDER       = ini_file.get_string_from_ini( section="WXST", key="Result-Folder",       default=os.path.join(os.path.abspath(os.curdir), "output"))

# WSVT --------------------------------------------
WSVT_IMAGE_FOLDER            = ini_file.get_string_from_ini( section="WSVT", key="Image-Folder",            default=os.path.join(os.path.abspath(os.curdir), "images"))
WSVT_REFERENCE_FOLDER        = ini_file.get_string_from_ini( section="WSVT", key="Reference-Folder",        default=os.path.join(os.path.abspath(os.curdir), "references"))
WSVT_RESULT_FOLDER           = ini_file.get_string_from_ini( section="WSVT", key="Result-Folder",           default=os.path.join(os.path.abspath(os.curdir), "output"))
WSVT_CALCULATION_HALF_WINDOW = ini_file.get_int_from_ini(    section="WSVT", key="Calculation-Half-Window", default=20)
WSVT_N_SCAN                  = ini_file.get_int_from_ini(    section="WSVT", key="N-Scan",                  default=1)


def store():
    ini_file.set_value_at_ini(section="Common", key="Energy",        value=ENERGY)
    ini_file.set_value_at_ini(section="Common", key="Distance",      value=DISTANCE)
    ini_file.set_value_at_ini(section="Common", key="Pixel-Size",    value=PIXEL_SIZE)
    ini_file.set_value_at_ini(section="Common", key="Scaling-V",     value=SCALING_V)
    ini_file.set_value_at_ini(section="Common", key="Scaling-H",     value=SCALING_H)
    ini_file.set_value_at_ini(section="Common", key="Use-Gpu",       value=USE_GPU)
    ini_file.set_value_at_ini(section="Common", key="Use-Wavelet",   value=USE_WAVELET)
    ini_file.set_value_at_ini(section="Common", key="Wavelet-Cut",   value=WAVELET_CUT)
    ini_file.set_value_at_ini(section="Common", key="Pyramid-Level", value=PYRAMID_LEVEL)
    ini_file.set_value_at_ini(section="Common", key="N-Iterations",  value=N_ITERATIONS)
    ini_file.set_value_at_ini(section="Common", key="Template-Size", value=TEMPLATE_SIZE)
    ini_file.set_value_at_ini(section="Common", key="Window-Search", value=WINDOW_SEARCH)
    ini_file.set_list_at_ini( section="Common", key="Crop",          values_list=CROP)
    ini_file.set_value_at_ini(section="Common", key="Down-Sampling", value=DOWN_SAMPLING)
    ini_file.set_value_at_ini(section="Common", key="Rebinning",     value=REBINNING)
    ini_file.set_value_at_ini(section="Common", key="N-Cores",       value=N_CORES)
    ini_file.set_value_at_ini(section="Common", key="N-Group",       value=N_GROUP)
    ini_file.set_value_at_ini(section="Common", key="Save-Images",   value=SAVE_IMAGES)
    ini_file.set_value_at_ini(section="Common", key="Verbose",       value=VERBOSE)

    ini_file.set_value_at_ini(section="WXST", key="Image-File-Name",     value=WXST_IMAGE_FILE_NAME)
    ini_file.set_value_at_ini(section="WXST", key="Reference-File-Name", value=WXST_REFERENCE_FILE_NAME)
    ini_file.set_value_at_ini(section="WXST", key="Dark-File-Name",      value=WXST_DARK_FILE_NAME)
    ini_file.set_value_at_ini(section="WXST", key="Flat-File-Name",      value=WXST_FLAT_FILE_NAME)
    ini_file.set_value_at_ini(section="WXST", key="Result-Folder",       value=WXST_RESULT_FOLDER)

    ini_file.set_value_at_ini(section="WSVT", key="Image-Folder",            value=WSVT_IMAGE_FOLDER)
    ini_file.set_value_at_ini(section="WSVT", key="Reference-Folder",        value=WSVT_REFERENCE_FOLDER)
    ini_file.set_value_at_ini(section="WSVT", key="Result-Folder",           value=WSVT_RESULT_FOLDER)
    ini_file.set_value_at_ini(section="WSVT", key="Calculation-Half-Window", value=WSVT_CALCULATION_HALF_WINDOW)
    ini_file.set_value_at_ini(section="WSVT", key="N-Scan",                  value=WSVT_N_SCAN)

    ini_file.push()

store()

class WaveletsAnalyzer(IWaveletsAnalyzer):
    def __init__(self):
        pass

    def get_current_setup(self) -> dict:
        return {}

    def process_image_WXST(self, mode=ProcessingMode.LIVE, n_threads=MAX_THREADS, **kwargs) -> dict:
        if mode == ProcessingMode.LIVE:
            return _process_image_WXST(**kwargs)
        else:
            raise NotImplementedError("Batch mode not available, yet")

    def process_images_WSVT(self, mode=ProcessingMode.LIVE, n_threads=MAX_THREADS, **kwargs) -> dict:
        if mode == ProcessingMode.LIVE:
            return _process_images_WSVT(**kwargs)
        else:
            raise NotImplementedError("Batch mode not available, yet")

def _process_image_WXST(**kwargs):
    arguments = {}
    arguments["img"]              = WXST_IMAGE_FILE_NAME
    arguments["ref"]              = WXST_REFERENCE_FILE_NAME
    arguments["dark"]             = WXST_DARK_FILE_NAME
    arguments["flat"]             = WXST_FLAT_FILE_NAME
    arguments["result_folder"]    = WXST_RESULT_FOLDER
    arguments["crop"]             = CROP
    arguments["p_x"]              = PIXEL_SIZE
    arguments["scaling_x"]        = SCALING_H
    arguments["scaling_y"]        = SCALING_V
    arguments["energy"]           = ENERGY
    arguments["distance"]         = DISTANCE
    arguments["down_sampling"]    = DOWN_SAMPLING
    arguments["GPU"]              = USE_GPU
    arguments["use_wavelet"]      = USE_WAVELET
    arguments["wavelet_lv_cut"]   = WAVELET_CUT
    arguments["pyramid_level"]    = PYRAMID_LEVEL
    arguments["n_iter"]           = N_ITERATIONS
    arguments["template_size"]    = TEMPLATE_SIZE
    arguments["window_searching"] = WINDOW_SEARCH
    arguments["nCores"]           = N_CORES
    arguments["nGroup"]           = N_GROUP
    arguments["verbose"]          = VERBOSE
    arguments["save_images"]      = SAVE_IMAGES

    return execute_process_image(**(arguments | kwargs))

def _process_images_WSVT(**kwargs):
    arguments = {}
    arguments["crop"]            = CROP
    arguments["folder_img"]      = WSVT_IMAGE_FOLDER
    arguments["folder_ref"]      = WSVT_REFERENCE_FOLDER
    arguments["folder_result"]   = WSVT_RESULT_FOLDER
    arguments["cal_half_window"] = WSVT_CALCULATION_HALF_WINDOW
    arguments["n_cores"]         = N_CORES
    arguments["n_group"]         = N_GROUP
    arguments["energy"]          = ENERGY
    arguments["pixel_size"]      = PIXEL_SIZE
    arguments["distance"]        = DISTANCE
    arguments["use_wavelet"]     = USE_WAVELET
    arguments["wavelet_ct"]      = WAVELET_CUT
    arguments["pyramid_level"]   = PYRAMID_LEVEL
    arguments["n_iteration"]     = N_ITERATIONS
    arguments["n_scan"]          = WSVT_N_SCAN
    arguments["use_GPU"]         = USE_GPU
    arguments["scaling_x"]       = SCALING_H
    arguments["scaling_y"]       = SCALING_V
    arguments["verbose"]         = VERBOSE
    arguments["save_images"]     = SAVE_IMAGES

    return execute_process_images(**(arguments | kwargs))
