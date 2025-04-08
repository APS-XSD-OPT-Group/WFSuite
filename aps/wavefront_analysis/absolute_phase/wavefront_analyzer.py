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
import glob
import random
import time
import pathlib
from threading import Thread

import numpy as np
import json

from aps.wavefront_analysis.absolute_phase.legacy.process_images_executor import execute_process_image
from aps.wavefront_analysis.absolute_phase.legacy.back_propagation_executor import execute_back_propagation

from aps.wavefront_analysis.absolute_phase.facade import IWavefrontAnalyzer, ProcessingMode, MAX_THREADS
from aps.wavefront_analysis.driver.wavefront_sensor import get_default_file_name_prefix, get_image_data, \
    PIXEL_SIZE, DETECTOR_RESOLUTION, INDEX_DIGITS

from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance
from aps.common.plot.image import apply_transformations

APPLICATION_NAME = "WAVEFRONT-ANALYSIS"

register_ini_instance(IniMode.LOCAL_JSON_FILE,
                      ini_file_name=".wavefront_analysis.json",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)

PATTERN_SIZE          = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Size",         default=4.942e-6)
PATTERN_THICKNESS     = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Thickness",    default=1.5e-6)
PATTERN_TRANSMISSION  = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Transmission", default=0.613)
RAN_MASK              = ini_file.get_string_from_ini( section="Mask", key="Pattern-Image",        default='RanMask5umB0.npy')
PROPAGATION_DISTANCE  = ini_file.get_float_from_ini(  section="Mask", key="Propagation-Distance", default=500e-3)

ENERGY                = ini_file.get_float_from_ini(  section="Source", key="Energy",            default=12398.0)
SOURCE_V              = ini_file.get_float_from_ini(  section="Source", key="Source-Size-V",     default=6.925e-6)
SOURCE_H              = ini_file.get_float_from_ini(  section="Source", key="Source-Size-H",     default=0.333e-6)
SOURCE_DISTANCE_V     = ini_file.get_float_from_ini(  section="Source", key="Source-Distance-V", default=1.5)
SOURCE_DISTANCE_H     = ini_file.get_float_from_ini(  section="Source", key="Source-Distance-H", default=1.5)

D_SOURCE_RECAL        = ini_file.get_boolean_from_ini(section="Execution", key="Source-Distance-Recalculation", default=True)
FIND_TRANSFER_MATRIX  = ini_file.get_boolean_from_ini(section="Execution", key="Find-Transfer-Matrix",          default=True)
IMAGE_TRANSFER_MATRIX = ini_file.get_list_from_ini(   section="Execution", key="Image-Transfer-Matrix",         default=[0, 1, 0], type=int)
CROP                  = ini_file.get_list_from_ini(   section="Execution", key="Crop",                          default=[-1], type=int)
ESTIMATION_METHOD     = ini_file.get_string_from_ini( section="Execution", key="Estimation-Method",             default='simple_speckle')
PROPAGATOR            = ini_file.get_string_from_ini( section="Execution", key="Propagator",                    default='RS')
IMAGE_OPS             = ini_file.get_dict_from_ini(   section="Execution", key="Image-Ops",                     default={"file" : [], "stream" :["T", "FH", "FV"]}, type=str)

CALIBRATION_PATH      = ini_file.get_string_from_ini( section="Reconstruction", key="Calibration-Path",  default=None)
MODE                  = ini_file.get_string_from_ini( section="Reconstruction", key="Mode",              default='centralLine')
LINE_WIDTH            = ini_file.get_int_from_ini(    section="Reconstruction", key="Line-Width",        default=10)
REBINNING             = ini_file.get_float_from_ini(  section="Reconstruction", key="Rebinning",         default=1.0)
DOWN_SAMPLING         = ini_file.get_float_from_ini(  section="Reconstruction", key="Down-Sampling",     default=1.0)
METHOD                = ini_file.get_string_from_ini( section="Reconstruction", key="Method",            default='WXST')
USE_GPU               = ini_file.get_boolean_from_ini(section="Reconstruction", key="Use-Gpu",           default=False)
USE_WAVELET           = ini_file.get_boolean_from_ini(section="Reconstruction", key="Use-Wavelet",       default=False)
WAVELET_CUT           = ini_file.get_int_from_ini(    section="Reconstruction", key="Wavelet-Cut",       default=2)
PYRAMID_LEVEL         = ini_file.get_int_from_ini(    section="Reconstruction", key="Pyramid-Level",     default=1)
N_ITERATIONS          = ini_file.get_int_from_ini(    section="Reconstruction", key="N-Iterations",      default=1)
TEMPLATE_SIZE         = ini_file.get_int_from_ini(    section="Reconstruction", key="Template-Size",     default=21)
WINDOW_SEARCH         = ini_file.get_int_from_ini(    section="Reconstruction", key="Window-Search",     default=20)
CROP_BOUNDARY         = ini_file.get_int_from_ini(    section="Reconstruction", key="Crop-Boundary",     default=-1)
N_CORES               = ini_file.get_int_from_ini(    section="Reconstruction", key="N-Cores",           default=16)
N_GROUP               = ini_file.get_int_from_ini(    section="Reconstruction", key="N-Group",           default=1)

KIND                    = ini_file.get_string_from_ini( section="Back-Propagation", key="Kind",                        default="1D")
REBINNING_BP            = ini_file.get_float_from_ini(  section="Back-Propagation", key="Rebinning",                   default=1.0)
SMOOTH_INTENSITY        = ini_file.get_boolean_from_ini(section="Back-Propagation", key="Smooth-Intensity",            default=False)
SIGMA_INTENSITY         = ini_file.get_int_from_ini(    section="Back-Propagation", key="Sigma-Intensity",             default=21)
SMOOTH_PHASE            = ini_file.get_boolean_from_ini(section="Back-Propagation", key="Smooth-Phase",                default=False)
SIGMA_PHASE             = ini_file.get_int_from_ini(    section="Back-Propagation", key="Sigma-Phase",                 default=21)
CROP_V                  = ini_file.get_int_from_ini(    section="Back-Propagation", key="Crop-V",                      default=500)
CROP_H                  = ini_file.get_int_from_ini(    section="Back-Propagation", key="Crop-H",                      default=500)
CROP_SHIFT_V            = ini_file.get_int_from_ini(    section="Back-Propagation", key="Crop-Shift-V",                default=0)
CROP_SHIFT_H            = ini_file.get_int_from_ini(    section="Back-Propagation", key="Crop-Shift-H",                default=0)
DISTANCE                = ini_file.get_float_from_ini(  section="Back-Propagation", key="2D, Propagation-Distance",    default=1.0)
DISTANCE_V              = ini_file.get_float_from_ini(  section="Back-Propagation", key="1D, Propagation-Distance-V",  default=1.0)
DISTANCE_H              = ini_file.get_float_from_ini(  section="Back-Propagation", key="1D, Propagation-Distance-H",  default=1.0)
DELTA_F_V               = ini_file.get_dict_from_ini(   section="Back-Propagation", key="Delta-F-V",                   default={"WXST" : 0.0, "SPINNet" : 0.0})
DELTA_F_H               = ini_file.get_dict_from_ini(   section="Back-Propagation", key="Delta-F-H",                   default={"WXST" : 0.0, "SPINNet" : 0.0})
RMS_RANGE_V             = ini_file.get_list_from_ini(   section="Back-Propagation", key="RMS-Range-V",                 default=[-2e-6, 2e-6], type=float)
RMS_RANGE_H             = ini_file.get_list_from_ini(   section="Back-Propagation", key="RMS-Range-H",                 default=[-2e-6, 2e-6], type=float)
MAGNIFICATION_V         = ini_file.get_float_from_ini(  section="Back-Propagation", key="Magnification-V",             default=0.028)
MAGNIFICATION_H         = ini_file.get_float_from_ini(  section="Back-Propagation", key="Magnification-H",             default=0.028)
SHIFT_HALF_PIXEL        = ini_file.get_boolean_from_ini(section="Back-Propagation", key="Shift-Half-Pixel",            default=False)
SCAN_BEST_FOCUS         = ini_file.get_boolean_from_ini(section="Back-Propagation", key="Scan-Best-Focus",             default=False)
USE_FIT                 = ini_file.get_boolean_from_ini(section="Back-Propagation", key="USe-Fit",                     default=True)
BEST_FOCUS_FROM         = ini_file.get_string_from_ini( section="Back-Propagation", key="Best-Focus-From",             default="rms")
BEST_FOCUS_SCAN_RANGE   = ini_file.get_list_from_ini(   section="Back-Propagation", key="2D, Best-Focus-Scan-Range",   default=[-0.001, 0.001, 0.0001], type=float)
BEST_FOCUS_SCAN_RANGE_V = ini_file.get_list_from_ini(   section="Back-Propagation", key="1D, Best-Focus-Scan-Range-V", default=[-0.001, 0.001, 0.0001], type=float)
BEST_FOCUS_SCAN_RANGE_H = ini_file.get_list_from_ini(   section="Back-Propagation", key="1D, Best-Focus-Scan-Range-H", default=[-0.001, 0.001, 0.0001], type=float)

SHOW_ALIGN_FIGURE     = ini_file.get_boolean_from_ini(section="Output", key="Show-Align-Figure",     default=False)
CORRECT_SCALE         = ini_file.get_boolean_from_ini(section="Output", key="Correct-Scale",         default=False)

def store():
    ini_file.set_value_at_ini(section="Mask", key="Pattern-Size",         value=PATTERN_SIZE)
    ini_file.set_value_at_ini(section="Mask", key="Pattern-Thickness",    value=PATTERN_THICKNESS)
    ini_file.set_value_at_ini(section="Mask", key="Pattern-Transmission", value=PATTERN_TRANSMISSION)
    ini_file.set_value_at_ini(section="Mask", key="Pattern-Image",        value=RAN_MASK)
    ini_file.set_value_at_ini(section="Mask", key="Propagation-Distance", value=PROPAGATION_DISTANCE)

    ini_file.set_value_at_ini(section="Source", key="Energy",               value=ENERGY)
    ini_file.set_value_at_ini(section="Source", key="Source-Size-V",        value=SOURCE_V)
    ini_file.set_value_at_ini(section="Source", key="Source-Size-H",        value=SOURCE_H)
    ini_file.set_value_at_ini(section="Source", key="Source-Distance-V",    value=SOURCE_DISTANCE_V)
    ini_file.set_value_at_ini(section="Source", key="Source-Distance-H",    value=SOURCE_DISTANCE_H)

    ini_file.set_value_at_ini(section="Execution", key="Source-Distance-Recalculation", value=D_SOURCE_RECAL)
    ini_file.set_value_at_ini(section="Execution", key="Find-Transfer-Matrix",          value=FIND_TRANSFER_MATRIX)
    ini_file.set_list_at_ini( section="Execution", key="Image-Transfer-Matrix",         values_list=IMAGE_TRANSFER_MATRIX)
    ini_file.set_list_at_ini( section="Execution", key="Crop",                          values_list=CROP)
    ini_file.set_value_at_ini(section="Execution", key="Estimation-Method",             value=ESTIMATION_METHOD)
    ini_file.set_value_at_ini(section="Execution", key="Propagator",                    value=PROPAGATOR)
    ini_file.set_dict_at_ini( section="Execution", key="Image-Ops",                     values_dict=IMAGE_OPS)

    ini_file.set_value_at_ini(section="Back-Propagation", key="Kind",                       value=KIND)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Rebinning",                  value=REBINNING_BP)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Smooth-Intensity",           value=SMOOTH_INTENSITY)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Sigma-Intensity",            value=SIGMA_INTENSITY )
    ini_file.set_value_at_ini(section="Back-Propagation", key="Smooth-Phase",               value=SMOOTH_PHASE)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Sigma-Phase",                value=SIGMA_PHASE)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Crop-H",                     value=CROP_H)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Crop-V",                     value=CROP_V)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Crop-Shift-H",               value=CROP_SHIFT_H)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Crop-Shift-V",               value=CROP_SHIFT_V)
    ini_file.set_value_at_ini(section="Back-Propagation", key="2D, Propagation-Distance",   value=DISTANCE)
    ini_file.set_value_at_ini(section="Back-Propagation", key="1D, Propagation-Distance-V", value=DISTANCE_V)
    ini_file.set_value_at_ini(section="Back-Propagation", key="1D, Propagation-Distance-H", value=DISTANCE_H)
    ini_file.set_dict_at_ini( section="Back-Propagation", key="Delta-F-V",                  values_dict=DELTA_F_V)
    ini_file.set_dict_at_ini( section="Back-Propagation", key="Delta-F-H",                  values_dict=DELTA_F_H)
    ini_file.set_list_at_ini( section="Back-Propagation", key="RMS-Range-V",                values_list=RMS_RANGE_V)
    ini_file.set_list_at_ini( section="Back-Propagation", key="RMS-Range-H",                values_list=RMS_RANGE_H)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Scan-Best-Focus",            value=SCAN_BEST_FOCUS)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Use-Fit",                    value=USE_FIT)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Best-Focus-From",            value=BEST_FOCUS_FROM)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Magnification-V",            value=MAGNIFICATION_V)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Magnification-H",            value=MAGNIFICATION_H)
    ini_file.set_list_at_ini( section="Back-Propagation", key="2D, Best-Focus-Scan-Range",      values_list=BEST_FOCUS_SCAN_RANGE)
    ini_file.set_list_at_ini( section="Back-Propagation", key="1D, Best-Focus-Scan-Range-V",    values_list=BEST_FOCUS_SCAN_RANGE_V)
    ini_file.set_list_at_ini( section="Back-Propagation", key="1D, Best-Focus-Scan-Range-H",    values_list=BEST_FOCUS_SCAN_RANGE_H)

    ini_file.set_value_at_ini(section="Reconstruction", key="Mode",           value=MODE)
    ini_file.set_value_at_ini(section="Reconstruction", key="Line-Width",     value=LINE_WIDTH)
    ini_file.set_value_at_ini(section="Reconstruction", key="Rebinning",      value=REBINNING)
    ini_file.set_value_at_ini(section="Reconstruction", key="Down-Sampling",  value=DOWN_SAMPLING)
    ini_file.set_value_at_ini(section="Reconstruction", key="Method",         value=METHOD)
    ini_file.set_value_at_ini(section="Reconstruction", key="Use-Gpu",        value=USE_GPU)
    ini_file.set_value_at_ini(section="Reconstruction", key="Use-Wavelet",    value=USE_WAVELET)
    ini_file.set_value_at_ini(section="Reconstruction", key="Wavelet-Cut",    value=WAVELET_CUT)
    ini_file.set_value_at_ini(section="Reconstruction", key="Pyramid-Level",  value=PYRAMID_LEVEL)
    ini_file.set_value_at_ini(section="Reconstruction", key="N-Iterations",   value=N_ITERATIONS)
    ini_file.set_value_at_ini(section="Reconstruction", key="Template-Size",  value=TEMPLATE_SIZE)
    ini_file.set_value_at_ini(section="Reconstruction", key="Window-Search",  value=WINDOW_SEARCH)
    ini_file.set_value_at_ini(section="Reconstruction", key="Crop-Boundary",  value=CROP_BOUNDARY)
    ini_file.set_value_at_ini(section="Reconstruction", key="N-Cores",        value=N_CORES)
    ini_file.set_value_at_ini(section="Reconstruction", key="N-Group",        value=N_GROUP)

    ini_file.set_value_at_ini(section="Output", key="Show-Align-Figure",     value=SHOW_ALIGN_FIGURE)
    ini_file.set_value_at_ini(section="Output", key="Correct-Scale",         value=CORRECT_SCALE)

    ini_file.push()

store()

class WavefrontAnalyzer(IWavefrontAnalyzer):
    def __init__(self,
                 data_collection_directory,
                 file_name_prefix=None,
                 simulated_mask_directory=None,
                 energy=ENERGY):
        self.__data_collection_directory = data_collection_directory
        self.__file_name_prefix          = file_name_prefix if not file_name_prefix is None else get_default_file_name_prefix()
        self.__simulated_mask_directory  = simulated_mask_directory
        self.__energy                    = energy

    def get_current_setup(self) -> dict:
        return {
            "data_collection_directory" : self.__data_collection_directory,
            "file_name_prefix" : self.__file_name_prefix,
            "simulated_mask_directory" : self.__simulated_mask_directory,
            "energy" : self.__energy
        }

    def generate_simulated_mask(self, image_index_for_mask: int = 1, data_collection_directory: str = None, **kwargs) -> [list, bool]:
        image_transfer_matrix, is_new_mask = _generate_simulated_mask(data_collection_directory=self.__data_collection_directory if data_collection_directory is None else data_collection_directory,
                                                                      file_name_prefix=self.__file_name_prefix,
                                                                      mask_directory=self.__simulated_mask_directory,
                                                                      energy=self.__energy,
                                                                      image_index=image_index_for_mask,
                                                                      **kwargs)
        return image_transfer_matrix, is_new_mask

    def get_wavefront_data(self, image_index: int, data_collection_directory: str = None, **kwargs) -> [np.ndarray, np.ndarray, np.ndarray]:
        image_ops = kwargs.get("image_ops", [])

        image, hh, vv = get_image_data(measurement_directory=self.__data_collection_directory if data_collection_directory is None else data_collection_directory,
                                       file_name_prefix=self.__file_name_prefix,
                                       image_index=image_index,
                                       **(kwargs))
        hh, vv, image = apply_transformations(hh, vv, image, image_ops)

        return hh, vv, image


    def process_image(self, image_index: int, data_collection_directory: str = None, **kwargs):
        return _process_image(data_collection_directory=self.__data_collection_directory if data_collection_directory is None else data_collection_directory,
                              file_name_prefix=self.__file_name_prefix,
                              mask_directory=self.__simulated_mask_directory,
                              energy=self.__energy,
                              image_index=image_index,
                              **kwargs)

    def process_images(self, data_collection_directory: str = None, mode=ProcessingMode.LIVE, n_threads=MAX_THREADS, **kwargs):
        data_collection_directory = self.__data_collection_directory if data_collection_directory is None else data_collection_directory

        if mode == ProcessingMode.LIVE:
            for file in os.listdir(data_collection_directory):
                if pathlib.Path(file).suffix == ".tif" and self.__file_name_prefix in file:
                    self.process_image(image_index=int(file.split('.tif')[0][-INDEX_DIGITS:]), verbose=kwargs.get("verbose", False))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)

            self.__active_threads = [None] * n_threads

            for i in range(n_threads):
                self.__active_threads[i] = ProcessingThread(thread_id=i+1,
                                                            data_collection_directory=data_collection_directory,
                                                            file_name_prefix=self.__file_name_prefix,
                                                            simulated_mask_directory=self.__simulated_mask_directory,
                                                            energy=self.__energy,
                                                            **kwargs)
                self.__active_threads[i].start()

    def wait_image_processing_to_end(self, **kwargs):
        active = True
        time.sleep(1)
        n_threads = len(self.__active_threads)
        status = np.full(n_threads, False)

        while(active):
            for i in range(n_threads): status[i] = self.__active_threads[i].is_alive()
            active = np.any(status, where=status==True)

            if active: time.sleep(1)

    def back_propagate_wavefront(self, image_index: int, data_collection_directory: str = None, **kwargs) -> dict:
        return _backpropagate_wavefront(data_collection_directory=self.__data_collection_directory if data_collection_directory is None else data_collection_directory,
                                        file_name_prefix=self.__file_name_prefix,
                                        mask_directory=self.__simulated_mask_directory,
                                        image_index=image_index,
                                        **kwargs)

from aps.common.singleton import synchronized_method

class ProcessingThread(Thread):
    def __init__(self, thread_id, 
                 data_collection_directory, 
                 file_name_prefix, 
                 simulated_mask_directory, 
                 energy, 
                 **kwargs):
        super(ProcessingThread, self).__init__(name="Thread #" + str(thread_id))
        self.__thread_id = thread_id
        self.__data_collection_directory = data_collection_directory
        self.__file_name_prefix          = file_name_prefix
        self.__simulated_mask_directory  = simulated_mask_directory
        self.__energy                    = energy
        self.__kwargs                    = kwargs

    @synchronized_method
    def check_new_data(self, images_list):
        image_indexes = []
        result_folder_list = glob.glob(os.path.join(os.path.dirname(images_list[0]), '*'))
        result_folder_list = [os.path.basename(f) for f in result_folder_list]

        for image in images_list:
            image_directory = os.path.basename(image).split('.tif')[0]
            if image_directory in result_folder_list:
                continue
            else:
                image_indexes.append(int(image_directory[-INDEX_DIGITS:]))
        return image_indexes

    def run(self):
        max_waiting_cycles = 60
        waiting_cycles     = 0

        while waiting_cycles < max_waiting_cycles:
            images_list = glob.glob(os.path.join(self.__data_collection_directory, self.__file_name_prefix + '_*.tif'), recursive=False)
            if len(images_list) == 0:
                waiting_cycles += 1
                print('Thread #' + str(self.__thread_id) + ' waiting for 1s for new data....')
            else:
                image_indexes = self.check_new_data(images_list)

                if len(image_indexes) == 0:
                    waiting_cycles += 1
                    print('Thread #' + str(self.__thread_id) + ' waiting for 1s for new data....')
                else:
                    random.shuffle(image_indexes)
                    if len(image_indexes) < 5: n = 1
                    else:                      n = 5

                    for image_index in image_indexes[0:n]: _process_image(self.__data_collection_directory,
                                                                          self.__file_name_prefix,
                                                                          self.__simulated_mask_directory,
                                                                          self.__energy,
                                                                          image_index,
                                                                          **self.__kwargs)
            time.sleep(1)

        print('Thread #' + str(self.__thread_id) + ' completed')

def _process_image(data_collection_directory, file_name_prefix, mask_directory, energy, image_index, **kwargs):
    index_digits = kwargs.get("index_digits", INDEX_DIGITS)
    verbose      = kwargs.get("verbose", False)
    img          = os.path.join(data_collection_directory, file_name_prefix + f"_%0{index_digits}i.tif" % image_index)
    image_data   = kwargs.get("image_data", None)
    image_ops    = kwargs.get("image_ops", IMAGE_OPS.get("file", []) if image_data is None else IMAGE_OPS.get("stream", []))

    dark           = None
    flat           = None
    mask_directory = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    result_folder  = os.path.join(os.path.dirname(img), os.path.basename(img).split('.tif')[0])

    # pattern simulation parameters
    pattern_path          = os.path.join(os.path.dirname(__import__("aps.wavefront_analysis.absolute_phase.legacy", fromlist=[""]).__file__), 'mask', RAN_MASK)
    propagated_pattern    = os.path.join(mask_directory, 'propagated_pattern.npz')
    propagated_patternDet = os.path.join(mask_directory, 'propagated_patternDet.npz')
    saving_path           = mask_directory

    return execute_process_image(img=img,
                                 image_data=image_data,
                                 image_ops=image_ops,
                                 dark=dark,
                                 flat=flat,
                                 result_folder=result_folder,
                                 pattern_path=pattern_path,
                                 propagated_pattern=propagated_pattern,
                                 propagated_patternDet=propagated_patternDet,
                                 saving_path=saving_path,
                                 crop=kwargs.get("crop", CROP),
                                 img_transfer_matrix=kwargs.get("image_transfer_matrix", IMAGE_TRANSFER_MATRIX),
                                 find_transferMatrix=False, # always false for just processing images
                                 p_x=kwargs.get("pixel_size", PIXEL_SIZE),
                                 det_res=kwargs.get("detector_resolution",DETECTOR_RESOLUTION),
                                 energy=energy,
                                 pattern_size=kwargs.get("pattern_size", PATTERN_SIZE),
                                 pattern_thickness=kwargs.get("pattern_thickness", PATTERN_THICKNESS),
                                 pattern_T=kwargs.get("pattern_transmission", PATTERN_TRANSMISSION),
                                 d_prop=kwargs.get("propagation_distance", PROPAGATION_DISTANCE),
                                 d_source_v=kwargs.get("source_distance_v", SOURCE_DISTANCE_V),
                                 d_source_h=kwargs.get("source_distance_h", SOURCE_DISTANCE_H),
                                 source_v=kwargs.get("source_size_v", SOURCE_V),
                                 source_h=kwargs.get("source_size_h", SOURCE_H),
                                 correct_scale=kwargs.get("correct_scale", CORRECT_SCALE),
                                 show_alignFigure=kwargs.get("show_align_figure", SHOW_ALIGN_FIGURE),
                                 d_source_recal=False,  # for mask generation only,
                                 propagator=kwargs.get("propagator", PROPAGATOR),
                                 cali_path=kwargs.get("calibration_path", CALIBRATION_PATH),
                                 mode=kwargs.get("mode", MODE),
                                 lineWidth=kwargs.get("line_width", LINE_WIDTH),
                                 rebinning=kwargs.get("rebinning", REBINNING),
                                 down_sampling=kwargs.get("down_sampling", DOWN_SAMPLING),
                                 crop_boundary=kwargs.get("crop_boundary", CROP_BOUNDARY),
                                 method=kwargs.get("method", METHOD),
                                 GPU=kwargs.get("use_gpu", USE_GPU),
                                 use_wavelet=kwargs.get("use_wavelet", USE_WAVELET),
                                 wavelet_lv_cut=kwargs.get("wavelet_lv_cut", WAVELET_CUT),
                                 n_iter=kwargs.get("n_iterations", N_ITERATIONS),
                                 pyramid_level=kwargs.get("pyramid_level", PYRAMID_LEVEL),
                                 template_size=kwargs.get("template_size", TEMPLATE_SIZE),
                                 window_searching=kwargs.get("window_search", WINDOW_SEARCH),
                                 nCores=kwargs.get("n_cores", N_CORES),
                                 nGroup=kwargs.get("n_group", N_GROUP),
                                 verbose=verbose)
    print("Image " + file_name_prefix + "_%05i.tif" % image_index + " processed")

def _generate_simulated_mask(data_collection_directory, file_name_prefix, mask_directory, energy, image_index=1, **kwargs) -> [list, bool]:
    index_digits = kwargs.get("index_digits", INDEX_DIGITS)
    verbose      = kwargs.get("verbose", False)

    dark = None
    flat = None
    img         = os.path.join(data_collection_directory, file_name_prefix + f"_%0{index_digits}i.tif" % image_index)
    image_data  = kwargs.get("image_data", None)
    image_ops   = kwargs.get("image_ops", IMAGE_OPS.get("file", []) if image_data is None else IMAGE_OPS.get("stream", []))

    mask_directory  = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    result_folder   = os.path.join(os.path.dirname(img), os.path.basename(img).split('.tif')[0])

    pattern_path    = os.path.join(os.path.dirname(__import__("aps.wavefront_analysis.absolute_phase.legacy", fromlist=[""]).__file__), 'mask', RAN_MASK)
    saving_path     = mask_directory

    if not os.path.exists(mask_directory): os.mkdir(mask_directory)

    if not os.path.exists(os.path.join(mask_directory, 'propagated_pattern.npz')) or \
       not os.path.exists(os.path.join(mask_directory, 'propagated_patternDet.npz')) or \
       not os.path.exists(os.path.join(mask_directory, "reference.json")):
        execute_process_image(img=img,
                              image_data=image_data,
                              image_ops=image_ops,
                              dark=dark,
                              flat=flat,
                              result_folder=result_folder,
                              pattern_path=pattern_path,
                              propagated_pattern=None,
                              propagated_patternDet=None,
                              saving_path=saving_path,
                              crop=kwargs.get("crop", CROP),
                              img_transfer_matrix=None,
                              find_transferMatrix=FIND_TRANSFER_MATRIX,
                              p_x=kwargs.get("pixel_size", PIXEL_SIZE),
                              det_res=kwargs.get("detector_resolution", DETECTOR_RESOLUTION),
                              energy=energy,
                              pattern_size=kwargs.get("pattern_size", PATTERN_SIZE),
                              pattern_thickness=kwargs.get("pattern_thickness", PATTERN_THICKNESS),
                              pattern_T=kwargs.get("pattern_transmission", PATTERN_TRANSMISSION),
                              d_prop=kwargs.get("propagation_distance", PROPAGATION_DISTANCE),
                              d_source_v=kwargs.get("source_distance_v", SOURCE_DISTANCE_V),
                              d_source_h=kwargs.get("source_distance_h", SOURCE_DISTANCE_H),
                              source_v=kwargs.get("source_size_v", SOURCE_V),
                              source_h=kwargs.get("source_size_h", SOURCE_H),
                              correct_scale=kwargs.get("correct_scale", CORRECT_SCALE),
                              show_alignFigure=kwargs.get("show_align_figure", SHOW_ALIGN_FIGURE),
                              d_source_recal=kwargs.get("source_distance_recalculation", D_SOURCE_RECAL),  # for mask generation only,
                              propagator=kwargs.get("propagator", PROPAGATOR),
                              cali_path=kwargs.get("calibration_path", CALIBRATION_PATH),
                              mode=kwargs.get("mode", MODE),
                              lineWidth=kwargs.get("line_width", LINE_WIDTH),
                              rebinning=kwargs.get("rebinning", REBINNING),
                              down_sampling=kwargs.get("down_sampling", DOWN_SAMPLING),
                              crop_boundary=kwargs.get("crop_boundary", CROP_BOUNDARY),
                              method=kwargs.get("method", METHOD),
                              GPU=kwargs.get("use_gpu", USE_GPU),
                              use_wavelet=kwargs.get("use_wavelet", USE_WAVELET),
                              wavelet_lv_cut=kwargs.get("wavelet_lv_cut", WAVELET_CUT),
                              n_iter=kwargs.get("n_iterations", N_ITERATIONS),
                              pyramid_level=kwargs.get("pyramid_level", PYRAMID_LEVEL),
                              template_size=kwargs.get("template_size", TEMPLATE_SIZE),
                              window_searching=kwargs.get("window_search", WINDOW_SEARCH),
                              nCores=kwargs.get("n_cores", N_CORES),
                              nGroup=kwargs.get("n_group", N_GROUP),
                              verbose=verbose)
        is_new_mask = True
        print("Simulated mask generated in " + mask_directory)
    else:
        is_new_mask = False
        if verbose: print("Simulated mask already generated in " + mask_directory)

    with open(os.path.join(mask_directory, "reference.json"), 'r') as file: parameters = json.load(file)

    return parameters["image_transfer_matrix"], is_new_mask

def _backpropagate_wavefront(data_collection_directory, file_name_prefix, mask_directory, image_index, **kwargs) -> dict:
    index_digits = kwargs.get("index_digits", INDEX_DIGITS)

    folder = os.path.join(data_collection_directory, (file_name_prefix + "_%0" + str(index_digits) + "i") % image_index)
    mask_directory = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory

    return execute_back_propagation(folder=folder,
                                    reference_folder=mask_directory,
                                    kind                   = kwargs.get("kind", KIND),
                                    mask_detector_distance = kwargs.get("mask_detector_distance", PROPAGATION_DISTANCE),
                                    pixel_size             = kwargs.get("pixel_size", PIXEL_SIZE),
                                    image_rebinning        = kwargs.get("image_rebinning", REBINNING),
                                    distance               = kwargs.get("propagation_distance", DISTANCE),
                                    distance_x             = kwargs.get("propagation_distance_h", DISTANCE_H),
                                    distance_y             = kwargs.get("propagation_distance_v", DISTANCE_V),
                                    rebinning              = kwargs.get("rebinning", REBINNING_BP),
                                    smooth_intensity       = kwargs.get("smooth_intensity", SMOOTH_INTENSITY),
                                    smooth_phase           = kwargs.get("smooth_phase", SMOOTH_PHASE),
                                    sigma_intensity        = kwargs.get("sigma_intensity", SIGMA_INTENSITY),
                                    sigma_phase            = kwargs.get("sigma_phase", SIGMA_PHASE),
                                    dim_x                  = kwargs.get("crop_h", CROP_H),
                                    dim_y                  = kwargs.get("crop_v", CROP_V),
                                    shift_x                = kwargs.get("crop_shift_h", CROP_SHIFT_H),
                                    shift_y                = kwargs.get("crop_shift_v", CROP_SHIFT_V),
                                    delta_f_x              = kwargs.get("delta_f_h", DELTA_F_H.get(METHOD, 0.0)),
                                    delta_f_y              = kwargs.get("delta_f_v", DELTA_F_V.get(METHOD, 0.0)),
                                    x_rms_range            = kwargs.get("rms_range_h", RMS_RANGE_H),
                                    y_rms_range            = kwargs.get("rms_range_v", RMS_RANGE_V),
                                    magnification_x        = kwargs.get("magnification_h", MAGNIFICATION_H),
                                    magnification_y        = kwargs.get("magnification_v", MAGNIFICATION_V),
                                    shift_half_pixel       = kwargs.get("shift_half_pixel", SHIFT_HALF_PIXEL),
                                    show_figure            = kwargs.get("show_figure", False),
                                    save_result            = kwargs.get("save_result", False),
                                    scan_best_focus        = kwargs.get("scan_best_focus", SCAN_BEST_FOCUS),
                                    use_fit                = kwargs.get("use_fit", USE_FIT),
                                    best_focus_from        = kwargs.get("best_focus_from", BEST_FOCUS_FROM),
                                    scan_rel_range         = kwargs.get("best_focus_scan_range", BEST_FOCUS_SCAN_RANGE),
                                    scan_x_rel_range       = kwargs.get("best_focus_scan_range_h", BEST_FOCUS_SCAN_RANGE_H),
                                    scan_y_rel_range       = kwargs.get("best_focus_scan_range_v", BEST_FOCUS_SCAN_RANGE_V),
                                    verbose                = kwargs.get("verbose", False))