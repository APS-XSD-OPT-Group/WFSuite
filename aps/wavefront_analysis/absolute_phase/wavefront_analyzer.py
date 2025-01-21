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
from PIL import Image

from aps.wavefront_analysis.absolute_phase.legacy.executor import execute_process_image
from aps.wavefront_analysis.absolute_phase.facade import IWavefrontAnalyzer, ProcessingMode, MAX_THREADS
from aps.wavefront_analysis.driver.wavefront_sensor import get_default_file_name_prefix

from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance

APPLICATION_NAME = "WAVEFRONT-ANALYSIS"

register_ini_instance(IniMode.LOCAL_JSON_FILE,
                      ini_file_name=".wavefront_analysis.json",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)

PIXEL_SIZE            = ini_file.get_float_from_ini(  section="Detector", key="Pixel-Size", default=0.65e-6)
IMAGE_SIZE_PIXEL_V    = ini_file.get_float_from_ini(  section="Detector", key="Image-Size-V", default=2560)
IMAGE_SIZE_PIXEL_H    = ini_file.get_float_from_ini(  section="Detector", key="Image-Size-H", default=2160)
DETECTOR_RESOLUTION   = ini_file.get_float_from_ini(  section="Detector", key="Resolution", default=1.5e-6)

PATTERN_SIZE          = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Size",         default=4.942e-6)
PATTERN_THICKNESS     = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Thickness",    default=1.5e-6)
PATTERN_T             = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Transmission", default=0.613)
RAN_MASK              = ini_file.get_string_from_ini( section="Mask", key="Pattern-Image",        default='RanMask5umB0.npy')
PROPAGATION_DISTANCE  = ini_file.get_float_from_ini(section="Mask", key="Propagation-Distance", default=500e-3)

SOURCE_V              = ini_file.get_float_from_ini(  section="Source", key="Source-Size-V",     default=6.925e-6)
SOURCE_H              = ini_file.get_float_from_ini(  section="Source", key="Source-Size-H",     default=0.333e-6)
SOURCE_DISTANCE_V     = ini_file.get_float_from_ini(  section="Source", key="Source-Distance-V", default=1.5)
SOURCE_DISTANCE_H     = ini_file.get_float_from_ini(  section="Source", key="Source-Distance-H", default=1.5)

D_SOURCE_RECAL        = ini_file.get_boolean_from_ini(section="Execution", key="Source-Distance-Recalculation", default=True)
CROP                  = ini_file.get_list_from_ini(   section="Execution", key="Crop",                          default=[-1], type=int)
ESTIMATION_METHOD     = ini_file.get_string_from_ini( section="Execution", key="Estimation-Method",             default='simple_speckle')
PROPAGATOR            = ini_file.get_string_from_ini( section="Execution", key="Propagator",                    default='RS')

CALIBRATION_PATH      = ini_file.get_string_from_ini( section="Reconstruction", key="Calibration-Path",  default=None)
MODE                  = ini_file.get_string_from_ini( section="Reconstruction", key="Mode",              default='centralLine')
LINE_WIDTH            = ini_file.get_int_from_ini(    section="Reconstruction", key="Line-Width",        default=10)
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

IMAGE_TRANSFER_MATRIX = ini_file.get_list_from_ini(   section="Output", key="Image-Transfer-Matrix", default=[0, 1, 0], type=int)
SHOW_ALIGN_FIGURE     = ini_file.get_boolean_from_ini(section="Output", key="Show-Align-Figure",     default=False)
CORRECT_SCALE         = ini_file.get_boolean_from_ini(section="Output", key="Correct-Scale",         default=False)

ini_file.set_value_at_ini(section="Detector", key="Pixel-Size", value=PIXEL_SIZE)
ini_file.set_value_at_ini( section="Detector", key="Image-Size-V", value=IMAGE_SIZE_PIXEL_V)
ini_file.set_value_at_ini( section="Detector", key="Image-Size-H", value=IMAGE_SIZE_PIXEL_H)

ini_file.set_value_at_ini(section="Mask", key="Pattern-Size",         value=PATTERN_SIZE)
ini_file.set_value_at_ini(section="Mask", key="Pattern-Thickness",    value=PATTERN_THICKNESS)
ini_file.set_value_at_ini(section="Mask", key="Pattern-Transmission", value=PATTERN_T)
ini_file.set_value_at_ini(section="Mask", key="Pattern-Image",        value=RAN_MASK)
ini_file.set_value_at_ini(section="Mask", key="Propagation-Distance", value=PROPAGATION_DISTANCE)

ini_file.set_value_at_ini(section="Source", key="Source-Size-V",        value=SOURCE_V)
ini_file.set_value_at_ini(section="Source", key="Source-Size-H",        value=SOURCE_H)
ini_file.set_value_at_ini(section="Source", key="Source-Distance-V",    value=SOURCE_DISTANCE_V)
ini_file.set_value_at_ini(section="Source", key="Source-Distance-H",    value=SOURCE_DISTANCE_H)

ini_file.set_value_at_ini(section="Execution", key="Source-Distance-Recalculation", value=D_SOURCE_RECAL)
ini_file.set_list_at_ini( section="Execution", key="Crop",                          values_list=CROP)
ini_file.set_value_at_ini(section="Execution", key="Estimation-Method",             value=ESTIMATION_METHOD)

ini_file.set_value_at_ini(section="Reconstruction", key="Mode",           value=MODE)
ini_file.set_value_at_ini(section="Reconstruction", key="Line-Width",     value=LINE_WIDTH   )
ini_file.set_value_at_ini(section="Reconstruction", key="Down-Sampling",  value=DOWN_SAMPLING)
ini_file.set_value_at_ini(section="Reconstruction", key="Method",         value=METHOD       )
ini_file.set_value_at_ini(section="Reconstruction", key="Use-Gpu",        value=USE_GPU      )
ini_file.set_value_at_ini(section="Reconstruction", key="Use-Wavelet",    value=USE_WAVELET  )
ini_file.set_value_at_ini(section="Reconstruction", key="Wavelet-Cut",    value=WAVELET_CUT  )
ini_file.set_value_at_ini(section="Reconstruction", key="Pyramid-Level",  value=PYRAMID_LEVEL)
ini_file.set_value_at_ini(section="Reconstruction", key="N-Iterations",   value=N_ITERATIONS)
ini_file.set_value_at_ini(section="Reconstruction", key="Template-Size",  value=TEMPLATE_SIZE)
ini_file.set_value_at_ini(section="Reconstruction", key="Window-Search",  value=WINDOW_SEARCH)
ini_file.set_value_at_ini(section="Reconstruction", key="Crop-Boundary",  value=CROP_BOUNDARY)
ini_file.set_value_at_ini(section="Reconstruction", key="N-Cores",        value=N_CORES      )
ini_file.set_value_at_ini(section="Reconstruction", key="N-Group",        value=N_GROUP      )

ini_file.set_list_at_ini( section="Output", key="Image-Transfer-Matrix", values_list=IMAGE_TRANSFER_MATRIX)
ini_file.set_value_at_ini(section="Output", key="Show-Align-Figure",     value=SHOW_ALIGN_FIGURE)

ini_file.push()

class WavefrontAnalyzer(IWavefrontAnalyzer):
    def __init__(self,
                 data_collection_directory,
                 file_name_prefix=get_default_file_name_prefix(),
                 simulated_mask_directory=None,
                 energy=20000.0):
        self.__data_collection_directory = data_collection_directory
        self.__file_name_prefix          = file_name_prefix
        self.__simulated_mask_directory  = simulated_mask_directory
        self.__energy                    = energy

    def generate_simulated_mask(self, image_index_for_mask: int = 1, data_collection_directory: str = None, verbose: bool = False, **kwargs) -> [list, bool]:
        image_transfer_matrix, is_new_mask = _generate_simulated_mask(data_collection_directory=self.__data_collection_directory if data_collection_directory is None else data_collection_directory,
                                                                      file_name_prefix=self.__file_name_prefix,
                                                                      mask_directory=self.__simulated_mask_directory,
                                                                      energy=self.__energy,
                                                                      image_index=image_index_for_mask,
                                                                      verbose=verbose)
        return image_transfer_matrix, is_new_mask

    def get_image_data(self, image_index: int, data_collection_directory: str = None, index_digits: int = 4, units="mm", **kwargs) -> [np.ndarray, np.ndarray, np.ndarray]:
        return _get_image_data(data_collection_directory=self.__data_collection_directory if data_collection_directory is None else data_collection_directory,
                               file_name_prefix=self.__file_name_prefix,
                               image_index=image_index,
                               index_digits=index_digits,
                               units=units)

    def process_image(self, image_index: int, data_collection_directory: str = None, verbose: bool = False, **kwargs):
        _process_image(data_collection_directory=self.__data_collection_directory if data_collection_directory is None else data_collection_directory,
                       file_name_prefix=self.__file_name_prefix,
                       mask_directory=self.__simulated_mask_directory,
                       energy=self.__energy,
                       image_index=image_index,
                       verbose=verbose)

    def process_images(self, data_collection_directory: str = None, mode=ProcessingMode.LIVE, n_threads=MAX_THREADS, verbose: bool = False, **kwargs):
        data_collection_directory = self.__data_collection_directory if data_collection_directory is None else data_collection_directory

        if mode == ProcessingMode.LIVE:
            for file in os.listdir(data_collection_directory):
                if pathlib.Path(file).suffix == ".tif" and self.__file_name_prefix in file:
                    self.process_image(image_index=int(file.split('.tif')[0][-5:]), verbose=verbose)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)

            self.__active_threads = [None] * n_threads

            for i in range(n_threads):
                self.__active_threads[i] = ProcessingThread(thread_id=i + 1,
                                                            data_collection_directory=data_collection_directory,
                                                            file_name_prefix=self.__file_name_prefix,
                                                            simulated_mask_directory=self.__simulated_mask_directory,
                                                            energy=self.__energy,
                                                            verbose=verbose)
                self.__active_threads[i].start()

    def wait_image_processing_to_end(self, verbose: bool = False, **kwargs):
        active = True
        time.sleep(1)
        n_threads = len(self.__active_threads)
        status = np.full(n_threads, False)

        while(active):
            for i in range(n_threads): status[i] = self.__active_threads[i].is_alive()
            active = np.any(status, where=status==True)

            if active: time.sleep(1)


class ProcessingThread(Thread):
    def __init__(self, thread_id, 
                 data_collection_directory, 
                 file_name_prefix, 
                 simulated_mask_directory, 
                 energy, 
                 verbose=False):
        super(ProcessingThread, self).__init__(name="Thread #" + str(thread_id))
        self.__thread_id = thread_id
        self.__data_collection_directory = data_collection_directory
        self.__file_name_prefix          = file_name_prefix
        self.__simulated_mask_directory  = simulated_mask_directory
        self.__energy                    = energy
        self.__verbose                   = verbose

    def run(self):
        def check_new_data(images_list):
            image_indexes      = []
            result_folder_list = glob.glob(os.path.join(os.path.dirname(images_list[0]), '*'))
            result_folder_list = [os.path.basename(f) for f in result_folder_list]

            for image in images_list:
                image_directory = os.path.basename(image).split('.tif')[0]
                if image_directory in result_folder_list: continue
                else: image_indexes.append(int(image_directory[-5:]))
            return image_indexes

        max_waiting_cycles = 60
        waiting_cycles     = 0

        while waiting_cycles < max_waiting_cycles:
            images_list   = glob.glob(os.path.join(self.__data_collection_directory, self.__file_name_prefix + '_*.tif'), recursive=False)
            if len(images_list) == 0:
                waiting_cycles += 1
                print('Thread #' + str(self.__thread_id) + ' waiting for 1s for new data....')
            else:
                image_indexes = check_new_data(images_list)

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
                                                                          self.__verbose)
            time.sleep(1)

        print('Thread #' + str(self.__thread_id) + ' completed')

def _get_image_data(data_collection_directory, file_name_prefix, image_index, index_digits=4, units="mm"):
    def load_image(file_path):
        if os.path.exists(file_path): return np.array(np.array(Image.open(file_path))).astype(np.float32).T
        else:                         raise ValueError('Error: wrong data path. No data is loaded:' + file_path)

    image = load_image(os.path.join(data_collection_directory, (file_name_prefix + "_%0" + str(index_digits) + "i.tif") % image_index))

    factor = 1e6 if units == "um" else (1e3 if units == "mm" else (1e2 if units == "cm" else 1.0))

    h_coord = np.linspace(-IMAGE_SIZE_PIXEL_H / 2, IMAGE_SIZE_PIXEL_H / 2, IMAGE_SIZE_PIXEL_H) * PIXEL_SIZE * factor
    v_coord = np.linspace(-IMAGE_SIZE_PIXEL_V / 2, IMAGE_SIZE_PIXEL_V / 2, IMAGE_SIZE_PIXEL_V) * PIXEL_SIZE * factor

    return image, h_coord, v_coord

def _process_image(data_collection_directory, file_name_prefix, mask_directory, energy, image_index, verbose=False):
    img            = os.path.join(data_collection_directory, file_name_prefix + "_%05i.tif" % image_index)
    dark           = None
    flat           = None
    mask_directory = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    result_folder  = os.path.join(os.path.dirname(img), os.path.basename(img).split('.tif')[0])

    # pattern simulation parameters
    pattern_path          = os.path.join(os.path.abspath(os.curdir), 'mask' , RAN_MASK)
    propagated_pattern    = os.path.join(mask_directory, 'propagated_pattern.npz')
    propagated_patternDet = os.path.join(mask_directory, 'propagated_patternDet.npz')
    saving_path           = mask_directory

    execute_process_image(img=img,
                          dark=dark,
                          flat=flat,
                          result_folder=result_folder,
                          pattern_path=pattern_path,
                          propagated_pattern=propagated_pattern,
                          propagated_patternDet=propagated_patternDet,
                          saving_path=saving_path,
                          crop=CROP,
                          img_transfer_matrix=IMAGE_TRANSFER_MATRIX,
                          find_transferMatrix=False,
                          det_size=[IMAGE_SIZE_PIXEL_V, IMAGE_SIZE_PIXEL_H],
                          p_x=PIXEL_SIZE,
                          det_res=DETECTOR_RESOLUTION,
                          energy=energy,
                          pattern_size=PATTERN_SIZE,
                          pattern_thickness=PATTERN_THICKNESS,
                          pattern_T=PATTERN_T,
                          d_prop=PROPAGATION_DISTANCE,
                          d_source_v=SOURCE_DISTANCE_V,
                          d_source_h=SOURCE_DISTANCE_H,
                          source_v=SOURCE_V,
                          source_h=SOURCE_H,
                          correct_scale=CORRECT_SCALE,
                          show_alignFigure=SHOW_ALIGN_FIGURE,
                          d_source_recal=False, # for mask generation only,
                          propagator=PROPAGATOR,
                          cali_path=CALIBRATION_PATH,
                          mode=MODE,
                          lineWidth=LINE_WIDTH,
                          down_sampling=DOWN_SAMPLING,
                          crop_boundary=CROP_BOUNDARY,
                          method=METHOD,
                          GPU=USE_GPU,
                          use_wavelet=USE_WAVELET,
                          wavelet_lv_cut=WAVELET_CUT,
                          n_iter=N_ITERATIONS,
                          pyramid_level=PYRAMID_LEVEL,
                          template_size=TEMPLATE_SIZE,
                          window_searching=WINDOW_SEARCH,
                          nCores=N_CORES,
                          nGroup=N_GROUP)
    print("Image " + file_name_prefix + "_%05i.tif" % image_index + " processed")

def _generate_simulated_mask(data_collection_directory, file_name_prefix, mask_directory, energy, image_index=1, verbose=False):
    dark = None
    flat = None
    img             = os.path.join(data_collection_directory, file_name_prefix + "_%05i.tif" % image_index)
    mask_directory  = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    result_folder   = os.path.join(os.path.dirname(img), os.path.basename(img).split('.tif')[0])
    pattern_path    = os.path.join(os.path.abspath(os.curdir), 'mask', RAN_MASK)
    saving_path     = mask_directory

    if not os.path.exists(mask_directory): os.mkdir(mask_directory)

    if not os.path.exists(os.path.join(mask_directory, 'propagated_pattern.npz')) or \
       not os.path.exists(os.path.join(mask_directory, 'propagated_patternDet.npz')) or \
       not os.path.exists(os.path.join(mask_directory, "image_transfer_matrix.npy")):
        execute_process_image(img=img,
                              dark=dark,
                              flat=flat,
                              result_folder=result_folder,
                              pattern_path=pattern_path,
                              propagated_pattern=None,
                              propagated_patternDet=None,
                              saving_path=saving_path,
                              crop=CROP,
                              img_transfer_matrix=IMAGE_TRANSFER_MATRIX,
                              find_transferMatrix=True,
                              det_size=[IMAGE_SIZE_PIXEL_V, IMAGE_SIZE_PIXEL_H],
                              p_x=PIXEL_SIZE,
                              det_res=DETECTOR_RESOLUTION,
                              energy=energy,
                              pattern_size=PATTERN_SIZE,
                              pattern_thickness=PATTERN_THICKNESS,
                              pattern_T=PATTERN_T,
                              d_prop=PROPAGATION_DISTANCE,
                              d_source_v=SOURCE_DISTANCE_V,
                              d_source_h=SOURCE_DISTANCE_H,
                              source_v=SOURCE_V,
                              source_h=SOURCE_H,
                              correct_scale=CORRECT_SCALE,
                              show_alignFigure=SHOW_ALIGN_FIGURE,
                              d_source_recal=D_SOURCE_RECAL,  # for mask generation only,
                              propagator=PROPAGATOR,
                              cali_path=CALIBRATION_PATH,
                              mode=MODE,
                              lineWidth=LINE_WIDTH,
                              down_sampling=DOWN_SAMPLING,
                              crop_boundary=CROP_BOUNDARY,
                              method=METHOD,
                              GPU=USE_GPU,
                              use_wavelet=USE_WAVELET,
                              wavelet_lv_cut=WAVELET_CUT,
                              n_iter=N_ITERATIONS,
                              pyramid_level=PYRAMID_LEVEL,
                              template_size=TEMPLATE_SIZE,
                              window_searching=WINDOW_SEARCH,
                              nCores=N_CORES,
                              nGroup=N_GROUP)
        is_new_mask = True
        print("Simulated mask generated in " + mask_directory)
    else:
        is_new_mask = False
        print("Simulated mask already generated in " + mask_directory)

    with open(os.path.join(mask_directory, "image_transfer_matrix.npy"), 'rb') as f: image_transfer_matrix = np.load(f, allow_pickle=False)

    return image_transfer_matrix.tolist(), is_new_mask
