#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2024-2026, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2024-2026. UChicago Argonne, LLC. This software was produced  #
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
from pathlib import Path
import numpy as np
import json

from aps.wf_suite.absolute_phase.legacy.process_images_executor import execute_process_image
from aps.wf_suite.absolute_phase.legacy.process_images_WSVT_executor import execute_process_images_WSVT
from aps.wf_suite.absolute_phase.legacy.back_propagation_executor import execute_back_propagation

from aps.wf_suite.absolute_phase.facade import IAbsolutePhaseAnalyzer, ProcessingMode, MAX_THREADS
import aps.wf_suite.driver.wavefront_sensor as ws

from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance

APPLICATION_NAME = "ABSOLUTE-PHASE-ANALYSIS"

register_ini_instance(IniMode.LOCAL_JSON_FILE,
                      ini_file_name=".absolute_phase_analysis.json",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)

data_directory = os.path.join(Path(os.path.dirname(__import__("aps.wf_suite", fromlist=[""]).__file__)).parents[1], "Data")

DATA_DIRECTORY        = ini_file.get_string_from_ini( section="General", key="Data-Directory",    default=data_directory)

PATTERN_SIZE          = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Size",         default=4.942e-6)
PATTERN_THICKNESS     = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Thickness",    default=1.5e-6)
PATTERN_TRANSMISSION  = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Transmission", default=0.613)
RAN_MASK              = ini_file.get_string_from_ini( section="Mask", key="Pattern-Image",        default='RanMask5umB0.npy')
PROPAGATION_DISTANCE  = ini_file.get_float_from_ini(  section="Mask", key="Propagation-Distance", default=500e-3)

# [MASK EXPOSURE MODEL] (ADDED) fabrication over/under-exposure of the mask.
# Defaults keep the model OFF, i.e. identical to previous behavior.
EXPOSURE_MODEL        = ini_file.get_string_from_ini( section="Mask", key="Exposure-Model",        default="off")             # "off", "ctr", or "distance"
EXPOSURE_AUTO         = ini_file.get_boolean_from_ini(section="Mask", key="Exposure-Auto",         default=True)              # True -> estimate bias from the image; False -> use Exposure-Bias
EXPOSURE_BIAS         = ini_file.get_float_from_ini(  section="Mask", key="Exposure-Bias",         default=0.0)               # manual edge shift (fraction of a feature) when Exposure-Auto is False
EXPOSURE_CORNER_SIGMA = ini_file.get_float_from_ini(  section="Mask", key="Exposure-Corner-Sigma", default=0.25)              # corner rounding radius (fraction of a feature)
EXPOSURE_SUPERSAMPLE  = ini_file.get_int_from_ini(    section="Mask", key="Exposure-Supersample",  default=8)                 # fine-grid up-sampling factor
EXPOSURE_ESTIMATE     = ini_file.get_string_from_ini( section="Mask", key="Exposure-Estimate",     default="match_quality")   # "match_quality" or "occupation"
EXPOSURE_BIAS_GRID    = ini_file.get_list_from_ini(   section="Mask", key="Exposure-Bias-Grid",    default=[-0.25, -0.125, 0.0, 0.125, 0.25], _type=float)  # auto-search candidates

ENERGY                = ini_file.get_float_from_ini(  section="Source", key="Energy",            default=12398.0)
SOURCE_V              = ini_file.get_float_from_ini(  section="Source", key="Source-Size-V",     default=6.925e-6)
SOURCE_H              = ini_file.get_float_from_ini(  section="Source", key="Source-Size-H",     default=0.333e-6)
SOURCE_DISTANCE_V     = ini_file.get_float_from_ini(  section="Source", key="Source-Distance-V", default=1.5)
SOURCE_DISTANCE_H     = ini_file.get_float_from_ini(  section="Source", key="Source-Distance-H", default=1.5)

D_SOURCE_RECAL        = ini_file.get_boolean_from_ini(section="Execution", key="Source-Distance-Recalculation", default=True)
FIND_TRANSFER_MATRIX  = ini_file.get_boolean_from_ini(section="Execution", key="Find-Transfer-Matrix",          default=True)
IMAGE_TRANSFER_MATRIX = ini_file.get_list_from_ini(   section="Execution", key="Image-Transfer-Matrix",         default=[0, 1, 0], _type=int)
CROP                  = ini_file.get_list_from_ini(   section="Execution", key="Crop",                          default=[-1],      _type=int)
ESTIMATION_METHOD     = ini_file.get_string_from_ini( section="Execution", key="Estimation-Method",             default='simple_speckle')
PROPAGATOR            = ini_file.get_string_from_ini( section="Execution", key="Propagator",                    default='RS')
IMAGE_OPS             = ini_file.get_dict_from_ini(   section="Execution", key="Image-Ops",                     default={"file" : [], "stream" :["T", "FH", "FV"]})

DARK                  = ini_file.get_string_from_ini( section="Reconstruction", key="Dark",  default=None)
FLAT                  = ini_file.get_string_from_ini( section="Reconstruction", key="Flat",  default=None)
CALIBRATION_PATH      = ini_file.get_string_from_ini( section="Reconstruction", key="Calibration-Path",  default=None)
MODE                  = ini_file.get_string_from_ini( section="Reconstruction", key="Mode",              default='centralLine')
LINE_WIDTH            = ini_file.get_int_from_ini(    section="Reconstruction", key="Line-Width",        default=10)
REBINNING             = ini_file.get_float_from_ini(  section="Reconstruction", key="Rebinning",         default=1.0)
DOWN_SAMPLING         = ini_file.get_float_from_ini(  section="Reconstruction", key="Down-Sampling",     default=1.0)
METHOD                = ini_file.get_string_from_ini( section="Reconstruction", key="Method",            default='WXST')

SPINNET_CONFIGURATION = ini_file.get_dict_from_ini(   section="Reconstruction", key="SPINNet-Configuration",
                                                      default={"SPINNet" :   {"type": "PO", "folder": "Result_pxShift_data_10k_T0p2_feature10_fp16_search3_longerTraining",       "model" : "training_model_002000.pt",          "setting" : "setting_002000.json"},
                                                               "SPINNetSD" : {"type": "PO", "folder": "SpeckleDisplacementNet_05-01_12hr_mirror_10k_EdgePad_Beta_2-5_04_18_2025", "model" : "best_model_epoch_3268_Val_0.00448.pt", "setting" : "training_results.json"}})

USE_GPU               = ini_file.get_boolean_from_ini(section="Reconstruction", key="Use-Gpu",           default=False)
# [GPU DEVICE SELECTION] (ADDED) which physical GPU CUDA_VISIBLE_DEVICES should
# expose for batch processing. Default 1 matches a 2-GPU workstation where
# device 0 drives the display and only device 1 is available for compute.
GPU_DEVICE_INDEX      = ini_file.get_int_from_ini(    section="Reconstruction", key="Gpu-Device-Index",  default=1)
# [GPU DEVICE SELECTION] (CHANGED) was set only inside process_images()'s
# BATCH branch, which a normal single-image GUI "Run" never reaches (it calls
# process_image() -> _process_image() directly, same bypass pattern as the
# GPU cache-release bug). Set here instead, at module import time, so it's
# guaranteed to run before ANY processing function -- and, crucially, before
# any CUDA call in this process, since CUDA_VISIBLE_DEVICES only has an
# effect if set before the first CUDA touch; once any torch.cuda call has run
# in a process, the device selection is locked for that process's lifetime.
# This means a NEW Gpu-Device-Index value only takes effect after a full
# restart of the launcher process -- editing the json and re-running within
# the same still-running session will not retroactively change it.
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(GPU_DEVICE_INDEX)
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
FILTER_INTENSITY        = ini_file.get_string_from_ini( section="Back-Propagation", key="Filter-Intensity",            default="gaussian")
SIGMA_INTENSITY         = ini_file.get_int_from_ini(    section="Back-Propagation", key="Sigma-Intensity",             default=21)
SMOOTH_PHASE            = ini_file.get_boolean_from_ini(section="Back-Propagation", key="Smooth-Phase",                default=False)
FILTER_PHASE            = ini_file.get_string_from_ini( section="Back-Propagation", key="Filter-Phase",                default="gaussian")
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
RMS_RANGE_V             = ini_file.get_list_from_ini(   section="Back-Propagation", key="RMS-Range-V",                 default=[-2e-6, 2e-6], _type=float)
RMS_RANGE_H             = ini_file.get_list_from_ini(   section="Back-Propagation", key="RMS-Range-H",                 default=[-2e-6, 2e-6], _type=float)
ENGINE                  = ini_file.get_string_from_ini( section="Back-Propagation", key="Engine",                      default="WOFRY")

# WOFRY
MAGNIFICATION_V         = ini_file.get_float_from_ini(  section="Back-Propagation", key="Magnification-V",             default=0.028)
MAGNIFICATION_H         = ini_file.get_float_from_ini(  section="Back-Propagation", key="Magnification-H",             default=0.028)
SHIFT_HALF_PIXEL        = ini_file.get_boolean_from_ini(section="Back-Propagation", key="Shift-Half-Pixel",            default=False)

# SRW
AUTO_RESIZE_BEFORE_PROPAGATION                         = ini_file.get_boolean_from_ini(section="Back-Propagation", key="Auto-Resize-Before-Propagation",                         default=False)
AUTO_RESIZE_AFTER_PROPAGATION                          = ini_file.get_boolean_from_ini(section="Back-Propagation", key="Auto-Resize-After-Propagation",                          default=False)
RELATIVE_PRECISION_FOR_PROPAGATION_WITH_AUTORESIZING   = ini_file.get_float_from_ini(  section="Back-Propagation", key="Relative-Precision-For-Propagation-With-Autoresizing",   default=1.0)
ALLOW_SEMIANALYTICAL_TREATMENT_OF_QUADRATIC_PHASE_TERM = ini_file.get_int_from_ini(    section="Back-Propagation", key="Allow-Semianalytical-Treatment-Of-Quadratic-Phase-Term", default=1)
DO_ANY_RESIZING_ON_FOURIER_SIDE_USING_FFT              = ini_file.get_boolean_from_ini(section="Back-Propagation", key="Do-Any-Resizing-On-Fourier-Side-Using-FFT",              default=False)
HORIZONTAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING       = ini_file.get_float_from_ini(  section="Back-Propagation", key="Horizontal-Range-Modification-Factor-At-Resizing",       default=1.0)
HORIZONTAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING  = ini_file.get_float_from_ini(  section="Back-Propagation", key="Horizontal-Resolution-Modification-Factor-At-Resizing",  default=1.0)
VERTICAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING         = ini_file.get_float_from_ini(  section="Back-Propagation", key="Vertical-Range-Modification-Factor-At-Resizing",         default=1.0)
VERTICAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING    = ini_file.get_float_from_ini(  section="Back-Propagation", key="Vertical-Resolution-Modification-Factor-At-Resizing",    default=1.0)

SCAN_BEST_FOCUS         = ini_file.get_boolean_from_ini(section="Back-Propagation", key="Scan-Best-Focus",             default=False)
USE_FIT                 = ini_file.get_boolean_from_ini(section="Back-Propagation", key="USe-Fit",                     default=True)
BEST_FOCUS_FROM         = ini_file.get_string_from_ini( section="Back-Propagation", key="Best-Focus-From",             default="rms")
BEST_FOCUS_SCAN_RANGE   = ini_file.get_list_from_ini(   section="Back-Propagation", key="2D, Best-Focus-Scan-Range",   default=[-0.001, 0.001, 0.0001], _type=float)
BEST_FOCUS_SCAN_RANGE_V = ini_file.get_list_from_ini(   section="Back-Propagation", key="1D, Best-Focus-Scan-Range-V", default=[-0.001, 0.001, 0.0001], _type=float)
BEST_FOCUS_SCAN_RANGE_H = ini_file.get_list_from_ini(   section="Back-Propagation", key="1D, Best-Focus-Scan-Range-H", default=[-0.001, 0.001, 0.0001], _type=float)
# [APERTURE SHAPE] (ADDED) 2D back-propagation aperture: "rectangular" (default), "circular", or "elliptical"
APERTURE_SHAPE          = ini_file.get_string_from_ini( section="Back-Propagation", key="Aperture-Shape",                default="rectangular")

SHOW_ALIGN_FIGURE     = ini_file.get_boolean_from_ini(section="Output", key="Show-Align-Figure",     default=False)
CORRECT_SCALE         = ini_file.get_boolean_from_ini(section="Output", key="Correct-Scale",         default=False)

# WSVT
WSVT_SCAN_POSITIONS_FILE = ini_file.get_string_from_ini( section="WSVT", key="Scan-Positions-File", default=None)
WSVT_N_SCAN              = ini_file.get_int_from_ini(    section="WSVT", key="N-Scan",              default=51)
WSVT_AUTO_SIGN           = ini_file.get_boolean_from_ini(section="WSVT", key="Auto-Sign",           default=True)
WSVT_SIGN_X              = ini_file.get_int_from_ini(    section="WSVT", key="Sign-X",              default=1)
WSVT_SIGN_Y              = ini_file.get_int_from_ini(    section="WSVT", key="Sign-Y",              default=1)
WSVT_POSITION_UNITS      = ini_file.get_string_from_ini( section="WSVT", key="Position-Units",      default="mm")

def store():
    ini_file.set_value_at_ini(section="General", key="Data-Directory", value=DATA_DIRECTORY)

    ini_file.set_value_at_ini(section="Mask", key="Pattern-Size",         value=PATTERN_SIZE)
    ini_file.set_value_at_ini(section="Mask", key="Pattern-Thickness",    value=PATTERN_THICKNESS)
    ini_file.set_value_at_ini(section="Mask", key="Pattern-Transmission", value=PATTERN_TRANSMISSION)
    ini_file.set_value_at_ini(section="Mask", key="Pattern-Image",        value=RAN_MASK)
    ini_file.set_value_at_ini(section="Mask", key="Propagation-Distance", value=PROPAGATION_DISTANCE)
    # [MASK EXPOSURE MODEL] (ADDED)
    ini_file.set_value_at_ini(section="Mask", key="Exposure-Model",        value=EXPOSURE_MODEL)
    ini_file.set_value_at_ini(section="Mask", key="Exposure-Auto",         value=EXPOSURE_AUTO)
    ini_file.set_value_at_ini(section="Mask", key="Exposure-Bias",         value=EXPOSURE_BIAS)
    ini_file.set_value_at_ini(section="Mask", key="Exposure-Corner-Sigma", value=EXPOSURE_CORNER_SIGMA)
    ini_file.set_value_at_ini(section="Mask", key="Exposure-Supersample",  value=EXPOSURE_SUPERSAMPLE)
    ini_file.set_value_at_ini(section="Mask", key="Exposure-Estimate",     value=EXPOSURE_ESTIMATE)
    ini_file.set_list_at_ini( section="Mask", key="Exposure-Bias-Grid",    values_list=EXPOSURE_BIAS_GRID)

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
    ini_file.set_value_at_ini(section="Back-Propagation", key="Filter-Intensity",           value=FILTER_INTENSITY)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Sigma-Intensity",            value=SIGMA_INTENSITY )
    ini_file.set_value_at_ini(section="Back-Propagation", key="Smooth-Phase",               value=SMOOTH_PHASE)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Filter-Phase",               value=FILTER_PHASE)
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
    ini_file.set_value_at_ini(section="Back-Propagation", key="Engine",                     value=ENGINE)
    
    # WOFRY
    ini_file.set_value_at_ini(section="Back-Propagation", key="Magnification-V",            value=MAGNIFICATION_V)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Magnification-H",            value=MAGNIFICATION_H)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Shift-Half-Pixel",           value=SHIFT_HALF_PIXEL)

    # SRW
    ini_file.set_value_at_ini(section="Back-Propagation", key="Auto-Resize-Before-Propagation",                         value=AUTO_RESIZE_BEFORE_PROPAGATION                        )
    ini_file.set_value_at_ini(section="Back-Propagation", key="Auto-Resize-After-Propagation",                          value=AUTO_RESIZE_AFTER_PROPAGATION                         )
    ini_file.set_value_at_ini(section="Back-Propagation", key="Relative-Precision-For-Propagation-With-Autoresizing",   value=RELATIVE_PRECISION_FOR_PROPAGATION_WITH_AUTORESIZING  )
    ini_file.set_value_at_ini(section="Back-Propagation", key="Allow-Semianalytical-Treatment-Of-Quadratic-Phase-Term", value=ALLOW_SEMIANALYTICAL_TREATMENT_OF_QUADRATIC_PHASE_TERM)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Do-Any-Resizing-On-Fourier-Side-Using-FFT",              value=DO_ANY_RESIZING_ON_FOURIER_SIDE_USING_FFT             )
    ini_file.set_value_at_ini(section="Back-Propagation", key="Horizontal-Range-Modification-Factor-At-Resizing",       value=HORIZONTAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING      )
    ini_file.set_value_at_ini(section="Back-Propagation", key="Horizontal-Resolution-Modification-Factor-At-Resizing",  value=HORIZONTAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING )
    ini_file.set_value_at_ini(section="Back-Propagation", key="Vertical-Range-Modification-Factor-At-Resizing",         value=VERTICAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING        )
    ini_file.set_value_at_ini(section="Back-Propagation", key="Vertical-Resolution-Modification-Factor-At-Resizing",    value=VERTICAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING   )

    ini_file.set_list_at_ini( section="Back-Propagation", key="RMS-Range-V",                values_list=RMS_RANGE_V)
    ini_file.set_list_at_ini( section="Back-Propagation", key="RMS-Range-H",                values_list=RMS_RANGE_H)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Scan-Best-Focus",            value=SCAN_BEST_FOCUS)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Use-Fit",                    value=USE_FIT)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Best-Focus-From",            value=BEST_FOCUS_FROM)
    ini_file.set_list_at_ini( section="Back-Propagation", key="2D, Best-Focus-Scan-Range",      values_list=BEST_FOCUS_SCAN_RANGE)
    ini_file.set_list_at_ini( section="Back-Propagation", key="1D, Best-Focus-Scan-Range-V",    values_list=BEST_FOCUS_SCAN_RANGE_V)
    ini_file.set_list_at_ini( section="Back-Propagation", key="1D, Best-Focus-Scan-Range-H",    values_list=BEST_FOCUS_SCAN_RANGE_H)
    # [APERTURE SHAPE] (ADDED)
    ini_file.set_value_at_ini(section="Back-Propagation", key="Aperture-Shape",             value=APERTURE_SHAPE)

    ini_file.set_value_at_ini(section="Reconstruction", key="Dark",           value=DARK)
    ini_file.set_value_at_ini(section="Reconstruction", key="Flat",           value=FLAT)
    ini_file.set_value_at_ini(section="Reconstruction", key="Mode",           value=MODE)
    ini_file.set_value_at_ini(section="Reconstruction", key="Line-Width",     value=LINE_WIDTH)
    ini_file.set_value_at_ini(section="Reconstruction", key="Rebinning",      value=REBINNING)
    ini_file.set_value_at_ini(section="Reconstruction", key="Down-Sampling",  value=DOWN_SAMPLING)
    ini_file.set_value_at_ini(section="Reconstruction", key="Method",         value=METHOD)
    ini_file.set_dict_at_ini( section="Reconstruction", key="SPINNet-Configuration", values_dict=SPINNET_CONFIGURATION)
    ini_file.set_value_at_ini(section="Reconstruction", key="Use-Gpu",        value=USE_GPU)
    # [GPU DEVICE SELECTION] (ADDED)
    ini_file.set_value_at_ini(section="Reconstruction", key="Gpu-Device-Index", value=GPU_DEVICE_INDEX)
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

    # WSVT
    ini_file.set_value_at_ini(section="WSVT", key="Scan-Positions-File", value=WSVT_SCAN_POSITIONS_FILE)
    ini_file.set_value_at_ini(section="WSVT", key="N-Scan",              value=WSVT_N_SCAN)
    ini_file.set_value_at_ini(section="WSVT", key="Auto-Sign",           value=WSVT_AUTO_SIGN)
    ini_file.set_value_at_ini(section="WSVT", key="Sign-X",              value=WSVT_SIGN_X)
    ini_file.set_value_at_ini(section="WSVT", key="Sign-Y",              value=WSVT_SIGN_Y)
    ini_file.set_value_at_ini(section="WSVT", key="Position-Units",      value=WSVT_POSITION_UNITS)

    ini_file.push()

store()

class AbsolutePhaseAnalyzer(IAbsolutePhaseAnalyzer):
    def __init__(self,
                 data_collection_directory,
                 file_name_prefix=None,
                 simulated_mask_directory=None,
                 energy=ENERGY):
        self.__data_collection_directory = data_collection_directory
        self.__file_name_prefix          = file_name_prefix if not file_name_prefix is None else ws.get_file_name_prefix() # TODO: here fnp must be different. Custom for offline otherwise the WS will dominate
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
        image, hh, vv = ws.get_image_data(measurement_directory=self.__data_collection_directory if data_collection_directory is None else data_collection_directory,
                                          file_name_prefix=self.__file_name_prefix,
                                          image_index=image_index,
                                          **kwargs)
        return hh, vv, image

    def process_image(self, image_index: int, data_collection_directory: str = None, **kwargs):
        return _process_image(data_collection_directory=self.__data_collection_directory if data_collection_directory is None else data_collection_directory,
                              file_name_prefix=self.__file_name_prefix,
                              mask_directory=self.__simulated_mask_directory,
                              energy=self.__energy,
                              image_index=image_index,
                              **kwargs)

    def process_images_WSVT(self, data_collection_directory: str = None, **kwargs):
        return _process_images_WSVT(data_collection_directory=self.__data_collection_directory if data_collection_directory is None else data_collection_directory,
                                    file_name_prefix=self.__file_name_prefix,
                                    mask_directory=self.__simulated_mask_directory,
                                    energy=self.__energy,
                                    **kwargs)

    def process_images(self, data_collection_directory: str = None, mode=ProcessingMode.LIVE, n_threads=MAX_THREADS, **kwargs):
        data_collection_directory = self.__data_collection_directory if data_collection_directory is None else data_collection_directory
        index_digits              = kwargs.get("index_digits", ws.INDEX_DIGITS)

        if mode == ProcessingMode.LIVE:
            for file in os.listdir(data_collection_directory):

                if   pathlib.Path(file).suffix == ".tif"  and self.__file_name_prefix in file: extension = ".tif"
                elif pathlib.Path(file).suffix == ".hdf5" and self.__file_name_prefix in file: extension = ".hdf5"
                else: continue

                # [GPU CACHE RELEASE] moved into _process_image itself (called
                # below), since it's the single shared function every caller
                # funnels through -- fixing it there covers this loop too.
                self.process_image(image_index=int(file.split(extension)[0][-index_digits:]), verbose=kwargs.get("verbose", False))
        else:
            # [GPU DEVICE SELECTION] moved to module level (see GPU_DEVICE_INDEX
            # above) since this BATCH branch is not the only, or even the
            # normal, entry point -- setting it here alone left the common
            # single-image "Run" path unaffected.

            self.__active_threads = [None] * n_threads

            for i in range(n_threads):
                self.__active_threads[i] = ProcessingThread(thread_id=i+1,
                                                            data_collection_directory=data_collection_directory,
                                                            file_name_prefix=self.__file_name_prefix,
                                                            index_digits=index_digits,
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
                 index_digits,
                 simulated_mask_directory,
                 energy, 
                 **kwargs):
        super(ProcessingThread, self).__init__(name="Thread #" + str(thread_id))
        self.__thread_id = thread_id
        self.__data_collection_directory = data_collection_directory
        self.__file_name_prefix          = file_name_prefix
        self.__index_digits              = index_digits
        self.__simulated_mask_directory  = simulated_mask_directory
        self.__energy                    = energy
        self.__kwargs                    = kwargs

    @synchronized_method
    def check_new_data(self, images_list):
        image_indexes = []
        result_folder_list = glob.glob(os.path.join(os.path.dirname(images_list[0]), '*'))
        result_folder_list = [os.path.basename(f) for f in result_folder_list]

        for image in images_list:
            image_directory = os.path.basename(image).split(pathlib.Path(image).suffix)[0]
            if image_directory in result_folder_list:
                continue
            else:
                image_indexes.append(int(image_directory[-self.__index_digits:]))
        return image_indexes

    def run(self):
        max_waiting_cycles = 60
        waiting_cycles     = 0

        while waiting_cycles < max_waiting_cycles:
            images_list_tif  = glob.glob(os.path.join(self.__data_collection_directory, self.__file_name_prefix + '_*.tif'), recursive=False)
            images_list_hdf5 = glob.glob(os.path.join(self.__data_collection_directory, self.__file_name_prefix + '_*.hdf5'), recursive=False)

            images_list = list(set(images_list_tif + images_list_hdf5))

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

                    # [GPU CACHE RELEASE] moved into _process_image itself
                    # (called here), since it's the single shared function
                    # every caller funnels through -- fixing it there covers
                    # this loop too.
                    for image_index in image_indexes[0:n]:
                        _process_image(self.__data_collection_directory,
                                       self.__file_name_prefix,
                                       self.__simulated_mask_directory,
                                       self.__energy,
                                       image_index,
                                       **self.__kwargs)
            time.sleep(1)

        print('Thread #' + str(self.__thread_id) + ' completed')

from aps.wf_suite.driver.wavefront_sensor import get_image_file_path

def _process_image(data_collection_directory, file_name_prefix, mask_directory, energy, image_index, **kwargs):
    data_directory = DATA_DIRECTORY

    index_digits    = kwargs.get("index_digits", None)
    verbose         = kwargs.get("verbose", False)
    image_file_name = kwargs.get("image_file_name", get_image_file_path(measurement_directory=data_collection_directory,
                                                                             file_name_prefix=file_name_prefix,
                                                                             image_index=image_index,
                                                                             index_digits=index_digits))
    image_data      = kwargs.get("image_data", None)

    use_flat = kwargs.get("use_flat")
    use_dark = kwargs.get("use_dark")

    dark           = None if (DARK is None or not use_dark) else os.path.join(data_collection_directory, DARK)
    flat           = None if (FLAT is None or not use_flat) else os.path.join(data_collection_directory, FLAT)
    mask_directory = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    result_folder  = os.path.join(os.path.dirname(image_file_name),
                                  os.path.basename(image_file_name).split(pathlib.Path(image_file_name).suffix)[0])



    # pattern simulation parameters
    pattern_path          = os.path.join(data_directory, 'absolute_phase', 'mask', RAN_MASK)
    propagated_pattern    = os.path.join(mask_directory, 'propagated_pattern.npz')
    propagated_patternDet = os.path.join(mask_directory, 'propagated_patternDet.npz')
    saving_path           = mask_directory

    method = kwargs.get("method", METHOD)
    spinnet_configuration = kwargs.get("spinnet_configuration", SPINNET_CONFIGURATION).get(method, {})
    trained_model_type    = spinnet_configuration.get("type", "")
    trained_model_folder  = spinnet_configuration.get("folder", "")
    trained_model         = spinnet_configuration.get("model", "")
    setting_path          = spinnet_configuration.get("setting", "")

    # [GPU CACHE RELEASE] (CHANGED) capture the result instead of returning
    # directly, so we can release cached-but-unused GPU memory before
    # returning. This is the single shared function every caller (GUI single
    # "Run", BATCH loop, LIVE polling loop) funnels through, so fixing it here
    # covers all of them -- unlike the earlier per-loop attempts in
    # process_images()/ProcessingThread.run(), which never fire for a plain
    # single-image GUI run since that path calls process_image() -> here
    # directly, bypassing those loops entirely.
    # try/finally so an exception raised anywhere inside execute_process_image
    # (e.g. during result plotting, after the GPU-heavy computation already
    # ran) still releases the cache before the error propagates.
    use_gpu_flag = kwargs.get("use_gpu", USE_GPU)
    try:
        result = execute_process_image(img=image_file_name,
                                     image_data=image_data,
                                     dark=dark,
                                     flat=flat,
                                     result_folder=result_folder,
                                     data_directory=data_directory,
                                     pattern_path=pattern_path,
                                     propagated_pattern=propagated_pattern,
                                     propagated_patternDet=propagated_patternDet,
                                     saving_path=saving_path,
                                     crop=kwargs.get("crop", CROP),
                                     img_transfer_matrix=kwargs.get("image_transfer_matrix", IMAGE_TRANSFER_MATRIX),
                                     find_transferMatrix=False, # always false for just processing images
                                     p_x=kwargs.get("pixel_size", ws.PIXEL_SIZE),
                                     det_res=kwargs.get("detector_resolution", ws.DETECTOR_RESOLUTION),
                                     energy=energy,
                                     pattern_size=kwargs.get("pattern_size", PATTERN_SIZE),
                                     pattern_thickness=kwargs.get("pattern_thickness", PATTERN_THICKNESS),
                                     pattern_T=kwargs.get("pattern_transmission", PATTERN_TRANSMISSION),
                                     d_prop=kwargs.get("propagation_distance", PROPAGATION_DISTANCE),
                                     # [MASK EXPOSURE MODEL] (ADDED) forward ini/kwargs to the executor;
                                     # exposure_bias=None triggers auto-estimation.
                                     exposure_model=kwargs.get("exposure_model", EXPOSURE_MODEL),
                                     exposure_bias=(None if kwargs.get("exposure_auto", EXPOSURE_AUTO)
                                                    else kwargs.get("exposure_bias", EXPOSURE_BIAS)),
                                     exposure_corner_sigma=kwargs.get("exposure_corner_sigma", EXPOSURE_CORNER_SIGMA),
                                     exposure_supersample=kwargs.get("exposure_supersample", EXPOSURE_SUPERSAMPLE),
                                     exposure_estimate=kwargs.get("exposure_estimate", EXPOSURE_ESTIMATE),
                                     exposure_bias_grid=kwargs.get("exposure_bias_grid", EXPOSURE_BIAS_GRID),
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
                                     trained_model_type=trained_model_type,
                                     trained_model_folder=trained_model_folder,
                                     trained_model=trained_model,
                                     setting_path=setting_path,
                                     GPU=use_gpu_flag,
                                     use_wavelet=kwargs.get("use_wavelet", USE_WAVELET),
                                     wavelet_lv_cut=kwargs.get("wavelet_lv_cut", WAVELET_CUT),
                                     n_iter=kwargs.get("n_iterations", N_ITERATIONS),
                                     pyramid_level=kwargs.get("pyramid_level", PYRAMID_LEVEL),
                                     template_size=kwargs.get("template_size", TEMPLATE_SIZE),
                                     window_searching=kwargs.get("window_search", WINDOW_SEARCH),
                                     nCores=kwargs.get("n_cores", N_CORES),
                                      nGroup=kwargs.get("n_group", N_GROUP),
                                      verbose=verbose)
    finally:
        if use_gpu_flag:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return result

def _process_images_WSVT(data_collection_directory, file_name_prefix, mask_directory, energy, **kwargs):
    data_directory = DATA_DIRECTORY

    index_digits = kwargs.get("index_digits", None)

    use_flat = kwargs.get("use_flat")
    use_dark = kwargs.get("use_dark")

    dark           = None if (DARK is None or not use_dark) else os.path.join(data_collection_directory, DARK)
    flat           = None if (FLAT is None or not use_flat) else os.path.join(data_collection_directory, FLAT)
    mask_directory = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory

    # Use the first image's name as the result folder (same convention as WXST)
    # so that the GUI and back-propagation can find the results
    image_file_name = get_image_file_path(measurement_directory=data_collection_directory,
                                          file_name_prefix=file_name_prefix,
                                          image_index=1,
                                          index_digits=index_digits)
    result_folder  = os.path.join(os.path.dirname(image_file_name),
                                  os.path.basename(image_file_name).split(pathlib.Path(image_file_name).suffix)[0])

    # pattern simulation parameters
    pattern_path          = os.path.join(data_directory, 'absolute_phase', 'mask', RAN_MASK)
    propagated_pattern    = os.path.join(mask_directory, 'propagated_pattern.npz')
    propagated_patternDet = os.path.join(mask_directory, 'propagated_patternDet.npz')
    simulated_ref_stack   = os.path.join(mask_directory, 'simulated_ref_stack.npz')
    saving_path           = mask_directory

    # [GPU CACHE RELEASE] (CHANGED) capture the result instead of returning
    # directly -- see the matching comment in _process_image above. try/finally
    # so an exception anywhere inside execute_process_images_WSVT still
    # releases the cache before the error propagates.
    use_gpu_flag = kwargs.get("use_gpu", USE_GPU)
    try:
        result = execute_process_images_WSVT(
            image_directory=data_collection_directory,
            scan_positions_file=kwargs.get("scan_positions_file", WSVT_SCAN_POSITIONS_FILE),
            data_directory=data_directory,
            result_folder=result_folder,
            pattern_path=pattern_path,
            propagated_pattern=propagated_pattern,
            propagated_patternDet=propagated_patternDet,
            simulated_ref_stack=simulated_ref_stack,
            process_after_mask=True,
            saving_path=saving_path,
            n_scan=kwargs.get("n_scan", WSVT_N_SCAN),
            sign_x=kwargs.get("sign_x", WSVT_SIGN_X),
            sign_y=kwargs.get("sign_y", WSVT_SIGN_Y),
            auto_sign=kwargs.get("auto_sign", WSVT_AUTO_SIGN),
            position_units=kwargs.get("position_units", WSVT_POSITION_UNITS),
            p_x=kwargs.get("pixel_size", ws.PIXEL_SIZE),
            det_res=kwargs.get("detector_resolution", ws.DETECTOR_RESOLUTION),
            energy=energy,
            pattern_size=kwargs.get("pattern_size", PATTERN_SIZE),
            pattern_thickness=kwargs.get("pattern_thickness", PATTERN_THICKNESS),
            pattern_T=kwargs.get("pattern_transmission", PATTERN_TRANSMISSION),
            d_prop=kwargs.get("propagation_distance", PROPAGATION_DISTANCE),
            # [MASK EXPOSURE MODEL] (ADDED) forward ini/kwargs to the WSVT executor;
            # exposure_bias=None triggers auto-estimation.
            exposure_model=kwargs.get("exposure_model", EXPOSURE_MODEL),
            exposure_bias=(None if kwargs.get("exposure_auto", EXPOSURE_AUTO)
                           else kwargs.get("exposure_bias", EXPOSURE_BIAS)),
            exposure_corner_sigma=kwargs.get("exposure_corner_sigma", EXPOSURE_CORNER_SIGMA),
            exposure_supersample=kwargs.get("exposure_supersample", EXPOSURE_SUPERSAMPLE),
            exposure_estimate=kwargs.get("exposure_estimate", EXPOSURE_ESTIMATE),
            exposure_bias_grid=kwargs.get("exposure_bias_grid", EXPOSURE_BIAS_GRID),
            d_source_v=kwargs.get("source_distance_v", SOURCE_DISTANCE_V),
            d_source_h=kwargs.get("source_distance_h", SOURCE_DISTANCE_H),
            source_v=kwargs.get("source_size_v", SOURCE_V),
            source_h=kwargs.get("source_size_h", SOURCE_H),
            correct_scale=kwargs.get("correct_scale", CORRECT_SCALE),
            show_alignFigure=kwargs.get("show_align_figure", SHOW_ALIGN_FIGURE),
            d_source_recal=kwargs.get("source_distance_recalculation", D_SOURCE_RECAL),
            estimation_method=kwargs.get("estimation_method", ESTIMATION_METHOD),
            propagator=kwargs.get("propagator", PROPAGATOR),
            # [DETECTOR CALIBRATION] (ADDED) forward ini/kwargs to the WSVT executor.
            cali_path=kwargs.get("calibration_path", CALIBRATION_PATH),
            img_transfer_matrix=kwargs.get("image_transfer_matrix", IMAGE_TRANSFER_MATRIX),
            find_transferMatrix=False,
            crop=kwargs.get("crop", CROP),
            dark=dark,
            flat=flat,
            rebinning=kwargs.get("rebinning", REBINNING),
            lineWidth=kwargs.get("line_width", LINE_WIDTH),
            cal_half_window=kwargs.get("window_search", WINDOW_SEARCH),
            n_cores=kwargs.get("n_cores", N_CORES),
            n_group=kwargs.get("n_group", N_GROUP),
            use_wavelet=kwargs.get("use_wavelet", USE_WAVELET),
            wavelet_lv_cut=kwargs.get("wavelet_lv_cut", WAVELET_CUT),
            pyramid_level=kwargs.get("pyramid_level", PYRAMID_LEVEL),
            n_iter=kwargs.get("n_iterations", N_ITERATIONS),
            use_GPU=use_gpu_flag,
            scaling_x=kwargs.get("scaling_x", 1.0),
            scaling_y=kwargs.get("scaling_y", 1.0),
            verbose=kwargs.get("verbose", False),
            save_images=kwargs.get("save_images", True))
    finally:
        if use_gpu_flag:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return result

def _generate_simulated_mask(data_collection_directory, file_name_prefix, mask_directory, energy, image_index=1, **kwargs) -> [list, bool]:
    index_digits = kwargs.get("index_digits", ws.INDEX_DIGITS)
    verbose      = kwargs.get("verbose", False)

    use_flat = kwargs.get("use_flat")
    use_dark = kwargs.get("use_dark")

    dark = None if (DARK is None or not use_dark) else os.path.join(data_collection_directory, DARK)
    flat = None if (FLAT is None or not use_flat) else os.path.join(data_collection_directory, FLAT)
    image_file_name = kwargs.get("image_file_name", get_image_file_path(measurement_directory=data_collection_directory,
                                                                        file_name_prefix=file_name_prefix,
                                                                        image_index=image_index,
                                                                        index_digits=index_digits))
    image_data  = kwargs.get("image_data", None)

    mask_directory  = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    result_folder  = os.path.join(os.path.dirname(image_file_name),
                                  os.path.basename(image_file_name).split(pathlib.Path(image_file_name).suffix)[0])

    pattern_path    = os.path.join(DATA_DIRECTORY, 'absolute_phase', 'mask', RAN_MASK)
    saving_path     = mask_directory

    # For mask generation, use "simple" method if WSVT is selected since the
    # single-shot executor doesn't support WSVT as a tracking method.
    # The mask generation only needs pattern simulation, not speckle tracking.
    mask_gen_method = kwargs.get("method", METHOD)
    if mask_gen_method == "WSVT":
        mask_gen_method = "simple"

    if not os.path.exists(mask_directory): os.mkdir(mask_directory)

    if not os.path.exists(os.path.join(mask_directory, 'propagated_pattern.npz')) or \
       not os.path.exists(os.path.join(mask_directory, 'propagated_patternDet.npz')) or \
       not os.path.exists(os.path.join(mask_directory, "reference.json")):
        # [GPU CACHE RELEASE] (CHANGED) try/finally so an exception anywhere
        # inside execute_process_image still releases the cache before the
        # error propagates -- see matching comment in _process_image.
        use_gpu_flag = kwargs.get("use_gpu", USE_GPU)
        try:
            execute_process_image(img=image_file_name,
                                  image_data=image_data,
                                  dark=dark,
                                  flat=flat,
                                  result_folder=result_folder,
                                  data_directory=DATA_DIRECTORY,
                                  pattern_path=pattern_path,
                                  propagated_pattern=None,
                                  propagated_patternDet=None,
                                  saving_path=saving_path,
                                  crop=kwargs.get("crop", CROP),
                                  img_transfer_matrix=None if FIND_TRANSFER_MATRIX else IMAGE_TRANSFER_MATRIX,
                                  find_transferMatrix=FIND_TRANSFER_MATRIX,
                                  p_x=kwargs.get("pixel_size", ws.PIXEL_SIZE),
                                  det_res=kwargs.get("detector_resolution", ws.DETECTOR_RESOLUTION),
                                  energy=energy,
                                  pattern_size=kwargs.get("pattern_size", PATTERN_SIZE),
                                  pattern_thickness=kwargs.get("pattern_thickness", PATTERN_THICKNESS),
                                  pattern_T=kwargs.get("pattern_transmission", PATTERN_TRANSMISSION),
                                  d_prop=kwargs.get("propagation_distance", PROPAGATION_DISTANCE),
                                  # [MASK EXPOSURE MODEL] (ADDED) forward ini/kwargs to the executor;
                                  # exposure_bias=None triggers auto-estimation. This call site
                                  # (_generate_simulated_mask, used by the "Generate Mask" GUI action)
                                  # was previously missing these entirely, so execute_process_image
                                  # fell back to its own internal default (exposure_model=None)
                                  # instead of the configured EXPOSURE_MODEL.
                                  exposure_model=kwargs.get("exposure_model", EXPOSURE_MODEL),
                                  exposure_bias=(None if kwargs.get("exposure_auto", EXPOSURE_AUTO)
                                                 else kwargs.get("exposure_bias", EXPOSURE_BIAS)),
                                  exposure_corner_sigma=kwargs.get("exposure_corner_sigma", EXPOSURE_CORNER_SIGMA),
                                  exposure_supersample=kwargs.get("exposure_supersample", EXPOSURE_SUPERSAMPLE),
                                  exposure_estimate=kwargs.get("exposure_estimate", EXPOSURE_ESTIMATE),
                                  exposure_bias_grid=kwargs.get("exposure_bias_grid", EXPOSURE_BIAS_GRID),
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
                                  method=mask_gen_method,
                                  GPU=use_gpu_flag,
                                  use_wavelet=kwargs.get("use_wavelet", USE_WAVELET),
                                  wavelet_lv_cut=kwargs.get("wavelet_lv_cut", WAVELET_CUT),
                                  n_iter=kwargs.get("n_iterations", N_ITERATIONS),
                                  pyramid_level=kwargs.get("pyramid_level", PYRAMID_LEVEL),
                                  template_size=kwargs.get("template_size", TEMPLATE_SIZE),
                                  window_searching=kwargs.get("window_search", WINDOW_SEARCH),
                                  nCores=kwargs.get("n_cores", N_CORES),
                                  nGroup=kwargs.get("n_group", N_GROUP),
                                  verbose=verbose)
        finally:
            if use_gpu_flag:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        is_new_mask = True
        print("Simulated mask generated in " + mask_directory)
    else:
        is_new_mask = False
        if verbose: print("Simulated mask already generated in " + mask_directory)

    with open(os.path.join(mask_directory, "reference.json"), 'r') as file: parameters = json.load(file)

    return parameters["image_transfer_matrix"], is_new_mask

def _backpropagate_wavefront(data_collection_directory, file_name_prefix, mask_directory, image_index, **kwargs) -> dict:
    index_digits   = kwargs.get("index_digits", ws.INDEX_DIGITS)
    index_digits   = index_digits if not index_digits is None else ws.INDEX_DIGITS
    folder         = kwargs.get("folder_name", os.path.join(data_collection_directory, (file_name_prefix + "_%0" + str(index_digits) + "i") % image_index))
    mask_directory = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory

    return execute_back_propagation(folder                 = folder,
                                    reference_folder       = mask_directory,
                                    kind                   = kwargs.get("kind", KIND),
                                    mask_detector_distance = kwargs.get("mask_detector_distance", PROPAGATION_DISTANCE),
                                    pixel_size             = kwargs.get("pixel_size", ws.PIXEL_SIZE),
                                    image_rebinning        = kwargs.get("image_rebinning", REBINNING),
                                    distance               = kwargs.get("propagation_distance", DISTANCE),
                                    distance_x             = kwargs.get("propagation_distance_h", DISTANCE_H),
                                    distance_y             = kwargs.get("propagation_distance_v", DISTANCE_V),
                                    rebinning              = kwargs.get("rebinning", REBINNING_BP),
                                    smooth_intensity       = kwargs.get("smooth_intensity", SMOOTH_INTENSITY),
                                    smooth_phase           = kwargs.get("smooth_phase", SMOOTH_PHASE),
                                    filter_intensity       = kwargs.get("filter_intensity", FILTER_INTENSITY),
                                    filter_phase           = kwargs.get("filter_phase", FILTER_PHASE),
                                    sigma_intensity        = kwargs.get("sigma_intensity", SIGMA_INTENSITY),
                                    sigma_phase            = kwargs.get("sigma_phase", SIGMA_PHASE),
                                    dim_x                  = kwargs.get("crop_h", CROP_H),
                                    dim_y                  = kwargs.get("crop_v", CROP_V),
                                    shift_x                = kwargs.get("crop_shift_h", CROP_SHIFT_H),
                                    shift_y                = kwargs.get("crop_shift_v", CROP_SHIFT_V),
                                    aperture_shape         = kwargs.get("aperture_shape", APERTURE_SHAPE),
                                    delta_f_x              = kwargs.get("delta_f_h", DELTA_F_H.get(METHOD, 0.0)),
                                    delta_f_y              = kwargs.get("delta_f_v", DELTA_F_V.get(METHOD, 0.0)),
                                    engine                 = kwargs.get("engine", ENGINE),
                                    magnification_x        = kwargs.get("magnification_h", MAGNIFICATION_H),
                                    magnification_y        = kwargs.get("magnification_v", MAGNIFICATION_V),
                                    shift_half_pixel       = kwargs.get("shift_half_pixel", SHIFT_HALF_PIXEL),
                                    auto_resize_before_propagation                         = kwargs.get("auto_resize_before_propagation", AUTO_RESIZE_BEFORE_PROPAGATION),
                                    auto_resize_after_propagation                          = kwargs.get("auto_resize_after_propagation", AUTO_RESIZE_AFTER_PROPAGATION),
                                    relative_precision_for_propagation_with_autoresizing   = kwargs.get("relative_precision_for_propagation_with_autoresizing", RELATIVE_PRECISION_FOR_PROPAGATION_WITH_AUTORESIZING),
                                    allow_semianalytical_treatment_of_quadratic_phase_term = kwargs.get("allow_semianalytical_treatment_of_quadratic_phase_term", ALLOW_SEMIANALYTICAL_TREATMENT_OF_QUADRATIC_PHASE_TERM),
                                    do_any_resizing_on_fourier_side_using_fft              = kwargs.get("do_any_resizing_on_fourier_side_using_fft", DO_ANY_RESIZING_ON_FOURIER_SIDE_USING_FFT),
                                    horizontal_range_modification_factor_at_resizing       = kwargs.get("horizontal_range_modification_factor_at_resizing", HORIZONTAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING),
                                    horizontal_resolution_modification_factor_at_resizing  = kwargs.get("horizontal_resolution_modification_factor_at_resizing", HORIZONTAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING),
                                    vertical_range_modification_factor_at_resizing         = kwargs.get("vertical_range_modification_factor_at_resizing", VERTICAL_RANGE_MODIFICATION_FACTOR_AT_RESIZING),
                                    vertical_resolution_modification_factor_at_resizing    = kwargs.get("vertical_resolution_modification_factor_at_resizing", VERTICAL_RESOLUTION_MODIFICATION_FACTOR_AT_RESIZING),
                                    x_rms_range            = kwargs.get("rms_range_h", RMS_RANGE_H),
                                    y_rms_range            = kwargs.get("rms_range_v", RMS_RANGE_V),
                                    show_figure            = kwargs.get("show_figure", False),
                                    save_result            = kwargs.get("save_result", False),
                                    scan_best_focus        = kwargs.get("scan_best_focus", SCAN_BEST_FOCUS),
                                    use_fit                = kwargs.get("use_fit", USE_FIT),
                                    best_focus_from        = kwargs.get("best_focus_from", BEST_FOCUS_FROM),
                                    scan_rel_range         = kwargs.get("best_focus_scan_range", BEST_FOCUS_SCAN_RANGE),
                                    scan_x_rel_range       = kwargs.get("best_focus_scan_range_h", BEST_FOCUS_SCAN_RANGE_H),
                                    scan_y_rel_range       = kwargs.get("best_focus_scan_range_v", BEST_FOCUS_SCAN_RANGE_V),
                                    verbose                = kwargs.get("verbose", False))