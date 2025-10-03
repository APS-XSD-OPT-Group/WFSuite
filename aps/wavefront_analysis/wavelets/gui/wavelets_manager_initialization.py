# !/usr/bin/env python
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

from aps.common.initializer import IniFacade
from aps.common.scripts.script_data import ScriptData
from aps.wavefront_analysis.wavelets import wavelets_analyzer as wa

def generate_initialization_parameters_from_ini(ini: IniFacade):
    # -----------------------------------------------------
    # Wavefront Analyzer

    wavelets_analyzer_configuration = {
        "common": {
            "distance": wa.DISTANCE,
            "energy": wa.ENERGY,
            "scaling_v": wa.SCALING_V,
            "scaling_h": wa.SCALING_H,
            "use_gpu": wa.USE_GPU,
            "use_wavelet": wa.USE_WAVELET,
            "wavelet_cut": wa.WAVELET_CUT,
            "pyramid_level": wa.PYRAMID_LEVEL,
            "n_iterations": wa.N_ITERATIONS,
            "template_size": wa.TEMPLATE_SIZE,
            "window_search": wa.WINDOW_SEARCH,
            "crop": wa.CROP,
            "down_sampling": wa.DOWN_SAMPLING,
            "rebinning": wa.REBINNING,
            "n_cores": wa.N_CORES,
            "n_group": wa.N_GROUP,
            "save_images": wa.SAVE_IMAGES,
            "verbose": wa.VERBOSE,
        },
        "WXST": {
            "WXST_image_file_name" : wa.WXST_IMAGE_FILE_NAME,
            "WXST_reference_file_name" : wa.WXST_REFERENCE_FILE_NAME,
            "WXST_dark_file_name" : wa.WXST_DARK_FILE_NAME,
            "WXST_flat_file_name" : wa.WXST_FLAT_FILE_NAME,
            "WXST_result_folder" : wa.WXST_RESULT_FOLDER,
        },
        "WSVT": {
            "WSVT_image_folder": wa.WSVT_IMAGE_FOLDER,
            "WSVT_reference_folder": wa.WSVT_REFERENCE_FOLDER,
            "WSVT_result_folder": wa.WSVT_RESULT_FOLDER,
            "WSVT_calculation_half_window": wa.WSVT_CALCULATION_HALF_WINDOW,
            "WSVT_n_scan": wa.WSVT_N_SCAN,
        }
    }

    # Here GUI specific ini

    calculation_type      = ini.get_int_from_ini(section="GUI", key="Calculation-Type", default=0)
    plot_rebinning_factor = ini.get_int_from_ini(section="GUI", key="Plot-Rebinning-Factor", default=4)
    use_flat              = ini.get_boolean_from_ini(section="WXST", key="Use-Flat",  default=False)
    use_dark              = ini.get_boolean_from_ini(section="WXST", key="Use-Dark",  default=False)

    return ScriptData(calculation_type=calculation_type,
                      plot_rebinning_factor=plot_rebinning_factor,
                      use_flat=use_flat,
                      use_dark=use_dark,
                      wavelets_analyzer_configuration=wavelets_analyzer_configuration)


def set_ini_from_initialization_parameters(initialization_parameters: ScriptData, ini: IniFacade):
    # -----------------------------------------------------
    # Wavefront Analyzer

    wavelets_analyzer_configuration = initialization_parameters.get_parameter("wavelets_analyzer_configuration")
    common_configuration            = wavelets_analyzer_configuration["common"]
    WXST_configuration              = wavelets_analyzer_configuration["WXST"]
    WSVT_configuration              = wavelets_analyzer_configuration["WSVT"]

    wa.ENERGY = common_configuration["energy"]
    wa.DISTANCE = common_configuration["distance"]
    wa.SCALING_V = common_configuration["scaling_v"]
    wa.SCALING_H = common_configuration["scaling_h"]
    wa.USE_GPU = common_configuration["use_gpu"]
    wa.USE_WAVELET = common_configuration["use_wavelet"]
    wa.WAVELET_CUT = common_configuration["wavelet_cut"]
    wa.PYRAMID_LEVEL = common_configuration["pyramid_level"]
    wa.N_ITERATIONS = common_configuration["n_iterations"]
    wa.TEMPLATE_SIZE = common_configuration["template_size"]
    wa.WINDOW_SEARCH = common_configuration["window_search"]
    wa.CROP = common_configuration["crop"]
    wa.DOWN_SAMPLING = common_configuration["down_sampling"]
    wa.REBINNING = common_configuration["rebinning"]
    wa.N_CORES = common_configuration["n_cores"]
    wa.N_GROUP = common_configuration["n_group"]
    wa.SAVE_IMAGES = common_configuration["save_images"]
    wa.VERBOSE = common_configuration["verbose"]

    wa.WXST_IMAGE_FILE_NAME = WXST_configuration["WXST_image_file_name"]
    wa.WXST_REFERENCE_FILE_NAME = WXST_configuration["WXST_reference_file_name"]
    wa.WXST_DARK_FILE_NAME = WXST_configuration["WXST_dark_file_name"] if initialization_parameters.get_parameter("use_dark") else "None"
    wa.WXST_FLAT_FILE_NAME = WXST_configuration["WXST_flat_file_name"] if initialization_parameters.get_parameter("use_flat") else "None"
    wa.WXST_RESULT_FOLDER = WXST_configuration["WXST_result_folder"]

    wa.WSVT_IMAGE_FOLDER = WSVT_configuration["WSVT_image_folder"]
    wa.WSVT_REFERENCE_FOLDER = WSVT_configuration["WSVT_reference_folder"]
    wa.WSVT_RESULT_FOLDER = WSVT_configuration["WSVT_result_folder"]
    wa.WSVT_CALCULATION_HALF_WINDOW = WSVT_configuration["WSVT_calculation_half_window"]
    wa.WSVT_N_SCAN = WSVT_configuration["WSVT_n_scan"]

    wa.store()

    # Here GUI specific ini

    ini.set_value_at_ini(section="GUI", key="Calculation-Type", value=initialization_parameters.get_parameter("calculation_type"))
    ini.set_value_at_ini(section="GUI", key="Plot-Rebinning-Factor", value=initialization_parameters.get_parameter("plot_rebinning_factor"))
    ini.set_value_at_ini(section="WXST", key="Use-Flat", value=initialization_parameters.get_parameter("use_flat"))
    ini.set_value_at_ini(section="WXST", key="Use-Dark", value=initialization_parameters.get_parameter("use_dark"))

    ini.push()