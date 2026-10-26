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
'''
    WSVT-based absolute phase measurement using simulated reference from mask distribution.
    The mask is scanned across multiple positions. The first image is used for pattern search
    to generate a simulated detector image (I_simu_whole). This simulated image is then shifted
    by the scan positions to create a synthetic reference stack. The WSVT solver from
    relative_metrology is used to process the measured image stack against the reference stack.

    2025
    by Xianbo Shi, based on process_images_executor.py by Zhi Qiao
'''

import os
import sys
import json
import copy
import shutil
import numpy as np
import scipy.ndimage as snd
import scipy.constants as sc

import threading
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib

from aps.wf_suite.common.arguments import Args
from aps.wf_suite.common.legacy.func import prColor, load_image, write_json, auto_crop, image_align, read_h5
from aps.wf_suite.common.legacy.gui_func import crop_gui
from aps.wf_suite.common.legacy.integration import frankotchellappa
from aps.wf_suite.relative_metrology.legacy.func import load_images
from aps.wf_suite.relative_metrology.legacy.WSVT import WSVT

from aps.wf_suite.absolute_phase.legacy.process_images_executor import (
    PatternSearch,
    ProcessImageResult,
    normalize,
    image_translation,
    get_local_curvature,
    do_recal_d_source,
    generate_reference_pattern,  # [MASK EXPOSURE MODEL] (ADDED)
)
from aps.wf_suite.absolute_phase.legacy.WXST_simplified import save_figure, save_figure_1D, save_data

lock = threading.Lock()


def _prep_for_xcorr(img, smooth_bg=0):
    """
    Clean an image so it can be safely used for phase cross-correlation.

    Flat/dark corrected images frequently contain NaN/inf (from division by
    (flat - dark) == 0), dead pixels (0) or hot pixels (spikes). These corrupt
    the FFT used by phase_cross_correlation and give meaningless shift estimates.

    This routine:
        1. replaces non-finite values with the median of the finite pixels,
        2. robustly clips dead/hot outliers using percentile bounds,
        3. optionally removes slow intensity variation (residual flat-field) to
           emphasize the speckle pattern,
        4. removes the mean.
    """
    out = np.asarray(img, dtype=np.float64)

    # 1. remove non-finite values
    finite = np.isfinite(out)
    if not finite.all():
        med = np.median(out[finite]) if finite.any() else 0.0
        out = np.where(finite, out, med)

    # 2. clip dead/hot pixels using robust percentile bounds
    lo, hi = np.percentile(out, [0.5, 99.5])
    if hi > lo:
        out = np.clip(out, lo, hi)

    # 3. high-pass to emphasize speckle (remove residual slow variation)
    if smooth_bg and smooth_bg > 1:
        size = int(smooth_bg)
        if size < min(out.shape):
            out = out - snd.uniform_filter(out, size=size)

    # 4. zero-mean
    out = out - np.mean(out)
    return out


def execute_process_images_WSVT(**arguments):
    """
    WSVT-based absolute phase measurement.

    Algorithm:
        1. Load scan positions from JSON file
        2. Load all measured images from image_directory (needed for pattern search)
        3. Generate simulated detector image (I_simu_whole) using first image + PatternSearch,
           which also determines/loads the source distance (d_sv, d_sh)
        4. Scale the scan positions to the detector plane using the geometric magnification
           (d_prop + d_source) / d_source, then auto-detect the sign convention from the
           flat/dark-corrected first two images (moved here so it uses corrected images and
           the projected scan shift)
        5. Create reference stack by shifting I_simu_whole by the projected scan positions,
           then crop both measured and reference stacks
        6. Run WSVT solver
        7. Save results

    Key difference from relative_metrology WSVT:
        - Reference images are SIMULATED from the mask pattern, not measured
        - Shifts are applied to the simulated image based on physical scan positions

    Key difference from single-shot absolute phase:
        - Multiple images at multiple scan positions instead of one
        - WSVT solver instead of WXST/SPINNet speckle tracking
    """

    # ===================== Default arguments =====================
    # Data paths
    arguments["image_directory"]       = arguments.get("image_directory")          # folder with measured images (TIF files)
    arguments["scan_positions_file"]   = arguments.get("scan_positions_file")      # JSON file with scan positions
    arguments["data_directory"]        = arguments.get("data_directory", '.')       # base data directory (contains absolute_phase/Au_delta.npy etc.)
    arguments["result_folder"]         = arguments.get("result_folder", './results')
    arguments["pattern_path"]          = arguments.get("pattern_path", './mask/RanMask5umB0.npy')
    arguments["propagated_pattern"]    = arguments.get("propagated_pattern", None)  # path to saved propagated pattern, or None to compute
    arguments["propagated_patternDet"] = arguments.get("propagated_patternDet", None)
    arguments["simulated_ref_stack"]   = arguments.get("simulated_ref_stack", None)   # path to saved reference stack npz, or None to compute
    arguments["process_after_mask"]    = arguments.get("process_after_mask", False)    # if True, continue processing after mask generation; if False, stop after saving simulated data
    arguments["saving_path"]           = arguments.get("saving_path", None)

    # Scan parameters
    arguments["n_scan"]                = arguments.get("n_scan", 51)               # number of scan positions to use
    arguments["sign_x"]                = arguments.get("sign_x", 1)                # sign convention for x: +1 or -1
    arguments["sign_y"]                = arguments.get("sign_y", 1)                # sign convention for y: +1 or -1
    arguments["auto_sign"]             = arguments.get("auto_sign", True)           # auto-detect sign_x/sign_y from first two images
    arguments["position_units"]        = arguments.get("position_units", "mm")     # units of positions in JSON: "mm" or "m"

    # Optics parameters
    arguments["p_x"]                   = arguments.get("p_x", 0.65e-6)
    arguments["det_res"]               = arguments.get("det_res", 1.5e-6)
    arguments["energy"]                = arguments.get("energy", 20e3)
    arguments["pattern_size"]          = arguments.get("pattern_size", 4.985e-6)
    arguments["pattern_thickness"]     = arguments.get("pattern_thickness", 1.5e-6)
    arguments["pattern_T"]             = arguments.get("pattern_T", 0.613)
    arguments["d_prop"]                = arguments.get("d_prop", 462e-3)
    arguments["d_source_v"]            = arguments.get("d_source_v", 60.0)
    arguments["d_source_h"]            = arguments.get("d_source_h", 60.0)
    arguments["source_v"]              = arguments.get("source_v", 10e-6)
    arguments["source_h"]              = arguments.get("source_h", 277e-6)
    arguments["propagator"]            = arguments.get("propagator", 'RS')

    # Pattern search parameters
    arguments["correct_scale"]         = arguments.get("correct_scale", False)
    arguments["show_alignFigure"]      = arguments.get("show_alignFigure", False)
    arguments["d_source_recal"]        = arguments.get("d_source_recal", False)   # recalculate source distance from pattern search
    arguments["estimation_method"]     = arguments.get("estimation_method", 'geometric')  # method for d_source recalculation: 'geometric' or 'simple_speckle'

    # [DETECTOR CALIBRATION] (ADDED) same mechanism as execute_process_image:
    # optional h5 file with 'dx'/'dy' datasets (full detector size) giving a
    # known systematic detector/WFS displacement error to subtract.
    arguments["cali_path"]             = arguments.get("cali_path", None)

    # [MASK EXPOSURE MODEL] (ADDED) mask fabrication over/under-exposure modeling.
    # Defaults keep the ORIGINAL behavior unchanged (model off). See
    # generate_reference_pattern in process_images_executor.
    arguments["exposure_model"]        = arguments.get("exposure_model", None)          # None/'off' (default), 'ctr', or 'distance'
    arguments["exposure_bias"]         = arguments.get("exposure_bias", None)           # None -> auto-estimate; float -> fixed edge shift (fraction of feature)
    arguments["exposure_corner_sigma"] = arguments.get("exposure_corner_sigma", 0.25)   # corner rounding radius (fraction of feature)
    arguments["exposure_supersample"]  = arguments.get("exposure_supersample", 8)       # fine-grid up-sampling factor
    arguments["exposure_estimate"]     = arguments.get("exposure_estimate", 'match_quality')  # 'match_quality' or 'occupation'
    arguments["exposure_bias_grid"]    = arguments.get("exposure_bias_grid", None)      # candidate biases for auto mode; None -> linspace(-0.25,0.25,5)
    arguments["img_transfer_matrix"]   = arguments.get("img_transfer_matrix", [1, 0, 0])
    arguments["find_transferMatrix"]   = arguments.get("find_transferMatrix", False)

    # Image processing
    arguments["crop"]                  = arguments.get("crop", [0, -1, 0, -1])
    arguments["dark"]                  = arguments.get("dark", None)
    arguments["flat"]                  = arguments.get("flat", None)
    arguments["rebinning"]             = arguments.get("rebinning", 1)
    arguments["lineWidth"]             = arguments.get("lineWidth", 5)  # line width for 1D profiles, in units of pattern_size

    # WSVT solver parameters
    arguments["cal_half_window"]       = arguments.get("cal_half_window", 20)
    arguments["n_cores"]               = arguments.get("n_cores", 4)
    arguments["n_group"]               = arguments.get("n_group", 1)
    arguments["use_wavelet"]           = arguments.get("use_wavelet", False)
    arguments["wavelet_lv_cut"]        = arguments.get("wavelet_lv_cut", 2)
    arguments["pyramid_level"]         = arguments.get("pyramid_level", 1)
    arguments["n_iter"]                = arguments.get("n_iter", 1)
    arguments["use_GPU"]               = arguments.get("use_GPU", False)
    arguments["scaling_x"]             = arguments.get("scaling_x", 1.0)
    arguments["scaling_y"]             = arguments.get("scaling_y", 1.0)

    # Output control
    arguments["verbose"]               = arguments.get("verbose", True)
    arguments["save_images"]           = arguments.get("save_images", True)

    args = Args(arguments)

    # Print all parameters
    for key, value in args.__dict__.items():
        prColor('{}: {}'.format(key, value), 'cyan')

    # ===================== Validate inputs =====================
    if args.image_directory is None:
        raise ValueError("image_directory must be specified")
    if args.scan_positions_file is None:
        raise ValueError("scan_positions_file must be specified")

    # ===================== Create result folder =====================
    result_folder = args.result_folder
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # ===================== Step 1: Load scan positions =====================
    prColor('Loading scan positions from: {}'.format(args.scan_positions_file), 'green')
    with open(args.scan_positions_file, 'r') as f:
        positions_data = json.load(f)

    x_positions = np.array(positions_data['position'][0])  # x shifts
    y_positions = np.array(positions_data['position'][1])  # y shifts

    # Convert positions to meters
    if args.position_units == "mm":
        x_positions_m = x_positions * 1e-3
        y_positions_m = y_positions * 1e-3
    elif args.position_units == "m":
        x_positions_m = x_positions
        y_positions_m = y_positions
    else:
        raise ValueError(f"Unknown position_units: {args.position_units}. Use 'mm' or 'm'.")

    n_available = len(x_positions)
    n_scan = min(args.n_scan, n_available)
    prColor('Using {} of {} available scan positions'.format(n_scan, n_available), 'green')

    # ===================== Step 2: Load all measured images =====================
    prColor('Loading measured images from: {}'.format(args.image_directory), 'green')
    img_stack_all = load_images(args.image_directory, '*.tif')
    prColor('Loaded {} images with shape {}'.format(img_stack_all.shape[0], img_stack_all.shape[1:]), 'green')

    if img_stack_all.shape[0] < n_scan:
        prColor('WARNING: Only {} images available, reducing n_scan from {}'.format(
            img_stack_all.shape[0], n_scan), 'red')
        n_scan = img_stack_all.shape[0]

    # Use first n_scan images
    img_stack = img_stack_all[:n_scan].astype(np.float32)

    # ===================== Sign convention (auto-detection deferred) =====================
    # NOTE (Modification 1): the auto sign detection is intentionally performed later,
    # inside the reference-stack generation block, AFTER:
    #   - the flat/dark correction is set up (so the sign is decided from corrected images), and
    #   - the source distance (d_sv, d_sh) is known (so the expected scan shift can be scaled
    #     to the detector plane by the geometric magnification, see Modification 2).
    sign_detection_info = {
        'auto_sign': bool(args.auto_sign),
        'sign_x': args.sign_x,
        'sign_y': args.sign_y,
    }

    # ===================== Step 3: Generate simulated detector image =====================
    # Use the first image for pattern search (same as single-shot)
    I_img_raw = img_stack[0].copy()

    # Set up simulation parameters (same structure as process_images_executor)
    para_simulation = {
        'data_directory': os.path.join(args.data_directory, "absolute_phase"),
        'p_x': args.p_x,
        'pattern_size': args.pattern_size,
        'pattern_T': args.pattern_T,
        'energy': args.energy,
        'pattern_thickness': args.pattern_thickness,
        'd_prop': args.d_prop,
        'd_sv': args.d_source_v,
        'd_sh': args.d_source_h,
        'sv': args.source_v,
        'sh': args.source_h,
        'det_res': args.det_res,
        'rebinning': args.rebinning,
        'propagator': args.propagator,
        'correct_scale': args.correct_scale,
        'showAlignFigure': args.show_alignFigure,
        'd_source_recal': args.d_source_recal,
        'estimation_method': args.estimation_method,
        'det_size': [int(I_img_raw.shape[0]), int(I_img_raw.shape[1])],
    }

    # Rebinning
    if args.rebinning > 1:
        prColor('Rebinning images with factor {}'.format(args.rebinning), 'red')
        from aps.wf_suite.absolute_phase.legacy.process_images_executor import rebin_2D
        new_stack = []
        for i in range(n_scan):
            _, _, rebinned = rebin_2D(None, None, img_stack[i], args.rebinning, exact=True)
            new_stack.append(rebinned)
        img_stack = np.array(new_stack, dtype=np.float32)
        I_img_raw = img_stack[0].copy()
        args.p_x *= args.rebinning
        para_simulation['det_size'] = [int(I_img_raw.shape[0]), int(I_img_raw.shape[1])]
        para_simulation['p_x'] = args.p_x

    prColor('Image shape: {}'.format(I_img_raw.shape), 'red')

    # Dark/flat handling
    if args.dark is None:
        dark = np.zeros(I_img_raw.shape)
    else:
        dark = load_image(args.dark).astype(np.float32)
        if args.rebinning > 1:
            _, _, dark = rebin_2D(None, None, dark, args.rebinning, exact=True)

    if args.flat is None:
        flat = snd.uniform_filter(I_img_raw, size=10 * (args.pattern_size / args.p_x))
    else:
        flat = load_image(args.flat).astype(np.float32)
        if args.rebinning > 1:
            _, _, flat = rebin_2D(None, None, flat, args.rebinning, exact=True)

    # ===================== Crop logic =====================
    if len(args.crop) == 4:
        if args.rebinning > 1:
            crop = args.crop
            if not args.crop[0] == 0:  crop[0] = int(args.crop[0] // args.rebinning)
            if not args.crop[1] == -1: crop[1] = int(args.crop[1] // args.rebinning)
            if not args.crop[2] == 0:  crop[2] = int(args.crop[2] // args.rebinning)
            if not args.crop[3] == -1: crop[3] = int(args.crop[3] // args.rebinning)
            args.crop = crop

        c1 = args.crop[1] if not args.crop[1] == -1 else I_img_raw.shape[0]
        c3 = args.crop[3] if not args.crop[3] == -1 else I_img_raw.shape[1]

        if (c1 - args.crop[0]) % 2 != 0: args.crop[1] = c1 - 1
        if (c3 - args.crop[2]) % 2 != 0: args.crop[3] = c3 - 1
    elif len(args.crop) == 1:
        if args.crop[0] == -1:
            args.crop = auto_crop(flat, shrink=0.85, to_int=True)
        elif args.crop[0] == 0:
            _, corner = crop_gui(I_img_raw)
            args.crop = [int(corner[0][0]), int(corner[1][0]),
                         int(corner[0][1]), int(corner[1][1])]
        else:
            if args.rebinning > 1:
                args.crop[0] = args.crop[0] // args.rebinning
            corner = [int(I_img_raw.shape[0] // 2 - args.crop[0] // 2),
                      int(I_img_raw.shape[0] // 2 + args.crop[0] // 2),
                      int(I_img_raw.shape[1] // 2 - args.crop[0] // 2),
                      int(I_img_raw.shape[1] // 2 + args.crop[0] // 2)]
            args.crop = corner
    else:
        raise ValueError('Wrong crop option. Use [0,-1,0,-1] for full, [-1] for auto, [N] for central, [y0,y1,x0,x1] for boundary')

    crop_edge = args.crop
    prColor('Crop region: {}'.format(crop_edge), 'green')

    # Save settings
    json_content = copy.deepcopy(args.__dict__)
    write_json(result_folder, 'setting', json_content)

    # ===================== Step 3-5: Generate or load simulated reference stack =====================
    # Determine whether we need to generate the simulated mask and reference stack
    # Check both if path is None AND if the file actually exists on disk
    # Cache policy: if the simulated files already exist on disk, reuse them and
    # just run the WSVT solver -- regardless of d_source_recal. Source-distance
    # recalculation only runs when the files must be generated from scratch
    # (i.e. one of them is missing). To force a recalculation, delete the cached
    # npz files so this generation path is taken again.
    generate_simulated_mask = (args.propagated_pattern is None or not os.path.exists(args.propagated_pattern) or
                               args.propagated_patternDet is None or not os.path.exists(args.propagated_patternDet))
    generate_ref_stack = args.simulated_ref_stack is None or not os.path.exists(args.simulated_ref_stack)

    saving_path = args.saving_path if args.saving_path is not None else args.image_directory
    if not os.path.exists(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    # Boundary crop function
    boundary_crop = lambda img: img[int(crop_edge[0]):int(crop_edge[1]),
                                    int(crop_edge[2]):int(crop_edge[3])]

    # [MASK EXPOSURE MODEL] (ADDED) function-level default so result.json save
    # below always has it, even when the reference stack is loaded from file.
    mask_exposure_info = None

    if generate_ref_stack:
        # --- Need to generate the reference stack from scratch ---

        # Normalize first image for pattern search
        I_img_raw_norm = (I_img_raw - dark) / (flat - dark)

        # Use central part for pattern search (same as single-shot)
        center_shift = [(crop_edge[0] + crop_edge[1]) // 2 - I_img_raw.shape[0] // 2,
                        (crop_edge[2] + crop_edge[3]) // 2 - I_img_raw.shape[1] // 2]

        image_transfer_matrix = None if args.find_transferMatrix else args.img_transfer_matrix

        # Build para_pattern dict needed by do_recal_d_source
        para_pattern = {
            'pattern_path': args.pattern_path,
            'propagated_pattern': args.propagated_pattern,
            'propagated_patternDet': args.propagated_patternDet,
            'saving_path': saving_path,
            # [MASK EXPOSURE MODEL] (ADDED) forward the exposure settings so the
            # source-distance recalculation uses the SAME model/bias as the main
            # matching path below (do_recal_d_source reads these via .get(...)).
            'exposure_model'        : args.exposure_model,
            'exposure_bias'         : args.exposure_bias,
            'exposure_corner_sigma' : args.exposure_corner_sigma,
            'exposure_supersample'  : args.exposure_supersample,
            'exposure_estimate'     : args.exposure_estimate,
            'exposure_bias_grid'    : args.exposure_bias_grid,
        }

        # Build a minimal para_XST dict (needed by do_recal_d_source signature, only used for simple_speckle method)
        para_XST = {
            'down_sampling': 1,
            'crop_boundary': [args.cal_half_window, args.cal_half_window],
            'method': 'simple',
            'template_size': args.cal_half_window,
            'window_searching': args.cal_half_window,
            'nCore': args.n_cores,
            'nGroup': args.n_group,
        }

        # to find the pattern from the reference image
        pattern_find = PatternSearch(ini_para=para_simulation)

        # -------------------------------- do the re-calculation of source distance -------------------------------------
        if args.d_source_recal and generate_simulated_mask:
            prColor('Re-calculate the source distance according to the current value', 'cyan')
            est_method = args.estimation_method
            d_source_recal_result = do_recal_d_source(I_img_raw_norm, boundary_crop(I_img_raw_norm),
                                                       para_pattern, pattern_find,
                                                       image_transfer_matrix, boundary_crop, crop_edge,
                                                       para_XST, para_simulation, result_folder,
                                                       method=est_method)

            prColor('use the recalculated source distance to re-generate the matched pattern', 'light_gray')

            para_simulation['d_sv_ini'] = para_simulation['d_sv']
            para_simulation['d_sh_ini'] = para_simulation['d_sh']

            para_simulation['d_sv'] = d_source_recal_result[0]
            para_simulation['d_sh'] = d_source_recal_result[1]
        else:
            para_simulation['d_sv_ini'] = para_simulation['d_sv']
            para_simulation['d_sh_ini'] = para_simulation['d_sh']

        print('change source distance to:', para_simulation['d_sv'], para_simulation['d_sh'])

        # to find the pattern from the reference image (re-create with potentially updated source distances)
        pattern_find = PatternSearch(ini_para=para_simulation)

        # Load or compute propagated pattern
        _propagated_pattern_exists = args.propagated_pattern is not None and os.path.exists(args.propagated_pattern)
        _propagated_patternDet_exists = args.propagated_patternDet is not None and os.path.exists(args.propagated_patternDet)

        if not _propagated_pattern_exists:
            prColor('Loading mask pattern: {}'.format(args.pattern_path), 'green')
            I_pattern = np.load(args.pattern_path).astype(np.float32)

            prColor('Propagating pattern to detector plane...', 'cyan')
            # [MASK EXPOSURE MODEL] (CHANGED) the original two lines below were:
            #     I_pattern = (1 - I_pattern)
            #     I_coh, I_det, I_prop = pattern_find.pattern_prop(I_pattern)
            # Routed through generate_reference_pattern: with the default
            # (exposure_model=None) it reproduces exactly that behavior, and
            # otherwise applies the fabrication over/under-exposure model.
            # I_pattern here is the raw mask (absorbing feature = 1, BEFORE inversion).
            I_coh, I_det, I_prop, mask_exposure_info = generate_reference_pattern(
                pattern_find, I_pattern, I_measured=I_img_raw_norm,
                exposure_model=args.exposure_model,
                exposure_bias=args.exposure_bias,
                corner_sigma=args.exposure_corner_sigma,
                supersample=args.exposure_supersample,
                estimate=args.exposure_estimate,
                bias_grid=args.exposure_bias_grid,
                img_transfer=image_transfer_matrix,
                result_folder=result_folder)

            # Modification 2: store the source distance used for the propagation so the scan
            # position projection magnification can be recovered when this file is reloaded.
            np.savez(os.path.join(saving_path, 'propagated_pattern.npz'),
                     I_coh=I_coh,
                     d_sv=para_simulation['d_sv'],
                     d_sh=para_simulation['d_sh'],
                     d_sv_ini=para_simulation['d_sv_ini'],
                     d_sh_ini=para_simulation['d_sh_ini'],
                     d_prop=para_simulation['d_prop'])
        elif not _propagated_patternDet_exists:
            prColor('Loading propagated pattern: {}'.format(args.propagated_pattern), 'green')
            data_content = np.load(args.propagated_pattern)
            I_coh = data_content['I_coh']

        # Generate simulated detector image
        if not _propagated_patternDet_exists:
            central_halfsize = 256
            center_crop = lambda img: img[
                max(0, min(img.shape[0], crop_edge[0] if (crop_edge[1] - crop_edge[0]) <= 2 * central_halfsize else int((crop_edge[0] + crop_edge[1]) / 2 - central_halfsize))):
                max(0, min(img.shape[0], crop_edge[1] if (crop_edge[1] - crop_edge[0]) <= 2 * central_halfsize else int((crop_edge[0] + crop_edge[1]) / 2 + central_halfsize))),
                max(0, min(img.shape[1], crop_edge[2] if (crop_edge[3] - crop_edge[2]) <= 2 * central_halfsize else int((crop_edge[2] + crop_edge[3]) / 2 - central_halfsize))):
                max(0, min(img.shape[1], crop_edge[3] if (crop_edge[3] - crop_edge[2]) <= 2 * central_halfsize else int((crop_edge[2] + crop_edge[3]) / 2 + central_halfsize)))
            ]
            I_img_central = center_crop(I_img_raw_norm)

            prColor('Pattern search on first image (center shape: {})...'.format(I_img_central.shape), 'cyan')

            if image_transfer_matrix is None:
                image_transfer_matrix = pattern_find.img_transfer_search(I_img_central, I_coh, result_folder)

            I_simu_whole, displace_x_offset, displace_y_offset = pattern_find.pattern_search(
                I_img_central, I_coh, image_transfer_matrix, center_shift)

            prColor('Saving simulated pattern (det plane)...', 'cyan')
            # Modification 2: store the (searched) source distance with the mask so the scan
            # position projection magnification stays consistent when this file is reloaded.
            np.savez(os.path.join(saving_path, 'propagated_patternDet.npz'),
                     I_simu_whole=I_simu_whole,
                     displace_x_offset=displace_x_offset,
                     displace_y_offset=displace_y_offset,
                     d_sv=para_simulation['d_sv'],
                     d_sh=para_simulation['d_sh'],
                     d_sv_ini=para_simulation['d_sv_ini'],
                     d_sh_ini=para_simulation['d_sh_ini'],
                     d_prop=para_simulation['d_prop'])
        else:
            prColor('Loading simulated pattern at detector plane: {}'.format(args.propagated_patternDet), 'green')
            data_content = np.load(args.propagated_patternDet)
            I_simu_whole = data_content['I_simu_whole']
            displace_x_offset = data_content['displace_x_offset']
            displace_y_offset = data_content['displace_y_offset']
            # Modification 2: recover the (searched) source distance stored with the mask so the
            # scan-position magnification is consistent with the loaded simulated reference.
            if 'd_sv' in data_content.files:
                para_simulation['d_sv'] = float(data_content['d_sv'])
                para_simulation['d_sh'] = float(data_content['d_sh'])
                if 'd_sv_ini' in data_content.files:
                    para_simulation['d_sv_ini'] = float(data_content['d_sv_ini'])
                    para_simulation['d_sh_ini'] = float(data_content['d_sh_ini'])
                else:
                    para_simulation['d_sv_ini'] = para_simulation.get('d_sv_ini', para_simulation['d_sv'])
                    para_simulation['d_sh_ini'] = para_simulation.get('d_sh_ini', para_simulation['d_sh'])
                prColor('  Loaded source distance from mask file: d_sv={}, d_sh={}'.format(
                    para_simulation['d_sv'], para_simulation['d_sh']), 'green')

        prColor('I_simu_whole shape: {}'.format(I_simu_whole.shape), 'green')

        # ===================== Step 4a (Modification 2): scan-position projection scaling =====================
        # The scan positions are recorded at the MASK plane. For a diverging (or converging)
        # beam, the pattern shadow on the DETECTOR plane is magnified geometrically: the mask
        # sits at distance d_sv/d_sh from the source and the detector at d_sv/d_sh + d_prop, so a
        # mask translation of delta projects to delta * (d_source + d_prop) / d_source on the
        # detector. d_sv/d_sh are the searched (or loaded) source distances from the pattern step.
        d_sv_used = para_simulation['d_sv']
        d_sh_used = para_simulation['d_sh']
        mag_v = (para_simulation['d_prop'] + d_sv_used) / d_sv_used
        mag_h = (para_simulation['d_prop'] + d_sh_used) / d_sh_used
        prColor('Scan-position projection magnification: mag_y={:.5f} (d_sv={}), mag_x={:.5f} (d_sh={})'.format(
            mag_v, d_sv_used, mag_h, d_sh_used), 'green')

        # ===================== Step 4b (Modification 1+3): auto-detect sign convention =====================
        # Performed here (not at load time) so it uses flat/dark-corrected images and the
        # projected (magnified) expected scan shift.
        if args.auto_sign and n_scan >= 2:
            prColor('Auto-detecting sign convention from first two (flat/dark-corrected) images...', 'cyan')
            from skimage.registration import phase_cross_correlation

            # Flat/dark-corrected, cropped first two frames, cleaned so that NaN/inf, dead or hot
            # pixels do not corrupt the FFT-based shift estimate (Modification 3).
            img0_corr = boundary_crop((img_stack[0] - dark) / (flat - dark))
            img1_corr = boundary_crop((img_stack[1] - dark) / (flat - dark))
            bg_size = max(3, int(round(10 * para_simulation['pattern_size'] / para_simulation['p_x'])))
            a0 = _prep_for_xcorr(img0_corr, smooth_bg=bg_size)
            a1 = _prep_for_xcorr(img1_corr, smooth_bg=bg_size)

            # Measure actual pixel shift between (cleaned) image 0 and image 1
            try:
                measured_shift, _, _ = phase_cross_correlation(
                    a0, a1, upsample_factor=10, normalization=None)
            except TypeError:
                measured_shift, _, _ = phase_cross_correlation(
                    a0, a1, upsample_factor=10)

            # measured_shift is [row_shift, col_shift] = [dy, dx] (image 1 relative to image 0)
            measured_dy = float(measured_shift[0])
            measured_dx = float(measured_shift[1])

            # Expected shift from scan positions, projected to the detector plane (in pixels)
            expected_dx_px = float((x_positions_m[1] - x_positions_m[0]) * mag_h / para_simulation['p_x'])
            expected_dy_px = float((y_positions_m[1] - y_positions_m[0]) * mag_v / para_simulation['p_x'])

            sign_detection_info['measured_dx_px'] = measured_dx
            sign_detection_info['measured_dy_px'] = measured_dy
            sign_detection_info['expected_dx_px'] = expected_dx_px
            sign_detection_info['expected_dy_px'] = expected_dy_px

            # Determine sign: if measured and expected have same sign (product > 0), sign = -1;
            # if opposite (product < 0), sign = +1.
            # Only update if the position change is significant (> 0.5 pixel)
            if abs(expected_dx_px) > 0.5 and abs(measured_dx) > 0.5:
                args.sign_x = -1 if (measured_dx * expected_dx_px > 0) else 1
                prColor('  Auto sign_x = {} (measured_dx={:.2f}px, expected_dx={:.2f}px)'.format(
                    args.sign_x, measured_dx, expected_dx_px), 'green')
            else:
                prColor('  X shift too small to determine sign (measured={:.2f}px, expected={:.2f}px), using sign_x={}'.format(
                    measured_dx, expected_dx_px, args.sign_x), 'yellow')

            if abs(expected_dy_px) > 0.5 and abs(measured_dy) > 0.5:
                args.sign_y = -1 if (measured_dy * expected_dy_px > 0) else 1
                prColor('  Auto sign_y = {} (measured_dy={:.2f}px, expected_dy={:.2f}px)'.format(
                    args.sign_y, measured_dy, expected_dy_px), 'green')
            else:
                prColor('  Y shift too small to determine sign (measured={:.2f}px, expected={:.2f}px), using sign_y={}'.format(
                    measured_dy, expected_dy_px, args.sign_y), 'yellow')

            sign_detection_info['sign_x'] = args.sign_x
            sign_detection_info['sign_y'] = args.sign_y
        elif args.auto_sign:
            prColor('Cannot auto-detect sign with fewer than 2 images, using manual sign_x={}, sign_y={}'.format(
                args.sign_x, args.sign_y), 'yellow')

        print('========================================')
        print('  sign_x = {}'.format(args.sign_x))
        print('  sign_y = {}'.format(args.sign_y))
        print('========================================')

        # --- Create reference stack by shifting I_simu_whole ---
        prColor('Creating reference stack by shifting simulated image...', 'cyan')

        # Crop the simulated whole image to the ROI
        I_simu_cropped = boundary_crop(I_simu_whole)
        prColor('Cropped simulated image shape: {}'.format(I_simu_cropped.shape), 'green')

        # Create reference stack: shift I_simu_whole by each scan position, then crop
        # Position 0 is the reference position (first image), so shifts are relative to position 0
        ref_stack = np.zeros((n_scan, I_simu_cropped.shape[0], I_simu_cropped.shape[1]), dtype=np.float32)

        for i in range(n_scan):
            # Relative shift from position 0, projected to the detector plane (Modification 2)
            dx_m = (x_positions_m[i] - x_positions_m[0]) * mag_h
            dy_m = (y_positions_m[i] - y_positions_m[0]) * mag_v

            # Convert to pixel shifts with sign convention
            # Note: scipy.ndimage.shift takes [row_shift, col_shift] = [dy, dx]
            dx_px = args.sign_x * dx_m / args.p_x
            dy_px = args.sign_y * dy_m / args.p_x

            if i == 0:
                # No shift for the first position
                ref_stack[i] = I_simu_cropped.copy()
            else:
                # Shift the full I_simu_whole first, then crop (more accurate at boundaries)
                I_shifted = snd.shift(I_simu_whole, [dy_px, dx_px], order=3, mode='nearest')
                ref_stack[i] = boundary_crop(I_shifted)

            if args.verbose and (i < 3 or i == n_scan - 1):
                prColor('  Scan {}: projected dx={:.4f}mm, dy={:.4f}mm -> dx_px={:.2f}, dy_px={:.2f}'.format(
                    i, dx_m * 1e3, dy_m * 1e3, dx_px, dy_px), 'cyan')

        # --- Crop and correct measured image stack ---
        prColor('Cropping measured image stack...', 'cyan')
        img_stack_cropped = np.zeros_like(ref_stack)
        for i in range(n_scan):
            img_corrected = (img_stack[i] - dark) / (flat - dark)
            img_stack_cropped[i] = boundary_crop(img_corrected)

        prColor('Cropped image stack shape: {}'.format(img_stack_cropped.shape), 'green')
        prColor('Reference stack shape: {}'.format(ref_stack.shape), 'green')

        # Align the first reference image to the first measured image (global alignment only once)
        # This corrects any bulk offset between simulation and measurement coordinate systems.
        prColor('Aligning reference to measured (first frame only)...', 'cyan')
        pos_shift, ref_aligned_0 = image_align(img_stack_cropped[0], ref_stack[0])
        prColor('  Global alignment shift = {}'.format(pos_shift), 'cyan')

        # Apply the same global shift to all reference frames
        from scipy.ndimage import fourier_shift
        for i in range(n_scan):
            if i == 0:
                ref_stack[i] = ref_aligned_0
            else:
                shifted = fourier_shift(np.fft.fftn(ref_stack[i]), pos_shift)
                ref_stack[i] = np.real(np.fft.ifftn(shifted))

        # NOTE: Do NOT per-image normalize or per-image align here.
        # The WSVT solver handles its own per-pixel normalization internally
        # (z-score across the stack dimension in pyramid_data()).
        # Per-image normalize()*255 would destroy the inter-frame intensity
        # variation that the solver relies on for displacement tracking.

        # Apply boundary crop to displacement offsets (matching WXST executor behavior)
        displace_x_offset = boundary_crop(displace_x_offset)
        displace_y_offset = boundary_crop(displace_y_offset)

        # Save the reference stack and all associated data for reuse
        prColor('Saving simulated reference stack to: {}'.format(saving_path), 'cyan')
        np.savez(os.path.join(saving_path, 'simulated_ref_stack.npz'),
                 ref_stack=ref_stack,
                 img_stack=img_stack_cropped,
                 displace_x_offset=displace_x_offset,
                 displace_y_offset=displace_y_offset,
                 crop=np.array(crop_edge),
                 n_scan=n_scan,
                 sign_x=args.sign_x,
                 sign_y=args.sign_y,
                 d_sv=para_simulation['d_sv'],
                 d_sh=para_simulation['d_sh'],
                 d_sv_ini=para_simulation['d_sv_ini'],
                 d_sh_ini=para_simulation['d_sh_ini'],
                 d_prop=para_simulation['d_prop'],
                 mag_x=mag_h,
                 mag_y=mag_v,
                 image_transfer_matrix=np.array(image_transfer_matrix))

        # Save reference info as JSON (include speckle_shift for back-propagation compatibility)
        write_json(result_path=saving_path,
                   file_name='reference',
                   data_dict={'image_transfer_matrix': image_transfer_matrix,
                              'speckle_shift': pos_shift.tolist(),
                              'n_scan': n_scan,
                              'sign_x': args.sign_x,
                              'sign_y': args.sign_y,
                              'crop': crop_edge})
        if saving_path != result_folder:
            shutil.copy(os.path.join(saving_path, 'reference.json'),
                        os.path.join(result_folder, 'reference.json'))

        # Check process_after_mask flag
        if generate_simulated_mask and not args.process_after_mask:
            prColor('Simulated mask and reference stack generated. process_after_mask=False, stopping here.', 'green')
            prColor('To continue processing, set process_after_mask=True or provide simulated_ref_stack path.', 'green')
            return None

    else:
        # --- Load pre-computed reference stack ---
        prColor('Loading simulated reference stack from: {}'.format(args.simulated_ref_stack), 'green')
        data_content = np.load(args.simulated_ref_stack)
        ref_stack = data_content['ref_stack']
        img_stack_cropped = data_content['img_stack']
        displace_x_offset = data_content['displace_x_offset']
        displace_y_offset = data_content['displace_y_offset']
        n_scan = min(n_scan, ref_stack.shape[0])
        ref_stack = ref_stack[:n_scan]
        img_stack_cropped = img_stack_cropped[:n_scan]

        # Recover the sign convention baked into the saved reference stack for reporting
        if 'sign_x' in data_content.files:
            args.sign_x = int(data_content['sign_x'])
            args.sign_y = int(data_content['sign_y'])
            sign_detection_info['sign_x'] = args.sign_x
            sign_detection_info['sign_y'] = args.sign_y
            sign_detection_info['auto_sign'] = False  # sign is fixed by the saved stack

        # [SOURCE-DISTANCE RECALCULATION FIX] (ADDED) Restore the source distances the
        # cached reference stack was generated with. This must happen before any scan-
        # position magnification is recomputed or reported so the loaded geometry stays
        # self-consistent. Older caches without these keys fall back to the current values.
        if 'd_sv' in data_content.files:
            para_simulation['d_sv'] = float(data_content['d_sv'])
            para_simulation['d_sh'] = float(data_content['d_sh'])
            para_simulation['d_sv_ini'] = (float(data_content['d_sv_ini'])
                                           if 'd_sv_ini' in data_content.files
                                           else para_simulation['d_sv'])
            para_simulation['d_sh_ini'] = (float(data_content['d_sh_ini'])
                                           if 'd_sh_ini' in data_content.files
                                           else para_simulation['d_sh'])
            prColor('  Loaded source distance from reference stack: d_sv={}, d_sh={}'.format(
                para_simulation['d_sv'], para_simulation['d_sh']), 'green')
        else:
            para_simulation['d_sv_ini'] = para_simulation.get('d_sv_ini', para_simulation['d_sv'])
            para_simulation['d_sh_ini'] = para_simulation.get('d_sh_ini', para_simulation['d_sh'])

        prColor('Loaded reference stack: {} images, shape {}'.format(
            ref_stack.shape[0], ref_stack.shape[1:]), 'green')

        # Copy reference.json from saving_path to result_folder if needed
        ref_json_src = os.path.join(saving_path, 'reference.json')
        if os.path.exists(ref_json_src) and saving_path != result_folder:
            shutil.copy(ref_json_src, os.path.join(result_folder, 'reference.json'))

    M_image = img_stack_cropped.shape[1]

    # ===================== Step 6: Run WSVT solver =====================
    prColor('Running WSVT solver with {} scan positions...'.format(n_scan), 'green')

    n_s_extend = 4  # hardcoded as in relative_metrology WSVT_executor

    WSVT_solver = WSVT(img_stack_cropped,
                       ref_stack,
                       M_image=M_image,
                       N_s_extend=n_s_extend,
                       cal_half_window=args.cal_half_window,
                       n_cores=args.n_cores,
                       n_group=args.n_group,
                       energy=args.energy,
                       p_x=args.p_x,
                       z=args.d_prop,
                       wavelet_level_cut=args.wavelet_lv_cut,
                       pyramid_level=args.pyramid_level,
                       n_iter=args.n_iter,
                       use_estimate=False,
                       use_wavelet=args.use_wavelet,
                       use_GPU=args.use_GPU,
                       scaling_x=args.scaling_x,
                       scaling_y=args.scaling_y,
                       crop=args.crop)

    # [SKIP RAW WSVT DUMP] (CHANGED) was: WSVT_solver.run(result_path=result_folder)
    # result_path only controls WSVT.run()'s own internal save block (writes
    # WSVT_result.hdf5/.json with the solver's raw, pre-executor-correction
    # displace/DPC/phase/transmission) -- it has no effect on the computation
    # itself (self.solver()/self.solver_cuda() run first, unconditionally).
    # This executor already saves its own final, corrected result set below
    # (single_shot_1.hdf5 via save_data), so the raw WSVT_result dump is
    # redundant here. Passing None skips it; relative_metrology's own
    # WSVT_executor.py calls .run(result_path=...) independently and is
    # unaffected by this change.
    WSVT_solver.run(result_path=None)

    # ===================== Step 7: Extract and post-process results =====================
    # Following the same post-processing as execute_process_image (area mode)
    displace = WSVT_solver.displace  # [displace_y, displace_x]
    displace_y = displace[0]
    displace_x = displace[1]

    prColor('WSVT processing complete.', 'green')

    # Add back the reference displacement offset (matching WXST executor behavior).
    # The WSVT solver returns the *residual* displacement between the measured and
    # simulated reference stacks. To get the absolute displacement, we must add back
    # the reference wavefront's displacement field (displace_x_offset, displace_y_offset)
    # computed from the spherical wavefront at the given source distance.
    # The WSVT solver crops its output by cal_half_window on each side, so we must
    # apply the same cropping to the offset arrays to match shapes.
    hw = args.cal_half_window
    displace_y_offset_cropped = displace_y_offset[hw:-hw, hw:-hw]
    displace_x_offset_cropped = displace_x_offset[hw:-hw, hw:-hw]
    displace_y_offset_cropped = displace_y_offset_cropped - np.mean(displace_y_offset_cropped)
    displace_x_offset_cropped = displace_x_offset_cropped - np.mean(displace_x_offset_cropped)
    displace_x += displace_x_offset_cropped
    displace_y += displace_y_offset_cropped

    # [DETECTOR CALIBRATION] (ADDED) subtract a known systematic detector/WFS
    # displacement error, same mechanism and file format as execute_process_image
    # in process_images_executor.py: an h5 file with 'dx'/'dy' datasets, full
    # detector size. Applied here (after combining residual + geometric offset)
    # rather than before, since WSVT has no offset-reuse across multiple
    # speckle_tracking calls (unlike single-shot's centralLine mode) -- both
    # placements are mathematically equivalent for WSVT's area-mode output.
    if args.cali_path is not None:
        prColor('load calibration data from file: {}'.format(args.cali_path), 'green')
        dx_cali = boundary_crop(read_h5(args.cali_path, 'dx'))[hw:-hw, hw:-hw]
        dy_cali = boundary_crop(read_h5(args.cali_path, 'dy'))[hw:-hw, hw:-hw]
    else:
        dx_cali = 0
        dy_cali = 0
    displace_x -= dx_cali
    displace_y -= dy_cali

    # Compute wavelength (same as PatternSearch.c_w)
    c_w = sc.value('inverse meter-electron volt relationship') / args.energy

    # Compute line profiles (same as original area mode)
    block_width = int(args.lineWidth * args.pattern_size / args.p_x) + 2 * args.cal_half_window

    line_displace_y = displace_y[:, int(displace_y.shape[1] // 2 - block_width // 2):int(displace_y.shape[1] // 2 - block_width // 2 + block_width)]
    line_displace_x = displace_x[int(displace_x.shape[0] // 2 - block_width // 2):int(displace_x.shape[0] // 2 - block_width // 2 + block_width), :]

    line_displace = [np.mean(line_displace_y, axis=1), np.mean(line_displace_x, axis=0)]
    line_displace = [line_displace[0] - np.mean(line_displace[0]), line_displace[1] - np.mean(line_displace[1])]

    line_curve = [np.gradient(line_displace[0]) / para_simulation['d_prop'],
                  np.gradient(line_displace[1]) / para_simulation['d_prop']]

    # Compute scaling factors from curvature (same as original)
    x_scaling = 1 / (1 + para_simulation['d_prop'] * np.mean(line_curve[1]))
    y_scaling = 1 / (1 + para_simulation['d_prop'] * np.mean(line_curve[0]))

    line_curve = [np.gradient(line_displace[0]) / para_simulation['d_prop'] * y_scaling,
                  np.gradient(line_displace[1]) / para_simulation['d_prop'] * x_scaling]

    # Compute DPC with scaling (same as original)
    DPC_y = displace_y * para_simulation['p_x'] / para_simulation['d_prop'] * y_scaling
    DPC_x = displace_x * para_simulation['p_x'] / para_simulation['d_prop'] * x_scaling

    avg_source_d_x = 1 / np.mean(line_curve[1])
    avg_source_d_y = 1 / np.mean(line_curve[0])

    # Compute phase using frankotchellappa (same as original)
    phase = frankotchellappa(DPC_x, DPC_y) * para_simulation['p_x'] * 2 * np.pi / c_w

    line_dpc = [line_displace[0] * para_simulation['p_x'] / para_simulation['d_prop'] * y_scaling,
                line_displace[1] * para_simulation['p_x'] / para_simulation['d_prop'] * x_scaling]
    line_phase = [np.cumsum(line_dpc[0]) * para_simulation['p_x'] * 2 * np.pi / c_w,
                  np.cumsum(line_dpc[1]) * para_simulation['p_x'] * 2 * np.pi / c_w]

    # Compute local curvature (same as original)
    curve_y, curve_x = get_local_curvature(displace_y * y_scaling, displace_x * x_scaling, para_simulation['d_prop'])

    avg_radius_y = 1 / np.mean(curve_y)
    avg_radius_x = 1 / np.mean(curve_x)

    prColor('mean radius of curvature: {}y    {}x'.format(avg_radius_y, avg_radius_x), 'cyan')

    # Compute intensity (flat cropped to phase size, same as original)
    flat_cropped = boundary_crop(flat)
    intensity = flat_cropped[(flat_cropped.shape[0] - phase.shape[0]) // 2: (flat_cropped.shape[0] + phase.shape[0]) // 2,
                             (flat_cropped.shape[1] - phase.shape[1]) // 2: (flat_cropped.shape[1] + phase.shape[1]) // 2]

    line_curve_filter = [snd.gaussian_filter(line_curve[0], 21), snd.gaussian_filter(line_curve[1], 21)]

    # ===================== Step 8: Save results (same format as original) =====================
    if args.save_images:
        with lock:
            matplotlib.use('Agg')

            # Filtered curvature line plot (same as original)
            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.plot(line_curve[0], 'k')
            plt.plot(line_curve_filter[0], 'r')
            plt.xlabel('[px]')
            plt.ylabel('[1/m]')
            plt.grid()
            plt.title('vertical curvature')
            plt.subplot(122)
            plt.plot(line_curve[1], 'k')
            plt.plot(line_curve_filter[1], 'r')
            plt.xlabel('[px]')
            plt.ylabel('[1/m]')
            plt.grid()
            plt.title('horizontal curvature')
            plt.savefig(os.path.join(result_folder, 'linecurve_filter.png'), dpi=150)
            plt.close()

    # Save result.json (same as original, plus sign detection info)
    result_dict = {'avg_source_d_x': float(avg_source_d_x),
                   'avg_source_d_y': float(avg_source_d_y),
                   'avg_radius_x':   float(avg_radius_x),
                   'avg_radius_y':   float(avg_radius_y),
                   'x_scaling':      float(x_scaling),
                   'y_scaling':      float(y_scaling),
                   # [SOURCE-DISTANCE RECALCULATION FIX] (ADDED) provenance of the
                   # requested (initial) and actually-used V/H source distances.
                   'source_distance_v_initial': float(para_simulation['d_sv_ini']),
                   'source_distance_h_initial': float(para_simulation['d_sh_ini']),
                   'source_distance_v_used':    float(para_simulation['d_sv']),
                   'source_distance_h_used':    float(para_simulation['d_sh']),
                   # [MASK EXPOSURE MODEL] (ADDED) chosen fabrication exposure model/bias
                   'exposure_model': (mask_exposure_info or {}).get('model'),
                   'exposure_bias':  (mask_exposure_info or {}).get('bias'),
                   'sign_detection': sign_detection_info}
    write_json(result_path=result_folder,
               file_name='result',
               data_dict=result_dict)

    if args.save_images:
        with lock:
            # Save 2D figures using save_figure (same as original, minus displace_fine which WSVT doesn't produce)
            save_figure(image_pair=[['displace_x', displace_x, '[px]'],
                                    ['displace_y', displace_y, '[px]'],
                                    ['curve_y', curve_y, '[1/m]'],
                                    ['curve_x', curve_x, '[1/m]'],
                                    ['phase', phase, '[rad]'],
                                    ['flat', flat_cropped, 'intensity']],
                        path=result_folder,
                        p_x=para_simulation['p_x'],
                        extention='.png')
            # Save 1D line profiles using save_figure_1D (same as original)
            save_figure_1D(image_pair=[['line_displace_x', line_displace[1], '[px]'],
                                       ['line_phase_x', line_phase[1], '[rad]'],
                                       ['line_displace_y', line_displace[0], '[px]'],
                                       ['line_phase_y', line_phase[0], '[rad]'],
                                       ['line_curve_y', line_curve_filter[0], '[1/m]'],
                                       ['line_curve_x', line_curve_filter[1], '[1/m]']],
                           path=result_folder, p_x=para_simulation['p_x'])

    # Save data as HDF5 using save_data (same as original)
    save_data(data={'intensity': intensity,
                    'displace_x': displace_x,
                    'displace_y': displace_y,
                    'phase': phase,
                    'line_phase_y': line_phase[0],
                    'line_displace_y': line_displace[0],
                    'line_curve_y': line_curve_filter[0],
                    'line_phase_x': line_phase[1],
                    'line_displace_x': line_displace[1],
                    'line_curve_x': line_curve_filter[1]},
              path_folder=result_folder,
              metadata={'p_x':        float(para_simulation['p_x']),
                        'x_scaling':  float(x_scaling),
                        'y_scaling':  float(y_scaling),
                        'energy':     float(args.energy),
                        'wavelength': float(c_w),
                        'd_prop':     float(para_simulation['d_prop'])})

    result = ProcessImageResult('area', intensity, phase, line_phase, line_displace, line_curve_filter)

    return result.to_dict()
