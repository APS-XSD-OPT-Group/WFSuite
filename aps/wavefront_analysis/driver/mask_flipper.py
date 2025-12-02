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
import time

from epics import PV
from collections import OrderedDict

from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance

APPLICATION_NAME = "MASK-FLIPPER"

register_ini_instance(IniMode.LOCAL_JSON_FILE,
                      ini_file_name=".mask_flipper.json",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)

MASK_ON_POSITION  = ini_file.get_int_from_ini(section="Flipper", key="Mask-On-Position", default=1)
MASK_OFF_POSITION = ini_file.get_int_from_ini(section="Flipper", key="Mask-Off-Position", default=6)

MASK_POSITION_WRITE  = ini_file.get_string_from_ini(section="Epics", key="Mask-Position-Write",  default="19idXEYE:FW102:POS")
MASK_POSITION_READ   = ini_file.get_string_from_ini(section="Epics", key="Mask-Position-Read",   default="19idXEYE:FW102:POS_RBV")
MASK_POSITIONS_WRITE = ini_file.get_string_from_ini(section="Epics", key="Mask-Positions-Write", default="19idXEYE:FW102:PCOUNT")
MASK_POSITIONS_READ  = ini_file.get_string_from_ini(section="Epics", key="Mask-Positions-Read",  default="19idXEYE:FW102:PCOUNT_RBV")

ini_file.set_value_at_ini(section="Flipper",   key="Mask-On-Position",  value=MASK_ON_POSITION)
ini_file.set_value_at_ini(section="Flipper",   key="Mask-Off-Position", value=MASK_OFF_POSITION)

ini_file.set_value_at_ini(section="Epics", key="Mask-Position-Write",  value=MASK_POSITION_WRITE )
ini_file.set_value_at_ini(section="Epics", key="Mask-Position-Read",   value=MASK_POSITION_READ  )
ini_file.set_value_at_ini(section="Epics", key="Mask-Positions-Write", value=MASK_POSITIONS_WRITE)
ini_file.set_value_at_ini(section="Epics", key="Mask-Positions-Read",  value=MASK_POSITIONS_READ )

ini_file.push()

class MaskFlipper():
    def __init__(self):
        self.__PV_dict = {
            "mask_position_write":  PV(MASK_POSITION_WRITE),
            "mask_position_read":   PV(MASK_POSITION_READ),
            "mask_positions_write": PV(MASK_POSITIONS_WRITE),
            "mask_positions_read":  PV(MASK_POSITIONS_READ),
        }

        positions_count = self.__PV_dict["mask_positions_read"].get()
        if MASK_ON_POSITION > positions_count:  raise ValueError("Mask ON position bigger than the total number of positions")
        if MASK_OFF_POSITION > positions_count: raise ValueError("Mask OFF position bigger than the total number of positions")


    # TODO: ADD timeout to avoid dead lock
    def set_mask_on(self):
        while self.__PV_dict["mask_positions_write"].get() != MASK_ON_POSITION:
            self.__PV_dict["mask_position_write"].put(MASK_ON_POSITION)
            time.sleep(0.1)

    # TODO: ADD timeout to avoid dead lock
    def set_mask_off(self):
        while self.__PV_dict["mask_positions_write"].get() != MASK_OFF_POSITION:
            self.__PV_dict["mask_position_write"].put(MASK_OFF_POSITION)
            time.sleep(0.1)
