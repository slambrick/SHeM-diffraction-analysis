# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:14:15 2023

@author: SamLambrick

Some example useage of the analysis and plotting code.
"""

import numpy as np
from shem_spot_profile import SpotProfile
import shem_spot_profile as ssp

#b_lif = SpotProfile.import_bshem(np.arange(40, 50), 'bshem_lif', np.arange(0, 91, 10))

mos290 = SpotProfile.import_ashem(np.arange(2065, 2102), 'mos2_90deg', 
                                  np.arange(0, 91, 2.5), z_zero=0.88e6 + 0.3e6)

mos290.shem_diffraction_plot(DK_max=90, rasterized=False)

mos2_full1 = mos290.wrap_around(60, crop="start")
#mos2_full1.shem_diffraction_plot(DK_max=90, rasterized=False)
mos2_full2 = mos290.wrap_around(60, crop=8)
mos2_full2.shem_diffraction_plot(DK_max=90, rasterized=False)
