# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:23:57 2023

@author: SamLambrick
"""


import numpy as np
from shem_spot_profile import SpotProfile
import shem_spot_profile as ssp
import matplotlib.pyplot as plt


z_zero = -4.6e6

mos2_mono = SpotProfile.import_ashem(np.arange(2173, 2191+1), '2023_08_MoS2_5um_hbn_sub', np.arange(0, 91, 5), 
                                     z_zero=1.05e6)
mos2_full = mos2_mono.wrap_around(60, crop=1)
mos2_mono.shem_raw_plot()
mos2_full.shem_raw_plot()

mos2_full.shem_diffraction_plot()

m3 = mos2_full.filter_by_var('DK', 0, 'below')
m3.interpolated_plot(N=201, method='cubic')
