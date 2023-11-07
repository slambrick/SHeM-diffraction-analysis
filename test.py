# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:14:15 2023

@author: SamLambrick

Some example useage of the analysis and plotting code.
"""

import numpy as np
from shem_spot_profile import SpotProfile
import shem_spot_profile as ssp
import matplotlib.pyplot as plt


z_zero = -4.6e9
pure_he = SpotProfile.import_bshem(np.arange(20, 26), 'bshem_lif', np.array([10, 20, 40, 50, 70, 80]), 
                                   detector=1, z_zero=z_zero)
pure_he2 = SpotProfile.import_bshem(np.arange(20, 26), 'bshem_lif', np.array([10, 20, 40, 50, 70, 80]), 
                                    detector=3, z_zero=z_zero)
mixed_he = SpotProfile.import_bshem(np.arange(30, 40), 'bshem_lif', np.arange(0, 91, 10),
                                    detector=3, z_zero=z_zero, T = ssp.effective_T(23))
mixed_ar = SpotProfile.import_bshem(np.arange(30, 40), 'bshem_lif', np.arange(0, 91, 10),
                                    detector=1, z_zero=z_zero)
mixed_he2 = SpotProfile.import_bshem(np.arange(40, 50), 'bshem_lif', np.arange(0, 91, 10),
                                    detector=3, z_zero=z_zero, T = ssp.effective_T(23))
mixed_ar2 = SpotProfile.import_bshem(np.arange(40, 50), 'bshem_lif', np.arange(0, 91, 10),
                                    detector=1, z_zero=z_zero)

_, a1, _ = pure_he.shem_diffraction_plot()
a1.set_title('Pure helium 1')
_, a2, _ = pure_he2.shem_diffraction_plot()
a2.set_title('Pure helium 2')
_, a3, _ = mixed_he.shem_diffraction_plot()
a3.set_title('Mixed helium 1')
_, a4, _ = mixed_ar.shem_diffraction_plot()
a4.set_title('Mixed argon 1')
_, a5, _ = mixed_he2.shem_diffraction_plot()
a5.set_title('Mixed helium 2')
_, a6, _ = mixed_ar2.shem_diffraction_plot()
a6.set_title('Mixed argon 2')

mixed_he.shem_raw_plot()

mixed_he_sym = mixed_he.wrap_around(90)
mixed_he_sym.shem_diffraction_plot()

if False:
    # Test MoS2
    mos290 = SpotProfile.import_ashem(np.arange(2065, 2102), 'mos2_90deg', 
                                      np.arange(0, 91, 2.5), z_zero=0.88e6 + 0.3e6)
    
    #mos290.shem_diffraction_plot(DK_max=90, rasterized=False)
    
    mos2_full1 = mos290.wrap_around(60, crop="start")
    #mos2_full1.shem_diffraction_plot(DK_max=90, rasterized=False)
    mos2_full2 = mos290.wrap_around(60, crop=8)
    #mos2_full2.shem_diffraction_plot(DK_max=90, rasterized=False)
    
    f1, a1, _ = mos2_full2.interpolated_plot(N=201)
    
    
    #Fit a 2D gaussian to a peak at approximatly [-24.7, -30.6]
    popt = ssp.find_diff_peak(mos2_full2, -24.7, -30.6, plotit=True)
    
    # Identify peaks, this uses smoothing of the data through a Gaussian fiter so
    # that there is no noise, and thus peaks are simily local maxima
    kx_points, ky_points = mos2_full2.find_peaks(plotit=True)
    
    a1.plot(kx_points, ky_points, 'o', color='red', label="Identified")
    
    kx_fitted = []
    ky_fitted = []
    for i in range(len(kx_points)):
        popt = ssp.find_diff_peak(mos2_full2, kx_points[i], ky_points[i],
                                  interpolation_method = "linear")
        kx_fitted.append(popt[1])
        ky_fitted.append(popt[2])
    
    a1.plot(kx_fitted, ky_fitted, 'o', color='black', label="fitted")
    a1.legend()
