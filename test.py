# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:14:15 2023

@author: SamLambrick
"""

import numpy as np
from shem_spot_profile import SpotProfile
import shem_spot_profile as ssp

b_lif = SpotProfile.import_bshem(np.arange(40, 50), 'bshem_lif', np.arange(0, 91, 10))