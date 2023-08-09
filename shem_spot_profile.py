#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 09:51:59 2022

@author: Sam Lambrick 2022-23  
@contributions: Ke Wang 2022-23

A module for importing, analysing, and plotting SHeM spot profile data.

The 2D Gaussian function is based on [this StackOverflow post](https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m).

A model for the Parallel speed ratio is used from [M Bergin 2017](http://doi.org/10.1016/j.ultramic.2019.112833).
and the data that was used to fit to Bergin's model is from [Toennies & Winkelman](http://doi.org/10.1063/1.434448).

SHeM (design for Cambridge A-SHeM):
    http://doi.org/10.1016/j.nimb.2014.06.028

2D diffraction with SHeM:
    http://doi.org/coming_soon!
"""

import numpy as np
from numpy import sqrt
from numpy import pi
import scipy.io
from scipy import interpolate as interp
import matplotlib.pyplot as plt
from matplotlib import cm # Colour palettes
#import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import pandas as pd
import os
import copy
import warnings

# Constants for the wavevector magnitude
m_u = 1.6605e-27 # kg
"""The atomic mass unit in kg."""
m_He = 4*m_u;    # kg
"""The mass of a helium-4 atom in kg."""
k_B = 1.380649e-23 # JK^-1
"""The Boltzmann constant in JK<sup>-1<\sup>"""
h = 6.62607015e-34 # Js
"""The Planck constant in Js."""

        
def energy_from_T(T):
    """Calculates incident helium energy in J from the beam temperature  - 
    assumes a pure beam.

    Parameters
    ----------
    T : float, int
        The beam temperature in K.

    Returns
    -------
    E: float
        The average energy of helium atoms in the beam, in J.
    """
    E = 5*k_B*T/2 # J
    return(E)


def effective_T(E):
    """Calculates the 'effective temperature' for the beam, that is the
    equivalent temperature for a pure helium beam with this energy. Accepts
    energy in meV.

    Parameters
    ----------
    E : float, int
        The average energy of helium atoms in a beam, in J.

    Returns
    -------
    T: float
        The temperature a pure helium beam would have to be at to give the
        atoms in the beam an energy of `E`. In units of K.
    """
    E = E*1.6e-19/1000
    T = 2*E/(5*k_B)
    return(T)


def speed_ratio(P_0, d_noz):
    """Calculates the parallel speed ratio for the specified nozzle
    pressure and diameter for a helium beam based on an emprical fit.
    
    The fit is from [M.Bergin 2018](http://doi.org/10.1016/j.ultramic.2019.112833)
    and is to data/simulation from [Toennies & Winkelman](http://doi.org/10.1063/1.434448).
    

    Parameters
    ----------
    P_0 : float, int
        Pressure of the beam, specified in torr.
    d_noz : float, int
        Diameter of the nozzle, specified in cm.

    Returns
    -------
    sR : float
        The predicted terminal, parallel speed ratio.
    """
    a = 0.43
    b = 0.76
    c = 0.84
    d = 5.2
    mu = 1.97
    lS = a*np.log10(P_0*d_noz) + b + c/(1 + np.exp(-d*(np.log10(P_0*d_noz) - mu)))
    sR = 10**lS
    return(sR)


def morse_V(z, x, y, params=(8.03, 1.35, 1.08, 0.102, 4)):
    """Calculates a corrugated Morse potential, from eq. 1.2 & 1.3 Celli et al.
    1995 https://doi.org/10.1063/1.449297.
    
    Inputs:
        z - height from surface
        x - lateral x position
        y - lateral y position
        params - 5 tuple, potential parameters (D0, alpha, alpha1, h, a)
    
    """
    D0, alpha, alpha1, h, a = params
    Q = h*(np.cos(2*pi*x/a) + np.cos(2*pi*y/a))
    v = D0*(np.exp(-2*alpha*(z - Q)) - 2*np.exp(-alpha1*z))
    return(v)


def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, f, g, h, k, l, m):
    '''A two dimensional gaussian function for fitting to diffraction peaks.
    
    Inputs:
        xy - tuple of x & y coordinates
        amplitude - height of the Gaussian
        x0 - x coordinate of the centre
        y0 - y coordinate of the centre
        sigma_x - x direction standard deviation
        sigma_y - y direction standard deviation
        theta - clockwise rotation (rad) of the Gaussian
        f,g,h,k,l,m - parameters for a 2nd order polynomial background
    '''
    
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    tot = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    tot = tot + f + g*x + h*y + k*x*y + l*x**2 + m*y**2
    return tot.ravel()


def background(xy, f, g, h, k, l, m):
    x, y = xy
    b = f + g*x + h*y + k*x*y + l*x**2 + m*y**2
    return(b)


def theta_of_z(z, plate="C06", WD=2):
    '''Calculates the detection angle theta from the z position in mm
    for the 1st gneration angular resolution pinhole plate for the Cambridge 
    A-SHeM. plate specification: #C06
    
    Can also calculate the detection angle from the z position in mm for the 
    mixed gas pinhole plate for the Cambridge B-SHeM. plate specification: #Z07
    
    Alternaticly: calculates the detection angle, theta, from the z position in mm for a
    45deg incidnece and 45deg detection pinhole plate with the specified design
    working distance WD
    '''
    if plate == "C06":
        return(np.arctan((7 - (z+1.5))/(z+1.5))*180/pi)
    elif plate == "standard":
        return(np.arctan((2*WD - z)/z)*180/pi)
    elif plate == "Z07":
        L = 6*np.tan(30*pi/180)
        return(np.arctan((L - z*np.tan(30*pi/180))/z)*180/pi) #TOOD
    else:
        raise ValueError('Unkwon pinhole plate, known: "C06", "Z07", "standard')


def load_z_scans(files, z_zero = 3.41e6, instrument = 'ashem'):
    """Loads a series of z scans taken at different aximulath orientations. The
    orientations need to provied seperatly as they are not stored in the data
    files. This file is written specifically for the Cambridge A-SHeM data 
    taken between 2020-23.
    

    Parameters
    ----------
    files : array_like of str
        The files that store the z scan data.
    z_zero : float, int, optional
        The "z zero": the value of z in stage coordinates that corresponds to 
        a distance of 0 from the pinhole plate. The default is 3.41e6.
    instrument : str, optional
        Which instrument? "ashem" or "bshem". The default is 'ashem'.

    Returns
    -------
    zs: numpy.ndarray
        The z values, in mm, for these z scans.
    meas: dict
    
    example_pos: numpy.ndarray 

    Note the contents of a Z-scan file for A-SHeM:  
        `z_positions`, `inputs`, `detector_mode`, `N_dwell`, `sampling_period`, `date_time`
        `det_params`, `p1`, `p2`, `p3`, `p_srce`, `p_diff`, `p_smpl`, `p_det`, `pause`
        `rotation_angle`, `counts`, `errors`, `finish_time`, `scan_time`
    """
   
    # Import the meas structure
    meas = scipy.io.loadmat(files[0])['meas']
    
    # Get the z positions (the same for all the data files
    # A-SHeM has positive z towards the pinhole plate
    # B-SHeM has positive z away from the pinhole plate
    if instrument == 'ashem':
        zs = (z_zero - meas['z_positions'][0][0]) 
    elif instrument == 'bshem': 
        zs = (meas['z_positions'][0][0] - z_zero) 
    else: 
        raise('Unkown instrument input in load_z_scans') 
    for j, item in enumerate(meas['inputs'][0][0][0]):
        if isinstance(item[0], str):
            if item[0] == 'example_pos':
                example_pos = meas['inputs'][0][0][0][j+1][0]
                break

    return(zs, meas, example_pos)


def load_z_scans_ashem(files_ind, path_name, z_zero = 3.41e6):
    ashem_fname = np.vectorize(lambda ind, p : '{}/Z{}.mat'.format(p, str(ind).zfill(6)))
    files = ashem_fname(files_ind, path_name)
    zs, meas, example_pos = load_z_scans(files, z_zero = z_zero, instrument = 'ashem')
    # Load in the data
    # Note the [0][0] needed to extract the data from the Matlab scruct
    data = np.zeros([len(files_ind), len(meas['counts'][0][0])])
    for i, f in enumerate(files):
        meas = scipy.io.loadmat(f)['meas']
        data[i,:] = meas['counts'][0][0].flatten()*1e9 # Convert to nA from A
    
    # A-SHeM uses nm
    return({'zs' : zs*1e-6, 'I' : data.transpose(), 'example_pos' : example_pos})


def load_z_scan_ashem_txt(fname, z_zero):
    """Reads in a single z scan for the A-SHeM where data has been saves in
    text files rather than in .mat files. Only loads a limited amount of
    metadata."""
    fid = open(fname)
    lines = fid.readlines()
    ind = 0
    example_position = np.zeros(3)
    for i, l in enumerate(lines):
        if l[0:15] == "Example position":
            l2 = l.rstrip().split(':')[1].split(',')
            for j in range(3):
                example_position[j] = float(l2[j])
        if l[0:3] == "Data":
            ind = i+1
    fid.close()
    df = pd.read_csv(fname, skiprows=ind)
    return(example_position, df['z(nm)'], df['I(A)'])


def load_z_scans_bshem(files_ind, path_name, z_zero=2.5e9, detector = 1):
    """Reads in z scans for the B-SHeM."""
    bshem_fname = np.vectorize(lambda ind, p : '{}/ZS{}.mat'.format(p, str(ind).zfill(6)))
    files = bshem_fname(files_ind, path_name)
    zs, meas, example_pos = load_z_scans(files, z_zero = z_zero, instrument = 'bshem')
    # Load in the data
    # Note the [0][0] needed to extract the data from the Matlab scruct
    data = np.zeros([len(files_ind), len(meas['counts'][0][0])])
    for i, f in enumerate(files):
        meas = scipy.io.loadmat(f)['meas']
        data[i,:] = meas['current_all'][0][0][:,detector].flatten()
    
    # B-SHeM uses pm
    return({'zs' : zs*1e-9, 'I' : data.transpose(), 'example_pos' : example_pos})


def load_z_scans_bshem_txt(files_ind, path_name, z_zero, detector = 1):
    """Reads in z scans for the A-SHeM where data has been saves in text files
    rather than in .mat files."""
    # TODO: read in ashem data from text files


def add_scale(ax, label, x_offset=0.08):
    '''A function to add an extra x scale to the polar plot with the provided
    axis handle. The offset determines where the scale is drawn.
    
    Inputs:
        ax - axis handle on which to add an extra scale
        label - text label for the new scale
        x_offset - how far to the left of the plot to add the new scale
    '''
    
    # add extra axes for the scale
    rect = ax.get_position()
    rect = (rect.xmin-x_offset, rect.ymin+rect.height/2, # x, y
            rect.width, rect.height/2) # width, height
    scale_ax = ax.figure.add_axes(rect)
    # hide most elements of the new axes
    for loc in ['right', 'top', 'bottom']:
        scale_ax.spines[loc].set_visible(False)
    scale_ax.tick_params(bottom=False, labelbottom=False)
    scale_ax.patch.set_visible(False) # hide white background
    # adjust the scale
    scale_ax.spines['left'].set_bounds(*ax.get_ylim())
    scale_ax.set_yticks(ax.get_yticks())
    scale_ax.set_ylim(ax.get_rorigin(), ax.get_rmax())
    scale_ax.set_ylabel(label)
    return(scale_ax)

def fit_single_peak():
    """Fits a 2D gaussian (+ background) to a diffraction peak."""
    return(0)


class SpotProfile:
    '''This class contains results on a spot profile SHeM measurement (currently
    loaded in from a series of z-scans) along with data analysis and plotting
    functions.'''
    
    def __init__(self, z, alpha_rotator, I, plate="C06", WD = 2):
        # TODO: example positions etc. not really being used yet!
        self.z = z                        # z position, matrix
        self.theta = theta_of_z(z, plate, WD) # Polar detection angle, matrix
        self.alpha_rotator = alpha_rotator# Rotator stage angle, matrix
        self.alpha = np.array([])         # azimuthal angle orientated with the crystal, matrix
        self.alpha_step = abs(alpha_rotator[0,0] - alpha_rotator[0, 1]) 
                    # Step in alpha, this is assumed to be constant across the scan
        self.signal = I                   # Signal levels, matrix
        self.kz = np.array([])            # wavevector z values
        self.DK = np.array([])            # parallel momentum transfer
        self.kx = np.array([])            # parallel momentum transfer projected into the x direction
        self.ky = np.array([])            # parallel momentum transfer projected into the y direction
        self.chosen_pos = np.array([])    # Example position used to create the profile
        self.example_alpha = np.nan       # Example alpha value used to create the profile (in stage coordinates)
        self.example_positions = np.array([])
                    # A list of all the example positions used for the z
                    # scans.
        self.incident_angle = 45       # The incidence angle in degrees, defaults to 45deg for A-SHeM
        self.alpha_zero = 0            # The value of alpha of one of the principle azimuths (for alignment)
        self.T = 300                   # Effective temperature in K (equivalent to pure beam)
        self.E = energy_from_T(self.T) # meV
    
    @classmethod
    def import_ashem(cls, file_ind, dpath, alphas, T=298, alpha_zero=0, z_zero = 3.41e6):
        '''Loads an experimental spot profile from a series of z scan data
        files.'''
        
        # Load the data
        data = load_z_scans_ashem(file_ind, dpath, z_zero = z_zero)
        
        # Number of rows (z values)
        # and columns (alpha values)
        r, c = data['I'].shape
        alphas = np.repeat(np.resize(alphas, [1, c]), r, axis=0)
        zs = np.repeat(data['zs'], c, axis=1)
        #alphas = np.repeat(np.resize(alphas, [1, c]), r, axis=0)
        sP = cls(z = zs, alpha_rotator = alphas, I = data['I'], plate = "C06")
        
        # The example image the spot was defined from was at 300deg
        sP.T = T
        sP.example_alpha = 300
        sP.set_alpha_zero(alpha_zero)
        sP.incident_angle = 45
        sP.calc_dK()
        return(sP)
    
    @classmethod
    def import_ashem_txt(cls, fname, dpath, alpha_zero=0):
        """Loads an expreimental spot profile from the text file data format."""
        # TODO: first load in from the meta 2d diffraction file the file names 
        # and the alpha values
        
        # TODO: Loop through each z scan file and load in the actual data
        #for f in file_ind:
        #    example_pos, z, I = load_z_scan_ashem_txt(dpath)
        
        # TODO: create the object
        #sP = cls(z = zs, alpha_rotator = alphas, I = data['I'], plate = "C06")

        # The example image the spot was defined from was at 300deg
        #sP.T = T
        #sP.example_alpha = 300
        #sP.set_alpha_zero(alpha_zero)
        #sP.calc_dK()
    
    
    @classmethod
    def import_bshem(cls, file_ind, dpath, alphas, T=298, alpha_zero=0, z_zero = 2.5e9, detector=1):
        # Load the data
        data = load_z_scans_bshem(file_ind, dpath, z_zero = z_zero, detector = detector)
        
        # Number of rows (z values)
        # and columns (alpha values)
        r, c = data['I'].shape
        alphas = np.repeat(np.resize(alphas, [1, c]), r, axis=0)
        zs = np.repeat(data['zs'], c, axis=1)
        #alphas = np.repeat(np.resize(alphas, [1, c]), r, axis=0)
        sP = cls(z = zs, alpha_rotator = alphas, I = data['I'], plate="Z07")
        
        # The example image the spot was defined from was at 300deg
        sP.T = T
        sP.example_alpha = 300
        sP.set_alpha_zero(alpha_zero)
        sP.incident_angle = 30
        sP.calc_dK()
        return(sP)

    @classmethod
    def import_ray(cls, data_dir, T=298, alpha_zero=0, plate="C06", WD=2):
        '''Import ray tracing simulation of a spot profile diffraction scan.
        Note that this is for a simualtion of the first generation angular
        resolution pinhole plate.'''
        
        # Dimulated data has each rotation stored in a single directory
        if data_dir[-1] != '/':
            data_dir = data_dir + '/'
        rot_dirs = os.listdir(data_dir)
        rot_dirs = [d for d in rot_dirs if not os.path.isfile(data_dir + d)]
        
        # Get the number of rotation angles
        alphas = [float(f[8:]) for f in rot_dirs]
        alphas.sort()
        n_alpha = len(alphas)
        
        # Get the number of z positions used
        mat_fname = data_dir + rot_dirs[0] + '/formatted/reconstructionSimulation.mat'
        zs = scipy.io.loadmat(mat_fname)['param']['sample_positions'][0][0][0]
        n_z = len(zs)
        I = np.zeros([n_z, n_alpha])
        for i, a in enumerate(alphas):
            mat_fname = data_dir + 'rotation{0:g}'.format(a) + '/formatted/reconstructionSimulation.mat'
            sim_data = scipy.io.loadmat(mat_fname)
            signals = sim_data['im']['single'][0][0][0] + sim_data['im']['multiple'][0][0][0]
            I[:,i] = signals
        
        if plate == "C06":
            # First A-SHeM diffraction pinholeplate has a recessed aperture, 
            # compensate for that
            zs = zs - 1.5;
        elif plate != "standard":
            ValueError('Unknon pinholeplate types, known: "C06" & "standard".')
   
        alphas = np.repeat(np.resize(np.array(alphas), [1, n_alpha]), n_z, axis=0)
        zs = np.repeat(np.resize(zs, [n_z, 1]), n_alpha, axis=1)
        sP = cls(z = zs, alpha_rotator = alphas, I = I, plate=plate, WD=WD)
        
        # The example image the spot was defined from was at 300deg
        sP.T = T
        sP.example_alpha = np.nan
        sP.set_alpha_zero(0)
        sP.incident_angle = 45
        sP.calc_dK()
        return(sP)
    
    @classmethod
    def import_previous(cls, fname):
        """Imports the diffraction data from a previously saved diffraction 
        data set - saved by save_to_text."""
        # TODO: write this
    
    def get_alpha_range(self):
        alphas = self.alpha[1,:]
        alpha_range = np.max(alphas) - np.min(alphas)
        return(alpha_range)
        
    def save_to_text(self, fname):
        """Saves the object to a text file for further analysis or saving for
        later without import from the raw z scans."""
        # TODO: write this
    
    def select_by_var(self, var, value):
        """Select a line scan of the data according to a specific value of one
        of the parameters, if the exact value is not found the closest value is
        used."""
        
        dif = np.abs(value - getattr(self, var))
        ind = np.where(dif == np.min(dif))
        chosen = getattr(self, var)[ind][0]
        if chosen != value:
            print('Using {} = {}deg.'.format(var, chosen))
        z = self.z[ind]
        theta = self.theta[ind]
        DK = self.DK[ind]
        I = self.signal[ind]
        alpha = self.alpha[ind]
        df =  pd.DataFrame({'z' : z, 'theta' : theta, 'DK' : DK, 'signal' : I, 'alpha': alpha})
        df.drop(var, axis=1, inplace=True)
        return((df, chosen))
    
    def wrap_around(self, symmetry, crop = "end"):
        """Wraps a data set around the full 360deg, i.e. to make a 360deg plot
        of an incomplete data set, e.g. a 90deg data set. Returns the data
        as a new object.
        Be aware that this method assumes that the step size in alpha is a
        factor of the symmetry factor, e.g. 1, 1.25, 2.5, 5, 7.5, 10deg 
        in 60deg or 90deg symmetry.
        """
        # TODO: this
        alpha_range = self.get_alpha_range()
        if (alpha_range < 60 and symmetry == 60) or (alpha_range < 90 and symmetry == 90):
            raise Exception("I don't think you have enough data to do this.")
        elif symmetry != 60 and symmetry != 90:
            warnings.warn("Untested symmetry, results unpredictable")
        
        # TODO: enable arbitrary cropping
        if crop == "end":
            crop = np.shape(self.alpha)[1] - int(symmetry/self.alpha_step) - 1
            ind = self.alpha_rotator <= symmetry
        elif crop == "start":
            crop = 0
            ind = self.alpha_rotator >= np.max(self.alpha_rotator) - symmetry
        elif isinstance(crop, (int, float)):
            # In this case the index of alpha to start at is chosen
            if crop < 0 or crop > int((np.max(self.alpha_rotator) - symmetry)/self.alpha_step):
                raise ValueError("Index to start at is not valid")
        else:
            raise ValueError("Can only crop data for wrapping around at the" +
                             " 'start', 'end' or an index to start at.")
        
        # Sectors of the plot that match to the part of the plot we will repeat
        stop_crop = crop + int(symmetry/self.alpha_step)+1
        alpha1 = self.alpha[:,crop:stop_crop]
        z1 = self.z[:,crop:stop_crop]
        I1 = self.signal[:,crop:stop_crop]
        n_rep = np.ceil(360/symmetry)
        
        alpha2 = alpha1[:,:-1]
        z2 = z1[:,:-1]
        I2 = I1[:,:-1]
        for i in range(1, int(n_rep)):
            if i < 10 + max(range(1, int(n_rep))):
                # Miss the first element on all but the last step
                alpha2 = np.concatenate([alpha2, alpha1[:,:-1] + i*(symmetry)], 1)
                z2 = np.concatenate([z2, z1[:,:-1]], 1)
                I2 = np.concatenate([I2, I1[:,:-1]], 1)
            else:
                alpha2 = np.concatenate([alpha2, alpha1 + i*(symmetry)], 1)
                z2 = np.concatenate([z2, z1], 1)
                I2 = np.concatenate([I2, I1], 1)
        wrapped = SpotProfile(z2, alpha2, I2)
        wrapped.T = self.T
        wrapped.set_alpha_zero(self.alpha_zero)
        wrapped.example_alpha = self.example_alpha
        wrapped.example_positions = self.example_positions.copy()
        wrapped.calc_dK()
        return(wrapped)
    
    def crop_alpha(self, alpha_min, alpha_max):
        """Crops the data according to a specified range of alpha values.
        Returns the data as a new object?"""
        # TODO: this

    def select_alpha(self, alpha):
        df, chosen_alpha = self.select_by_var('alpha', alpha)
        return((df, chosen_alpha))
    
    def select_theta(self, theta):
        df, chosen_theta = self.select_by_var('theta', theta)
        return((df, chosen_theta))
    
    def select_z(self, z):
        df, chosen_z = self.select_by_var('z', z)
        return((df, chosen_z))
    
    def select_DK(self, DK):
        df, chosen_DK = self.select_by_var('DK', DK)
        return((df, chosen_DK))
    
    def line_plot(self, xvar, var, value, figsize=[5, 3.5], logplot=False, ax=None, 
                  rect=[0.15,0.15,0.7,0.7], **kwargs):
        """Produce a plot of one line of the data set, selected for the
        specified value of the specified variable."""
        
        df, chosen = self.select_by_var(var, value)
        if ax == None:
            f = plt.figure(figsize=figsize)
            ax = f.add_axes(rect)
        else:
            f = ax.get_figure()
        if logplot:
            df['signal'] = np.log10(df['signal'])
        df.plot(x = xvar, y = 'signal', ax = ax, **kwargs)
        if logplot:
            ax.set_ylabel('$\\log_{10}(I/\\mathrm{nA})$')
        else:
            ax.set_ylabel('I/nA')
        ax.grid(True)
        return(f, ax, df, chosen)
     
    def line_plot_raw(self, alpha, figsize=[5, 3.5], logplot=False, ax=None, 
                      rect=[0.15,0.15,0.7,0.7], **kwargs):
        f, ax, df, chosen_alpha =self.line_plot('z', 'alpha', alpha, figsize=figsize, 
                                                logplot=logplot, ax=ax, rect=rect, **kwargs)
        ax.set_xlabel('$z/\\mathrm{mm}$')
        ax.set_title('1D plot of a zscan at $\\alpha$={}$^\\circ$'.format(chosen_alpha))
        return(f, ax, df)
    
    def line_plot_diffraction(self, alpha, figsize=[5, 3.5], logplot=False, ax=None, 
                              rect=[0.15,0.15,0.7,0.7], **kwargs):
        f, ax, df, chosen_alpha =self.line_plot('DK', 'alpha', alpha, figsize=figsize, 
                                                logplot=logplot, ax=ax, rect=rect, **kwargs)
        ax.set_xlabel('$\\Delta K/\\mathrm{nm}^{-1}$')
        ax.set_title('1D diffraction scan at $\\alpha$={}$^\\circ$'.format(chosen_alpha))
        return(f, ax, df)
    
    def line_plot_theta(self, alpha, figsize=[5, 3.5], logplot=False, ax=None, 
                        rect=[0.15,0.15,0.7,0.7], **kwargs):
        f, ax, df, chosen_alpha =self.line_plot('theta', 'alpha', alpha, figsize=figsize, 
                                                logplot=logplot, ax=ax, rect=rect, **kwargs)
        ax.set_xlabel('$\\theta_f/^\\circ$')
        ax.set_title('1D diffraction scan at $\\alpha$={}$^\\circ$'.format(chosen_alpha))
        return(f, ax, df)
    
    def line_plot_alpha(self, z, figsize=[5, 3.5], logplot=False, ax=None, 
                        rect=[0.15,0.15,0.7,0.7], **kwargs):
        f, ax, df, chosen_alpha =self.line_plot('alpha', 'z', z, figsize=figsize, 
                                                logplot=logplot, ax=ax, rect=rect, **kwargs)
        ax.set_xlabel('$\\alpha/^\\circ$')
        ax.set_title('$\\alpha$ scan at z={}mm'.format(z))
        return(f, ax, df)
    
    def calc_dK(self):
        """Calculates the in plane momentum transfer for the data file and the
        projected in plane momentum transfer (psuedo) k<sub>x</sub>, 
        k<sub>y</sub>. Values are calculated in nm<sup>-1</sup>.
        

        Returns
        -------
        None.

        """
        K = 2*pi*sqrt(2*m_He*self.E)/h # m^-1
        #K = k*np.sin(incident_angle*pi/180) 
        #K = 2*pi*sqrt(5*m_He*k_B*self.T)/h; # m^-1
        # Calculates the parallel momentum transfer in nm^-1
        self.DK = K*(np.sin(self.theta*pi/180) - np.sin(self.incident_angle*pi/180) )/1e9;
        # Calculate the projected k values
        self.kx = -K*( (np.sin(self.theta*pi/180) - np.sin(self.incident_angle*pi/180) )*np.cos(self.alpha*pi/180) )/1e9;
        self.ky = -K*(np.sin(self.theta*pi/180) - np.sin(self.incident_angle*pi/180) )*np.sin(self.alpha*pi/180)/1e9;
    

    def set_alpha_zero(self, alpha_zero=0):
        """Sets the correct 0 for the azimuthal angle such that 0 lies along
        one of the principle azimuths. 
        
        You will probably want to plot the data first to identify a principle
        azimuth, then you can set the `alpha_zero`.
        

        Parameters
        ----------
        alpha_zero : int, float, optional
            The value of alpha that is to be set to be 0. The default is 0.

        Returns
        -------
        None.

        """
        self.alpha = self.alpha_rotator - alpha_zero
        self.alpha_zero = alpha_zero
        
    
    def shem_cartesian_plot(self, var, colourmap = cm.viridis,
                            figsize = [6,4], rasterized = True):
        """Plots the 2D diffraction pattern on Cartesian coordinates with alpha
        on the x axis and the specified variable on the y axis.
        

        Parameters
        ----------
        var : str
            The name of the variable to plot on the y-axis, e.g. .
        colourmap : matplotlib.colors.ListedColormap, optional
            The colormap to use in the plot. The default is cm.viridis.
        figsize : array_like, optional
            Figure size in inches. The default is [6,4].
        rasterized : bool, optional
            Should rasterized be used when creating the plot, False will
            increase the resource use. The default is True.

        Returns
        -------
        fig: matplotlib.figure.Figure
            Matplotlib figure object for the plot
        ax1: matplotlib.axes._axes.Axes
            Matplotlib axis object for the plot
        mesh1: matplotlib.collections.QuadMesh
            Matplotlib mesh object returned from `pcolorplot`.
        """
        
        fig, ax1 = plt.subplots()
        fig.set_size_inches(figsize[0], figsize[1])
        Z = np.log10(self.signal)
        mesh1 = ax1.pcolormesh(self.alpha, getattr(self, var), Z,
                               edgecolors='face', cmap = colourmap, rasterized=rasterized)
        ax1.set_xlabel('$\\alpha/^\\circ$')
        ax1.set_ylabel(var)
        ax1.grid(alpha=0.33)
        cbar = fig.colorbar(mesh1, label="$\\log_{10}(I/\\mathrm{nA})$",)
        return(fig, ax1, mesh1)
        
        
    def shem_polar_plot(self, var, colourmap = cm.viridis, bar_location = 'right', 
                        figsize = [8,6], rasterized = True, DK_invert=True, logplot=True):
        if var == 'DK':
            # If we plot data with -DK we invert it
            if DK_invert:
                sP = self.filter_by_var('z', 2, 'above')
                sP.DK = -sP.DK
            else:
                sP = self.filter_by_var('z', 2, 'below')
        else:
            sP = self
        fig = plt.figure(figsize=figsize)
        # The colourbar can go either on the left or the right
        if bar_location == 'right':
            gs = fig.add_gridspec(1,2,width_ratios=[10,0.5], wspace=0, left=0.11, right=0.88, top=0.9, bottom=0.14)
            ax1=plt.subplot(gs[0], projection="polar", aspect=1.)
            ax2 = plt.subplot(gs[1])
        elif bar_location == 'bottom':
            # The subplot size on the grid is determining the width of the colorbar
            gs = fig.add_gridspec(2,3,height_ratios=[10,0.5], width_ratios=[1,4,1])
            ax1=plt.subplot(gs[0:3], projection="polar", aspect=1.)
            ax2 = plt.subplot(gs[4])
        else:
            raise ValueError('Unknown colorbar location.')
        # The main plot, mask any nan (e.g. values that have been masked out)
        Z = np.log10(sP.signal) if logplot else sP.signal
        mesh1 = ax1.pcolormesh(sP.alpha*pi/180, getattr(sP, var), Z, 
                               edgecolors='face', cmap = colourmap, rasterized=rasterized)
        # Thicker axis lines
        ax1.spines[:].set_linewidth(1.5)
        ax1.set_xlabel('$\\alpha$')
        ax1.grid(alpha=0.33)
        ax1.tick_params(axis='y', colors=[0.9,0.9,0.9])
        label_str = "$\\log_{10}(I/\\mathrm{nA})$" if logplot else "$I/\\mathrm{nA}$"
        if bar_location == 'right':
            fig.colorbar(mesh1, cax=ax2, label=label_str, shrink=0.1)
        elif bar_location == 'bottom':
            fig.colorbar(mesh1, cax=ax2, label=label_str, orientation='horizontal')
            plt.subplots_adjust(hspace=0.3)
        return(fig, ax1, ax2)
    
    def shem_diffraction_plot(self, colourmap = cm.viridis, bar_location='right',
                              figsize=[8,6], rasterized=True, DK_invert=True, 
                              x_offset=0.08, DK_max=85, logplot=True):
        fig, ax1, ax2 = self.shem_polar_plot('DK', colourmap=colourmap, bar_location=bar_location, 
                                             figsize=figsize, rasterized=rasterized, DK_invert=DK_invert,
                                             logplot=logplot)
        ax1.set_yticks([0, 25, 50, 75])
        ax1.set_ylim(0, DK_max)
        ax1.tick_params(axis='y', colors=[0.9,0.9,0.9])
        add_scale(ax1, label = '$\Delta K/\\mathrm{nm}^{-1}$', x_offset = x_offset)
        return(fig, ax1, ax2)
    
    def shem_raw_plot(self, colourmap = cm.viridis,  bar_location='right',
                      figsize=[8,6], rasterized=True, x_offset = 0.08,
                      logplot=True, z_max=7):
        fig, ax1, ax2 = self.shem_polar_plot('z', colourmap=colourmap, bar_location=bar_location, 
                                             figsize=figsize, rasterized=rasterized, DK_invert=True,
                                             logplot=logplot)
        ax1.set_yticks([0, 1, 2, 3, 4, 5, 6])
        ax1.set_ylim(0, z_max)
        ax1.tick_params(axis='y', colors=[0.9,0.9,0.9])
        add_scale(ax1, label = 'z/mm', x_offset = x_offset)
        return(fig, ax1, ax2)
    
    def filter_by_var(self, var, value, direction):
        """Filters the data above or below the specified value for the
        specified variable. direction may be 'above' or 'below'."""
        if direction == 'above':
            ind = getattr(self, var) < value
        elif direction == 'below':
            ind = getattr(self, var) > value
        else:
            raise ValueError('Do not understand input')
        sP = copy.deepcopy(self)
        sP.signal[ind] = np.nan#
        return(sP)
    
    def grid_interpolate(self, kx, ky, N=101, method='nearest'):
        '''Produces a 2D cartesian grid of the data for the provided kx and ky
        vectors.'''
        kx = np.linspace(kx[0], kx[1], N)
        ky = np.linspace(ky[0], ky[1], N)
        kxx, kyy = np.meshgrid(kx, ky)
        z = self.signal.ravel()
        ind = ~np.isnan(z)
        x=self.kx.ravel()              #Flat input into 1d vector
        x=x[ind]   #eliminate any NaN
        y=self.ky.ravel()
        y=y[ind]
        z=z[ind]
        I = interp.griddata((x, y), z, (kxx, kyy), method=method)
        return(I, kxx, kyy)
    
    def interpolated_plot(self, kx=(-80, 80), ky=(-80,80), N=101, method='nearest', 
                          ax=None, limiting_circle=True, figsize=(8,8)):
        '''Produce an interpolated plot of the data with N points along the
        kx and ky axes. Useful for checking the output of interpolation'''
        I, kxx, kyy = self.grid_interpolate(kx, ky, N, method=method)
        if ax == None:
            f = plt.figure(figsize=figsize)
            ax = f.add_axes([0.15, 0.1, 0.7, 0.9], aspect='equal')
        else:
            f = ax.get_figure()
        if limiting_circle:
            patch = patches.Circle((0, 0), radius = max(kx), transform = ax.transData)
        im = ax.pcolormesh(kxx, kyy, np.log10(I), edgecolors='face', rasterized=True)
        if limiting_circle:
            im.set_clip_path(patch)
        #ax.axis('equal')
        ax.set_xlabel('$k_x/\mathrm{nm}^{-1}$')
        ax.set_ylabel('$k_y/\mathrm{nm}^{-1}$')
        ax.set_title('Interpolated k-plot, method = '+ method)
        return(f, ax, im)
    
    def shift_centre(self, D_kx, D_ky, T = 293):
        '''Translates the diffraction pattern by the specified amount in kx and
        ky. Note: this does not yet fully propogate the change through all the
        variables.'''
        K = 2*pi*sqrt(5*m_He*k_B*self.T)/h; # m^-1
        K = K/1e9
        a = 1.5 #mm
        b = 3.5 #mm
        cP = copy.deepcopy(self)
        
        # Shifted kx,ky
        cP.kx = cP.kx + D_kx
        cP.ky = cP.ky + D_ky
        #cP.DK = sqrt(cP.kx**2 + cP.ky**2)
        #cP.theta = np.arcsin(cP.DK/K + 1/sqrt(2))
        #cP.z = 2*b/(np.tan(cP.theta*pi/180) + 1) - a
        #TODO: calculate new alpha....
        cP.alpha = np.arctan2(cP.ky, cP.kx)*180/pi + 180
        cP.DK = cP.ky/np.sin(cP.alpha*pi/180)
        cP.alpha = cP.alpha - 180
        return(cP)
    
    def identify_peaks(self, interpolation_method="nearest"):
        """Identifies initial guesses of the diffraction peak locations in the
        data set. By default plots these on an interpolated plot."""
        # TODO: do this, perhaps using 
        I, kx, ky = self.grid_interpolate((-80, 80), (-80,80), N=101)
        return(0)
