#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 09:51:59 2022

@author: Sam Lambrick 2022-23
@contributions: Ke Wang 2022-23

A module for importing, analysing, and plotting SHeM spot profile data.

The 2D Gaussian function is based on this StackOverflow post:
    https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m

A model for the Parallel speed ratio is used from M Bergin 2017:
    http://doi.org/10.1016/j.ultramic.2019.112833

Data that was used to fit to Bergin's model is from Toennies & Winkelman :
    http://doi.org/10.1063/1.434448
"""

import numpy as np
from numpy import pi
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm # Colour palettes
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import os
from scipy import interpolate as interp
import pandas as pd
import copy

# Constants for the wavevector magnitude
m_u = 1.6605e-27 # kg
m_He = 4*m_u;    # kg
k_B = 1.380649e-23 # JK^-1
h = 6.62607015e-34 # Js
        

def speed_ratio(P_0, d_noz):
    """Calculates the parallel speed ratio, S, for the specified nozzle
    pressure and diameter for a helium beam based on an emprical fit.
    
    The fit is from M.Bergin 2018: doi.org/10.1016/j.ultramic.2019.112833
    and is to data/simulation from Toennies & Winkelman doi.org/10.1063/1.434448.
    
    Inputs:
        P_0 - nozzle pressure in torr
        d_noz - noxxle diameter in cm
    """
    a = 0.43
    b = 0.76
    c = 0.84
    d = 5.2
    mu = 1.97
    lS = a*np.log10(P_0*d_noz) + b + c/(1 + np.exp(-d*(np.log10(P_0*d_noz) - mu)))
    return(10**(lS))

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
    
    Alternaticly: calculates the detection angle, theta, from the z position in mm for a
    45deg incidnece and 45deg detection pinhole plate with the specified design
    working distance WD
    '''
    if plate == "C06":
        return(np.arctan((7 - (z+1.5))/(z+1.5))*180/np.pi)
    elif plate == "standard":
        return(np.arctan((2*WD - z)/z)*180/np.pi)
    else:
        raise ValueError('Unkwon pinhole plate, known: "C06" and "standard')


def load_z_scans(files_ind, path_name, z_zero = 3.41e6):
    '''Loads a series of z scans taken at different aximulath orientations. The
    orientations need to provied seperatly as they are not stored in the data
    files. This file is written specifically for the Cambridge A-SHeM data 
    taken between 2020-23.
    
    The value of the z zero position should be given in um.
    
    Inputs:
        files_ind - list-like of file index number of the Z scans
        path_name - path to the data files
        z_sero - the z=0 value (in stage coordinates, so nm for A-SHeM) of the
                 z stage. The default value is for A-SHeM measurements of LiF
                 taken in the Autumn of 2020.
    
    Note the contents of a Z-scan file for A-SHeM:
        z_positions, inputs, detector_mode, N_dwell, sampling_period, date_time
        det_params, p1, p2, p3, p_srce, p_diff, p_smpl, p_det, pause
        rotation_angle, counts, errors, finish_time, scan_time
    '''
   
    # Import the meas structure
    fname = '{}/Z{}.mat'.format(path_name,str(files_ind[0]).zfill(6))
    meas = scipy.io.loadmat(fname)['meas']

    # Note the [0][0] needed to extract the data from the Matlab scruct
    data = np.zeros([len(files_ind), len(meas['counts'][0][0])])
    
    # Get the z positions (the same for all the data files)
    zs = (z_zero - meas['z_positions'][0][0])*1e-6
    for j, item in enumerate(meas['inputs'][0][0][0]):
        if isinstance(item[0], str):
            if item[0] == 'example_pos':
                example_pos = meas['inputs'][0][0][0][j+1][0]
                break

    # Load in the signals and the example position
    for i, ind in enumerate(files_ind):
        fname = '{}/Z{}.mat'.format(path_name,str(ind).zfill(6))
        meas = scipy.io.loadmat(fname)['meas']
        data[i,:] = meas['counts'][0][0].flatten()*1e9 # Convert to nA from A
    
    return({'zs' : zs, 'I' : data.transpose(), 'example_pos' : example_pos})


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
        self.alpha_zero = 0
        self.T = 300
    
    @classmethod
    def import_ashem(cls, file_ind, dpath, alphas, T=298, alpha_zero=0):
        '''Loads an experimental spot profile from a series of z scan data
        files.'''
        
        # Load the data
        data = load_z_scans(file_ind, dpath)
        
        # Number of rows (z values)
        # and columns (alpha values)
        r, c = data['I'].shape
        alphas = np.repeat(np.resize(alphas, [1, c]), r, axis=0)
        zs = np.repeat(data['zs'], c, axis=1)
        #alphas = np.repeat(np.resize(alphas, [1, c]), r, axis=0)
        sP = cls(z = zs, alpha_rotator = alphas, I = data['I'])
        
        # The example image the spot was defined from was at 300deg
        sP.T = T
        sP.example_alpha = 300
        sP.set_alpha_zero(alpha_zero)
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
        zs = scipy.io.loadmat(data_dir + rot_dirs[0] + '/formatted/reconstructionSimulation.mat')['param']['sample_positions'][0][0][0]
        n_z = len(zs)
        I = np.zeros([n_z, n_alpha])
        for i, a in enumerate(alphas):
            sim_data = scipy.io.loadmat(data_dir + 'rotation{0:g}'.format(a) + '/formatted/reconstructionSimulation.mat')
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
        sP.calc_dK()
        return(sP)
    
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
        '''Calculates the momentum transer for the data file and the
        "psuedo" kx, ky that we have defined.'''
        K = 2*np.pi*np.sqrt(5*m_He*k_B*self.T)/h; # m^-1
        # Calculates the parallel momentum transfer in nm^-1
        self.DK = K*(np.sin(self.theta*np.pi/180) - 1/np.sqrt(2))/1e9;
        # Calculate the projected k values
        self.kx = -K*( (np.sin(self.theta*np.pi/180)-1/np.sqrt(2))*np.cos(self.alpha*np.pi/180) )/1e9;
        self.ky = -K*(np.sin(self.theta*np.pi/180)-1/np.sqrt(2))*np.sin(self.alpha*np.pi/180)/1e9;
    

    def set_alpha_zero(self, alpha_zero=0):
        '''Sets the correct 0 for the azimuthal angle such that 0 lies along
        one of the principle azimuths: alpha_zero in degrees.'''
        self.alpha = self.alpha_rotator - alpha_zero
        self.alpha_zero = alpha_zero
        
    
    def shem_cartesian_plot(self, var, colourmap = cm.viridis,
                            figsize = [6,4], rasterized = True):
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
                        figsize = [8,6], rasterized = True, DK_invert=True):
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
            raise ValueError('Unknon colorbar location.')
        # The main plot, mask any nan (e.g. values that have been masked out)
        Z = np.log10(sP.signal)
        mesh1 = ax1.pcolormesh(sP.alpha*np.pi/180, getattr(sP, var), Z, 
                               edgecolors='face', cmap = colourmap,  rasterized=rasterized)
        # Thicker axis lines
        ax1.spines[:].set_linewidth(1.5)
        ax1.set_xlabel('$\\alpha$')
        ax1.grid(alpha=0.33)
        ax1.tick_params(axis='y', colors=[0.9,0.9,0.9])
        if bar_location == 'right':
            fig.colorbar(mesh1, cax=ax2, label="$\\log_{10}(I/\\mathrm{nA})$", shrink=0.1)
        elif bar_location == 'bottom':
            fig.colorbar(mesh1, cax=ax2, label="$\\log_{10}(I/\\mathrm{nA})$", orientation='horizontal')
            plt.subplots_adjust(hspace=0.3)
        return(fig, ax1, ax2)
    
    def shem_diffraction_plot(self, colourmap = cm.viridis, bar_location='right',
                              figsize=[8,6], rasterized=True, DK_invert=True, x_offset=0.08, DK_max=85):
        fig, ax1, ax2 = self.shem_polar_plot('DK', colourmap=colourmap, bar_location=bar_location, 
                                             figsize=figsize, rasterized=rasterized, DK_invert=DK_invert)
        ax1.set_yticks([0, 25, 50, 75])
        ax1.set_ylim(0, DK_max)
        ax1.tick_params(axis='y', colors=[0.9,0.9,0.9])
        add_scale(ax1, label = '$\Delta K/\\mathrm{nm}^{-1}$', x_offset = x_offset)
        return(fig, ax1, ax2)
    
    def shem_raw_plot(self, colourmap = cm.viridis,  bar_location='right',
                      figsize=[8,6], rasterized=True, x_offset = 0.08):
        fig, ax1, ax2 = self.shem_polar_plot('z', colourmap=colourmap, bar_location=bar_location, 
                                             figsize=figsize, rasterized=rasterized, DK_invert=True)
        ax1.set_yticks([0, 1, 2, 3, 4, 5, 6])
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
        K = 2*np.pi*np.sqrt(5*m_He*k_B*self.T)/h; # m^-1
        K = K/1e9
        a = 1.5 #mm
        b = 3.5 #mm
        cP = copy.deepcopy(self)
        
        # Shifted kx,ky
        cP.kx = cP.kx + D_kx
        cP.ky = cP.ky + D_ky
        #cP.DK = np.sqrt(cP.kx**2 + cP.ky**2)
        #cP.theta = np.arcsin(cP.DK/K + 1/np.sqrt(2))
        #cP.z = 2*b/(np.tan(cP.theta*pi/180) + 1) - a
        #TODO: calculate new alpha....
        cP.alpha = np.arctan2(cP.ky, cP.kx)*180/pi + 180
        cP.DK = cP.ky/np.sin(cP.alpha*pi/180)
        cP.alpha = cP.alpha - 180
        return(cP)