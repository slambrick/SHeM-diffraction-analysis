#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 09:51:59 2022

@author: Sam Lambrick
@contributions: Ke Wang

A module for importing, analysing, and plotting SHeM spot profile data.

The 2D Gaussian function is based on this StackOverflow post:
    https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
"""

import numpy as np
from numpy import pi
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm # Colour palettes
import matplotlib.gridspec as gridspec
import os
from scipy import interpolate as interp
import pandas as pd


def speed_ratio(P_0, d_noz):
    """This fit is from M.Bergi2018 doi.org/10.1016/j.ultramic.2019.112833
    and is to data/simulation from Toennies & Winkelman doi.org/10.1063/1.434448."""
    a = 0.43
    b = 0.76
    c = 0.84
    d = 5.2
    mu = 1.97
    lS = a*np.log10(P_0*d_noz) + b + c/(1 + np.exp(-d*(np.log10(P_0*d_noz) - mu)))
    return(10**(lS))

def morse_V(z, x, y, params=(8.03, 1.35, 1.08, 0.102, 4)):
    """Calculates a corrugated Morse potential of the form used by Celli et 
    al 1985., x,y,z in Angstroms."""
    D0, alpha, alpha1, h, a = params
    Q = h*(np.cos(2*pi*x/a) + np.cos(2*pi*y/a))
    v = D0*(np.exp(-2*alpha*(z - Q)) - 2*np.exp(-alpha1*z))
    return(v)

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, f, g, h, k, l, m):
    '''Two dimensional gaussian for fitting.'''
    
    x, y = xy
    # TODO: add a rotation to this for more general fitting 
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    tot = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    #f, g, h, k, l, m = offset
    tot = tot + f + g*x + h*y + k*x*y + l*x**2 + m*y**2
    return tot.ravel()

def background(xy, f, g, h, k, l, m):
    x, y = xy
    tot = f + g*x + h*y + k*x*y + l*x**2 + m*y**2
    return(tot)


def theta_of_z(z):
    '''Calculates the detection angle theta from the z position in mm
    for the 1st gneration angular resolution pinhole plate.
    plate: #C06'''
    return(np.arctan((7 - (z+1.5))/(z+1.5))*180/np.pi)


def load_z_scans(files_ind, path_name, z_zero = 3.41e6):
    '''Loads a series of z scans taken at different aximulath orientations. The
    orientations need to provied seperatly as they are not stored in the data
    files (as of 5/12/20).
    
    The value of the z zero position should be given in um.
    
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
    '''A function to add an extra scale to the polar plot with the provided
    axis handle. The offset determines where the scale is drawn.'''
    
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


class SpotProfile:
    '''This class contains results on a spot profile SHeM measurement (currently
    loaded in from a series of z-scans) along with data analysis and plotting
    functions.'''
    
    def __init__(self, z, alpha_rotator, I):
        # TODO: example positions etc. not really being used yet!
        self.z = z                        # z position, matrix
        self.theta = theta_of_z(z)        # Polar detection angle, matrix
        self.alpha_rotator = alpha_rotator# Rotator stage angle, matrix
        self.alpha = np.array([])         # azimuthal angle orientated with the crystal, matrix
        self.signal = I                   # Signal levels, matrix
        self.kz = np.array([])            # wavevector z values
        self.DK = np.array([])            # parallel momentum transfer
        self.kx = np.array([])
        self.ky = np.array([])
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
    
    # TODO: make the simulated data work too!
    @classmethod
    def import_ray(cls, data_dir, T=298, alpha_zero=0):
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
        zs = zs - 1.5;
   
        alphas = np.repeat(np.resize(np.array(alphas), [1, n_alpha]), n_z, axis=0)
        zs = np.repeat(np.resize(zs, [n_z, 1]), n_alpha, axis=1)
        sP = cls(z = zs, alpha_rotator = alphas, I = I)
        
        # The example image the spot was defined from was at 300deg
        sP.T = T
        sP.example_alpha = np.nan
        sP.set_alpha_zero(0)
        sP.calc_dK()
        return(sP)
    
    def select_by_var(self, var, value):
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
    
    def line_plot(self, xvar, var, value, figsize=[5, 3.5], logplot=False):
         df, chosen = self.select_by_var(var, value)
         f = plt.figure(figsize=figsize)
         ax = f.add_axes([0,0,1,1])
         if logplot:
             df['signal'] = np.log10(df['signal'])
         df.plot(x = xvar, y = 'signal', ax = ax)
         if logplot:
             ax.set_ylabel('$\\log_{10}$(Intensity/nA)')
         else:
             ax.set_ylabel('Intensity/nA')
         ax.grid(True)
         return(f, ax, df, chosen)
     
    def line_plot_raw(self, alpha, figsize=[5, 3.5], logplot=False):
        f, ax, df, chosen_alpha =self.line_plot('z', 'alpha', alpha, figsize=figsize, logplot=logplot)
        ax.set_xlabel('$z/\\mathrm{mm}$')
        ax.set_title('1D plot of a zscan at $\\alpha$={}$^\\circ$'.format(chosen_alpha))
        return(f, ax, df)
    
    def line_plot_diffraction(self, alpha, figsize=[5, 3.5], logplot=False):
        f, ax, df, chosen_alpha =self.line_plot('DK', 'alpha', alpha, figsize=figsize, logplot=logplot)
        ax.set_xlabel('$\\Delta K/\\mathrm{nm}^{-1}$')
        ax.set_title('1D diffraction scan at $\\alpha$={}$^\\circ$'.format(chosen_alpha))
        return(f, ax, df)
    
    def line_plot_theta(self, alpha, figsize=[5, 3.5], logplot=False):
        f, ax, df, chosen_alpha =self.line_plot('theta', 'alpha', alpha, figsize=figsize, logplot=logplot)
        ax.set_xlabel('$\\theta_f/^\\circ$')
        ax.set_title('1D diffraction scan at $\\alpha$={}$^\\circ$'.format(chosen_alpha))
        return(f, ax, df)
    
    def line_plot_alpha(self, z, figsize=[5, 3.5], logplot=False):
        f, ax, df, chosen_alpha =self.line_plot('alpha', 'z', z, figsize=figsize, logplot=logplot)
        ax.set_xlabel('$\\alpha/^\\circ$')
        ax.set_title('$\\alpha$ scan at z={}mm'.format(z))
        return(f, ax, df)
    
    def calc_dK(self):
        '''Calculates the momentum transer for the data file and the
        "psuedo" kx, ky that we have defined.'''
        # Constants for the wavevector magnitude
        m_u = 1.6605e-27 # kg
        m_He = 4*m_u;
        k_B = 1.380649e-23 # JK^-1
        h = 6.62607015e-34 # Js
        K = 2*np.pi*np.sqrt(5*m_He*k_B*self.T)/h; # m^-1

        # Calculates the parallel momentum transfer in nm^-1
        self.DK = K*(np.sin(self.theta*np.pi/180) - 1/np.sqrt(2))/1e9;
        # Calculate the projected k values
        self.kx = K*( (np.sin(self.theta*np.pi/180)-1/np.sqrt(2))*np.cos(self.alpha*np.pi/180) )/1e9;
        self.ky = K*(np.sin(self.theta*np.pi/180)-1/np.sqrt(2))*np.sin(self.alpha*np.pi/180)/1e9;
    

    def set_alpha_zero(self, alpha_zero=0):
        '''Sets the correct 0 for the azimuthal angle such that 0 lies along
        one of the principle azimuths: alpha_zero in degrees'''
        self.alpha = self.alpha_rotator - alpha_zero
        self.alpha_zero = alpha_zero
        #self.phi(obj.alpha < 0) = self.phi(self.alpha < 0) + 360
        #self.phi(obj.alpha >= 360) = self.phi(self.alpha  >= 360) + 360
        # TODO: tbh this simple version appears to be working fine, but I'm not
        # 100% convinced by it...
    
    def shem_diffraction_plot(self, colourmap = cm.viridis, bar_location='right', figsize=[8,6], rasterized=True):
        sP = self.filter_by_var('z', 2)
        fig = plt.figure(figsize=figsize)
        if bar_location == 'right':
            gs = fig.add_gridspec(1,2,width_ratios=[10,0.5])
            ax1=plt.subplot(gs[0], projection="polar", aspect=1.)
            ax2 = plt.subplot(gs[1])
        elif bar_location == 'bottom':
            # The subplot size on the grid is determining the width of the colorbar
            gs = fig.add_gridspec(2,3,height_ratios=[10,0.5], width_ratios=[1,4,1])
            ax1=plt.subplot(gs[0:3], projection="polar", aspect=1.)
            ax2 = plt.subplot(gs[4])
        else:
            raise ValueError('Unknon colorbar location.')
        mesh1 = ax1.pcolormesh(sP.alpha*np.pi/180, -sP.DK, np.log10(sP.signal), 
                               edgecolors='face', cmap = colourmap,  rasterized=rasterized)
        ax1.set_xlabel('$\\alpha$')
        ax1.grid()
        ax1.set_yticks([0, 25, 50, 75])
        ax1.tick_params(axis='y', colors=[0.9,0.9,0.9])
        #ax1.annotate('DK/nm^-1', (28*np.pi/180, 92), annotation_clip=False)
        add_scale(ax1, label = '$\Delta K/\\mathrm{nm}^{-1}$')
        if bar_location == 'right':
            fig.colorbar(mesh1, cax=ax2, label="$\\log_{10}(I/\\mathrm{nA})$")
        elif bar_location == 'bottom':
            ax2.set
            fig.colorbar(mesh1, cax=ax2, label="$\\log_{10}(I/\\mathrm{nA})$", orientation='horizontal')
            plt.subplots_adjust(hspace=0.3)
        return(fig, ax1, ax2)
    
    def shem_raw_plot(self, colourmap = cm.viridis):
        fig = plt.figure(figsize=[8,6])
        gs = fig.add_gridspec(1,2,width_ratios=[10,0.5])
        ax1=plt.subplot(gs[0], projection="polar", aspect=1.)
        ax2 = plt.subplot(gs[1])
        mesh1 = ax1.pcolormesh(self.alpha*np.pi/180, self.z, np.log10(self.signal), 
                               edgecolors='face', cmap = colourmap)
        ax1.set_xlabel('$\\alpha$')
        ax1.grid()
        ax1.set_yticks([0, 1, 2, 3, 4, 5, 6])
        ax1.tick_params(axis='y', colors=[0.9,0.9,0.9])
        #ax1.annotate('DK/nm^-1', (28*np.pi/180, 92), annotation_clip=False)
        add_scale(ax1, label = 'z/mm')
        fig.colorbar(mesh1, cax=ax2, label="$\\log_{10}(I/\\mathrm{nA})$")
        return(fig, ax1, ax2)
    
    
    def filter_by_var(self, var, value):
        # TODO: allow filtering by other variabbles
        if var == 'z':
            # Drop data that is z < 2 <=> Dk > 0
            sh = self.z.shape
            ind = self.z > value
            z2 = self.z[ind]
            ax2 = sh[1]
            ax1 = int(z2.size/ax2)
            sP = SpotProfile(z = self.z[ind].reshape(ax1, ax2),
                             alpha_rotator = self.alpha_rotator[ind].reshape(ax1, ax2),
                             I =  self.signal[ind].reshape(ax1, ax2))
        sP.set_alpha_zero(self.alpha_zero)
        sP.calc_dK()
        return(sP)
    
    def grid_interpolate(self, kx, ky, N=101, method='nearest'):
        '''Produces a 2D cartesian grid of the data for the provided kx and ky
        vectors.'''
        kx = np.linspace(kx[0], kx[1], N)
        ky = np.linspace(ky[0], ky[1], N)
        kxx, kyy = np.meshgrid(kx, ky)
        I = interp.griddata((self.kx.flatten(), self.ky.flatten()), 
                        self.signal.flatten(), (kxx, kyy), method=method)
        return(I, kxx, kyy)
    
    def interpolated_plot(self, kx=(-80, 80), ky=(-80,80), N=101, method='nearest', ax=None):
        '''Produce an interpolated plot of the data with N points along the
        kx and ky axes. Useful for checking the output of interpolation'''
        I, kxx, kyy = self.grid_interpolate(kx, ky, N, method=method)
        if ax == None:
            f = plt.figure()
            ax = f.add_axes([0.15, 0.1, 0.7, 0.9])
        else:
            f = ax.get_figure()
        ax.pcolormesh(kxx, kyy, np.log10(I), edgecolors='face')
        ax.axis('equal')
        ax.set_xlabel('$k_x/\mathrm{nm}^{-1}$')
        ax.set_ylabel('$k_y/\mathrm{nm}^{-1}$')
        ax.set_title('Interpolated k-plot, method = '+ method)
        return(f, ax)
