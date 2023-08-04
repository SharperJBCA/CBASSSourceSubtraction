#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:55:51 2023

@author: sharper
"""
# import as necessary
import numpy as np
import healpy as hp
from matplotlib import pyplot
import os
from types import GenericAlias 
from astropy.visualization.wcsaxes import SphericalCircle


def todegrees(theta,phi):
    return np.degrees(phi), np.degrees(np.pi/2-theta)

def tothetaphi(x,y):
    return np.pi/2-y*np.pi/180., x*np.pi/180.

HealpixMap = GenericAlias(np.ndarray, (float,))

loops = {'GCS':[1,[328,22]],
         'I':[2,[44,60]],
         'IX':[3,[7,26]],
         'X':[4,[48,15]],
         'III':[5,[112,32]],
         'IIIs':[6,[109,-38]],
         'VIIb':[7,[16,-66]],
         'Is':[8,[337,-35]],
         'VIII':[9,[332,-15]],
         '':[10,[319,20]], # XIV
         ' ':[11,[0,0]], # XIII
         'XI':[12,[232,10]],
         'XII':[13,[250,-60]]}
loops = {'I':[2,[44,60]],
         'IX':[3,[7,26]],
         'X':[4,[48,15]],
         'III':[5,[112,32]],
         'IIIs':[6,[109,-38]],
         'VIIb':[7,[16,-66]],
         'XI':[12,[232,10]]}

# Interpolate using great circle distances
def greatcircleinterp(lons, lats, distance=0.5):
    ''' It takes a list of coordinates in degrees and it interpolates along the
    great circle, where the spacing between subsequent points is given by the
    distance parameter, in degrees '''
    # Fill inbetween
    interplons = []
    interplats = []
    for i in np.arange(np.size(lons))[:-1]:
        mylons, mylats = greatcircle(ptlon1=lons[i], ptlat1=lats[i], ptlon2=lons[i+1], ptlat2=lats[i+1], distance=distance)
        if i ==0: # copy the first point too
            interplons.append(mylons)
            interplats.append(mylats)
        else: # don't copy the first point since it would be a duplicate
            interplons.append(mylons[1:])
            interplats.append(mylats[1:])

    # Flatten the arrays
    interplons = [item for sublist in interplons for item in sublist]
    interplats = [item for sublist in interplats for item in sublist]

    return interplons, interplats

def read_regions_file(loop_number, AME=False,regs_dir='scripts/plotting/regs/'):
    ''' Read Matias' region file '''
    
    # Function to format the angles properly when read from Matias' files
    def format_angle(lon_str, lat_str):
        ''' Converts the strings from reading the file into decimal degrees '''

        # Function to convert +ddd:mm:ss.sss into decimal degrees
        def dms2decimal(string):
            angle = string.split(':')
            angle = [float(x) for x in angle]
            decimal_angle = angle[0]+angle[1]/60.+angle[2]/3600.
            return decimal_angle

        # Format angles properly
        def turn_decimal(str):
            if ':' in str:
                dec = dms2decimal(str)
            else:
                dec = float(str)
            return dec

        lon = turn_decimal(lon_str)
        lat = turn_decimal(lat_str)

        return lon, lat

    # Read the file and return the coordinates
    import glob
    if isinstance(loop_number, str):
        address = f'{regs_dir}/amespur{loop_number[3]}.reg'
    else:
        address = glob.glob(f'{regs_dir}/short_{loop_number}xyrad*.reg')[0]

    import csv
    longitudes = []
    latitudes = []
    with open(address) as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            lon_str, lat_str = [x for x in line[0].split(' ') if x!='']
            # Format angles properly
            lon, lat = format_angle(lon_str, lat_str)
            # Append numbers
            longitudes.append(lon)
            latitudes.append(lat)
        return longitudes, latitudes

    # Open file and read coordinates
    return read_regions_file(loop_number)


def ud_grade_qucov(inmap, cov, nside_out : int, nocorr : bool = False) -> (np.ndarray, np.ndarray):
    """
    function to degrade polarization (Q,U) maps correctly with a covariance matrix

    Parameters
    ----------
    inmap : np.ndarray[float] (Nmaps, Npix)
       Input high-resolution map (should be Q, U only)
    cov : np.ndarray[float] (Nmaps, Npix)
       Noise covariance matrix (QQ, UU, QU) - must have same units as inmap
    nside_out : int
       Output nside requested

    Keywords
    --------

    nocorr : bool (Default=False, take correlations into account)
        If set to True, assume no correlations between Q and U and do a simple downgrade (much faster)

    Returns
    -------
    
    outmap : np.ndarray[float] (Nmaps, Npix)
        Output degraded map at requested Nside
    outcov : np.ndarray[float] (Nmaps, Npix)
        Covariance matrix elements for degraded map (QQ, UU, QU)


    VERSION HISTORY:
    ----------------
    28-Aug-2019  C. Dickinson  1st go
    10-Mar-2020  C. Dickinson  Tidied up version and fully checked
    11-Mar-2020  C. Dickinson  Added nocorr option to allow much speedier operation


    NOTES:

        -Assumes map are RING ordered (automatically converted to ring when reading in anyway)
        -Pixels at edge of unobserved region may not survive - currently throw any sub-pixels away to be safe
        -Input covariances should not be tiny (e.g. 10^-20) otherwise (10^-12 is ok)
            the flagging of bad pixels may not work

    """    
    # check parameters....(to-do!)
    if isinstance(inmap, list):
        inmap = np.array(inmap)
    if isinstance(cov, list):
        cov = np.array(cov)
    # useful info
    npix_out = hp.nside2npix(nside_out)
    nside_in = hp.get_nside(inmap)    
    npix_in = hp.nside2npix(nside_in)

    # get the pixel ID for all high-res pixels in the low res map
    ip_out = np.arange(npix_out)
    ip_in = np.arange(npix_in)
    glon, glat = hp.pix2ang(nside_in, ip_in, lonlat=True)
    pixid = hp.ang2pix(nside_out, glon, glat, lonlat=True)

    # create array for hires weight matrix for later
    w_in = np.zeros((3,npix_in))  ## QQ, UU, QU

    # create arrays for final outputs
    outmap = np.zeros((2,npix_out)) + hp.UNSEEN  # Q, U
    outcov = np.zeros((3,npix_out)) + hp.UNSEEN # QQ, UU, QU

    # Find where there are good data for later
    goodvals = np.array(np.nonzero(inmap[0,:] > -1e20)).flatten()
    goodvals_bool = inmap[0,:] > -1e20  # boolean version for later

    # ensure bad pixels are set to the correct bad value for later (for e.g. ud_grade)
    badvals_bool = inmap[0,:] < -1e20
    inmap[:,badvals_bool] = hp.UNSEEN

    # Do full calculation when nocorr=False (Default)
    if (nocorr == False):
        
        # Loop over each pixel that contains good data to noise weight it first
        for i in goodvals:
            #for i in goodvals[0:300000]:  # testing
        
            # progress print statements
            if (i / 100000 == i // 100000):
                print('Weighting map: {0:.0f} out of {1:.0f} ({2:.1f}%)'.format(i,npix_in,i/npix_in*100))
            
            # 1. High-res map is noise-weighted

            # get covariance matrix
            c = [ [cov[0,i], cov[2,i]], [cov[2,i], cov[1,i]] ]

            # invert it to get inverse variances
            w = np.linalg.inv(np.array(c))
        
            # do the noise weighting
            inmap[:,i] = np.dot(w,inmap[:,i])
        
            # store the inverse variance (weight) at hires for later
            w_in[:,i] = [w[0,0], w[1,1], w[0,1]] 
    
        # 2a. Downgrade the weighted map    ###by summing the values (not the mean, so power=-2)
        #     Note that pess=True so not to average bad values but may lose some pixels this way!!
        inmap = hp.ud_grade(inmap, nside_out, pess=True, power=-2)

        #Set bad pixels to bad value and remember for later
        badvals = np.nonzero(inmap < -1e10) # set to -1e10 because it has been multiplied by the weights and degraded!
        inmap[:,badvals] = hp.UNSEEN
    
        # loop over each pixel in the final map
        for i in ip_out:

            # progress print statements
            if (i / 100 == i // 100):
                print('Making downgraded map/cov: {0:.0f} out of {1:.0f} ({2:.1f}%)'.format(i,npix_out,i/npix_out*100))

            # get subpixels - making sure we don't include bad data (a little slow but ok)
            usepix = np.nonzero((pixid == i) & (goodvals_bool))

            # skip to next pixel if no usable data
            if (np.size(usepix) == 0):
                continue
                
            # 2b. Downgrade the covariance matrix by summing weights and then inverting
            w_lores = [ [np.sum(w_in[0,usepix]), np.sum(w_in[2,usepix])], [np.sum(w_in[2,usepix]), np.sum(w_in[1,usepix])] ]
            cov_lores = np.linalg.inv(w_lores)

            # 3. Multiply weighted map by lowres covariance matrix (1/sum of weights)
            outmap[:,i] = np.dot(cov_lores,inmap[:,i])
        
            # put the results back into the array (QQ, UU, QU)
            outcov[:,i] = [cov_lores[0,0], cov_lores[1,1], cov_lores[0,1]]


        # put bad pixels back to bad value
        outmap[:,badvals] = hp.UNSEEN
        outcov[:,badvals] = hp.UNSEEN

    # if nocorr != True then do simple downgrade
    else:

        #Set bad pixels to bad value 
        badvals = np.nonzero(inmap < -1e10) # 
        inmap[:,badvals] = hp.UNSEEN
        cov[:,badvals]   = hp.UNSEEN
                
        outmap = hp.ud_grade(inmap, nside_out)
        outcov = hp.ud_grade(cov, nside_out, power=2)

        
    return outmap, outcov

def unwrap_angles(angles : np.ndarray) -> np.ndarray:
    """
    function to take some pol angles and minimize their differences and unwrapping for fitting RMs
    simple algorithm which assumes rm is the minimum it can be (n=0), and the angle variation is continuous
    Angles should be in degrees and frequency should be increasing

    Parameters
    ----------
    angles : ndarray[float]
        Angles in degrees.

    Returns
    -------
    None.

    """
    angles_unwrapped = angles.copy()
    
    # only look at useable values
    goodvals = np.array(np.nonzero(np.abs(angles_unwrapped) < 1e20)).flatten()
    angles_temp = angles_unwrapped[goodvals]
    
    for i in np.arange(np.size(angles_temp)-1):

        # get difference with next good value
        diff = angles_temp[i] - angles_temp[i+1]

        # make +/- 180 deg changes as necessary
        if (diff > 90):
            angles_temp[i+1] = angles_temp[i+1] + 180 
        elif (diff < -90):  # elseif ensures this isn't done twice
             angles_temp[i+1] = angles_temp[i+1] - 180

    # put back into original array including badvals if they are there
    angles_unwrapped[goodvals] = angles_temp[:]
             #    
    return angles_unwrapped

def get_pixel_angles(nside : int, lonlat : bool=True):
    """Get all angles for a healpix grid"""
    
    pixels = np.arange(12*nside**2, dtype=int)
    theta,phi = hp.pix2ang(nside, pixels)

    if lonlat:
        theta = (np.pi/2.-theta)*180./np.pi 
        phi = phi*180./np.pi 
        return phi, theta
    else:
        return theta, phi

def mollview_func(*args,savefig=None,**kwargs):
    """ """
    figure = pyplot.figure(1)
    hp.mollview(*args,fig=1,**kwargs)
    if savefig != None:
        if not os.path.exists(os.path.dirname(savefig)):
            os.makedirs(os.path.dirname(savefig))
        pyplot.savefig(savefig)
    pyplot.close(figure)
    
from astropy.wcs import WCS 
from dataclasses import dataclass, field 

from matplotlib.figure import Figure 
from matplotlib.axes import Axes

from astropy.visualization.wcsaxes.frame import EllipticalFrame
from astropy.visualization import simple_norm, HistEqStretch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from reproject import reproject_from_healpix

#from cmcrameri import cm
from matplotlib import cm

@dataclass 
class Mollview:
    
    map : HealpixMap = field(default_factory=lambda : np.zeros(1)) 
    wcs : WCS = field(default_factory=lambda : WCS(naxis=2)) 
    Nx : int = 0
    Ny : int = 0
    interpolation : str = 'nearest-neighbor' 
    
    # Matplotlib info 
    axes : Axes = field(default_factory = lambda : None )
    figure : Figure = field(default_factory = lambda : None )
    
    def __post_init__(self): 
        
        
        # build wcs first
    
        cdelt = 1./6
        self.Nx,self.Ny = int(360//cdelt*0.895), int(180//cdelt*0.895)
        self.wcs.wcs.crpix=[self.Nx//2,self.Ny//2+2]
        self.wcs.wcs.cdelt=[-cdelt, cdelt]
        self.wcs.wcs.crval=[0,0]
        self.wcs.wcs.ctype=['GLON-MOL','GLAT-MOL']
        
    def __call__(self, m : HealpixMap,
                 axes = None, 
                 figure = None, 
                 norm : str =None, 
                 vmin : float =None, vmax : float =None, cmap=cm.viridis): 
        """
        

        Parameters
        ----------
        m : HealpixMap
            DESCRIPTION.
        norm : str, optional
            DESCRIPTION. The default is None.
        vmin : float, optional
            DESCRIPTION. The default is None.
        vmax : float, optional
            DESCRIPTION. The default is None.
        cmap : TYPE, optional
            DESCRIPTION. The default is cm.batlow_r.

        Returns
        -------
        None.

        """
        
        # now reproject
        array, footprint = reproject_from_healpix((m,'galactic'), self.wcs,
                                                  shape_out=[self.Ny,self.Nx],
                                                  nested=False,order=self.interpolation)
        array[(array == hp.UNSEEN) | (np.abs(array) > 1e10)] =np.nan
        

        if isinstance(figure, type(None)):
            self.figure = pyplot.figure()
        else:
            self.figure = figure 
        if isinstance(axes, type(None)):    
            self.axes = pyplot.subplot(111,projection=self.wcs,frame_class=EllipticalFrame)
        else:
            self.axes = axes 
        self.img = self.imshow(array,vmin=vmin,vmax=vmax,cmap=cmap,interpolation='bilinear', norm=None)
        
    def remove_ticks(self):
        
        self.axes.coords[0].set_ticks_visible(False)
        self.axes.coords[0].set_ticklabel_visible(False)
        self.axes.coords[1].set_ticks_visible(False)
        self.axes.coords[1].set_ticklabel_visible(False)
        
    def add_grid(self, color='k'):
        """Add grid to image"""
        self.axes.coords.grid(color=color)
        self.axes.coords['glon'].set_ticklabel(color=color)

    def add_colorbar(self, unit_label=' ',  ticks=None):
        """Add colorbar"""
        
        axins1 = inset_axes(self.axes, width='100%', height='5%', loc='upper center', 
                    bbox_to_anchor=(0.,-1.,1,1), 
                    bbox_transform=self.axes.transAxes)
        cb = self.figure.colorbar(self.img,cax=axins1,orientation='horizontal',ticks=ticks)
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.xaxis.set_label_position('bottom')
        cb.set_label(unit_label)
        
    def add_loops(self):
        """ Add synchrotron loops to mollview plot""" 
        for k,v in loops.items():
            lons,lats = read_regions_file(v[0], regs_dir='../ancillary_data/loops/')
            lons, lats = greatcircleinterp(lons, lats, distance=0.1)
    
            self.axes.plot(lons,lats,transform=self.axes.get_transform('galactic'),lw=4,ls='--',color='k')
            self.axes.text(v[1][0],v[1][1],k, color='k', transform=self.axes.get_transform('galactic'),ha='center')


    def imshow(self, array,vmin=None,vmax=None,cmap=None,interpolation='nearest', norm='hist'):
        """Wrapper for matplotlib imshow that allows for different normalisations""" 
        
        array, vmin, vmax, norm_module = self.norm(array, vmin, vmax, norm) 
        self.img = self.axes.imshow(array,norm=norm_module,cmap=cmap,vmin=vmin,vmax=vmax)

        return self.img 
    
    def contourf(self, m,levels=[0.5,1],vmin=None,vmax=None,cmap=None,interpolation='nearest'):
        array, footprint = reproject_from_healpix((m,'galactic'), self.wcs,
                                                  shape_out=[self.Ny,self.Nx],
                                                  nested=False,order=self.interpolation)
        array[(array == hp.UNSEEN) | (np.abs(array) > 1e10)] = np.nan
        axes_contour = pyplot.subplot(111,projection=self.wcs,frame_class=EllipticalFrame)

        
        contourf = axes_contour.contourf(array,levels=levels,cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.5)
        array[np.isnan(array)] = 0
        contour = axes_contour.contour(array,colors='k',levels=levels,vmin=vmin,vmax=vmax,linewidths=0.5)
        return contourf 

    def norm(self, array, vmin, vmax, norm):
        """Normalise data""" 
        
        if isinstance(norm, type(None)):
            norm_module = None 
        else:
            if norm =='hist':
                amin = np.nanmin(array)
                st = HistEqStretch(array-np.nanmin(array))
                array = st(array-np.nanmin(array))
                norm_module=None
                if not isinstance(vmin,type(None)):
                    vmin = st(np.array([vmin-amin]))[0]
                if not isinstance(vmax,type(None)):
                    vmax = st(np.array([vmax-amin]))[0]
            else:
                norm_module = simple_norm(array,norm)
        return array, vmin, vmax, norm_module


    def text(self,x,y,text,**kwargs):
        """Wrapper for matplotlib text""" 
        self.axes.text(x,y,text,transform=self.axes.get_transform('galactic'),**kwargs)
                
                
@dataclass 
class Gnomview:
    
    map : HealpixMap = field(default_factory=lambda : np.zeros(1)) 
    wcs : WCS = field(default_factory=lambda : WCS(naxis=2)) 
    Nx : int = 256
    Ny : int = 256
    interpolation : str = 'nearest-neighbor' 
    
    crval : list = field(default_factory=lambda : [0,0]) 
    cdelt : list = field(default_factory=lambda : [-5./60.,5./60.])
    # Matplotlib info 
    axes : Axes = field(default_factory = lambda : None )
    figure : Figure = field(default_factory = lambda : None )
    
    def __post_init__(self): 
        
        
        # build wcs first
    
        self.wcs.wcs.crpix=[self.Nx//2,self.Ny//2]
        self.wcs.wcs.cdelt=self.cdelt
        self.wcs.wcs.crval=self.crval
        self.wcs.wcs.ctype=['GLON-TAN','GLAT-TAN']
        
    def __call__(self, m : HealpixMap,
                 axes = None, 
                 figure = None, 
                 norm : str =None, 
                 vmin : float =None, vmax : float =None, cmap=cm.viridis): 
        """
        

        Parameters
        ----------
        m : HealpixMap
            DESCRIPTION.
        norm : str, optional
            DESCRIPTION. The default is None.
        vmin : float, optional
            DESCRIPTION. The default is None.
        vmax : float, optional
            DESCRIPTION. The default is None.
        cmap : TYPE, optional
            DESCRIPTION. The default is cm.batlow_r.

        Returns
        -------
        None.

        """
        
        # now reproject
        m[m == 0] = hp.UNSEEN
        m[np.isnan(m)] = hp.UNSEEN 
        array, footprint = reproject_from_healpix((m,'galactic'), self.wcs,
                                                  shape_out=[self.Ny,self.Nx],
                                                  nested=False,order=self.interpolation)
        array[(array == hp.UNSEEN) | (np.abs(array) > 1e10)] =np.nan

        if np.nansum(array) == 0:
            raise ValueError('No data to plot')
        
        if isinstance(figure, type(None)):
            self.figure = pyplot.figure()
        else:
            self.figure = figure 
        if isinstance(axes, type(None)):    
            self.axes = pyplot.subplot(111,projection=self.wcs)
        else:
            self.axes = axes 
        self.img = self.imshow(array,vmin=vmin,vmax=vmax,cmap=cmap,interpolation='bilinear', norm=None)

        lon = self.axes.coords[0]
        lat = self.axes.coords[1]
        lon.set_axislabel('Galactic Longitude')
        lat.set_axislabel('Galactic Latitude')

        return self.img
        
    def remove_ticks(self):
        
        self.axes.coords[0].set_ticks_visible(False)
        self.axes.coords[0].set_ticklabel_visible(False)
        self.axes.coords[1].set_ticks_visible(False)
        self.axes.coords[1].set_ticklabel_visible(False)
        
    def add_grid(self, color='k'):
        """Add grid to image"""
        self.axes.coords.grid(color=color)
        self.axes.coords['glon'].set_ticklabel(color=color)

    def add_colorbar(self, unit_label=' ',  ticks=None):
        """Add colorbar"""
        
        axins1 = inset_axes(self.axes, width='5%', height='100%', loc='upper center', 
                    bbox_to_anchor=(0.6,0.,1,1), 
                    bbox_transform=self.axes.transAxes)
        cb = self.figure.colorbar(self.img,cax=axins1,orientation='vertical',ticks=ticks)
        #cb.ax.xaxis.set_ticks_position('bottom')
        #cb.ax.xaxis.set_label_position('bottom')
        cb.set_label(unit_label)

    def contourf(self, m,vmin=None,vmax=None,cmap=None,levels=[0,1],interpolation='nearest',alpha=0.5):
        array, footprint = reproject_from_healpix((m,'galactic'), self.wcs,
                                                  shape_out=[self.Ny,self.Nx],
                                                  nested=False,order=self.interpolation)
        array[(array == hp.UNSEEN) | (np.abs(array) > 1e10)] = np.nan
        axes_contour = pyplot.subplot(111,projection=self.wcs)

        
        contourf = axes_contour.contourf(array,levels=levels,cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
        array[np.isnan(array)] = 0
        contour = axes_contour.contour(array,colors='k',levels=levels,vmin=vmin,vmax=vmax,linewidths=0.5)
        return contourf 


    def imshow(self, array,vmin=None,vmax=None,cmap=None,interpolation='nearest', norm='hist'):
        """Wrapper for matplotlib imshow that allows for different normalisations""" 
        
        array, vmin, vmax, norm_module = self.norm(array, vmin, vmax, norm) 
        self.img = self.axes.imshow(array,norm=norm_module,cmap=cmap,vmin=vmin,vmax=vmax)

        return self.img 
    
    def norm(self, array, vmin, vmax, norm):
        """Normalise data""" 
        
        if isinstance(norm, type(None)):
            norm_module = None 
        else:
            if norm =='hist':
                amin = np.nanmin(array)
                st = HistEqStretch(array-np.nanmin(array))
                array = st(array-np.nanmin(array))
                norm_module=None
                if not isinstance(vmin,type(None)):
                    vmin = st(np.array([vmin-amin]))[0]
                if not isinstance(vmax,type(None)):
                    vmax = st(np.array([vmax-amin]))[0]
            else:
                norm_module = simple_norm(array,norm)
        return array, vmin, vmax, norm_module