import numpy as np
from matplotlib import pyplot
import healpy as hp
from SourceSubtraction.Tools import gsl_funcs
import os 
import healpy as hp
import h5py

def pixel_space(flux, glon, glat, nside=1024, lside=256,lmax=None,
             cutoff=None, fwhm=1, skymask=None, write_out=False,
             frequency=4.76,beam_model=None,verbose=False,
             use_corr_flux=False):
    """
    Bins sources into map in pixel-space.

    Returns:
    Source map, mask, source_fluxes, source_lons, source_lats
    """

    kb = 1.3806503e-23
    c = 2.99792458e8
    nu = frequency*1e9
    beam_solid_angle = 1.133 * (np.pi/180.)**2
    pixel_solid_angle = 4*np.pi/(12*nside**2)
    calfactor = 2 * kb * (nu/c)**2 * 1e26 * pixel_solid_angle
    if isinstance(lmax, type(None)):
        lmax = 3*lside

    if not isinstance(beam_model, type(None)):
        dtheta = np.abs(beam_model[0,0]-beam_model[1,0])*np.pi/180.
        beam_solid_angle = 2*np.pi*np.sum(beam_model[:,1]*np.sin(beam_model[:,0]*np.pi/180.))*dtheta
        bl = hp.beam2bl(beam_model[:,1],beam_model[:,0]*np.pi/180.,lmax)
    if not isinstance(fwhm, type(None)):
        bl = hp.gauss_beam(fwhm*np.pi/180.,lmax)

    pixels = hp.ang2pix(nside, np.pi/2-glat*np.pi/180., glon*np.pi/180.)
    pixel_edges = np.arange(hp.nside2npix(nside)+1)
    sum_map = np.histogram(pixels, bins=pixel_edges, weights=flux)[0] # Jy/beam

    # Apply beam here:
    alm = hp.map2alm(sum_map)
    alm = hp.almxfl(alm,bl)
    sum_map = hp.alm2map(alm,nside)

    return sum_map / calfactor 

def alm_space(flux, glon, glat, nside=1024, lside=256,lmax=None,
             cutoff=None, fwhm=1, skymask=None, write_out=False,
             frequency=4.76,beam_model=None,verbose=False,
             use_corr_flux=False):
    """
    Bins sources into map in alm-space.

    Returns:
    Source map, mask, source_fluxes, source_lons, source_lats
    """

    kb = 1.3806503e-23
    c = 2.99792458e8
    nu = frequency*1e9
    beam_solid_angle = 1.133 * (np.pi/180.)**2
    pixel_solid_angle = 4*np.pi/(12*nside**2)
    calfactor = 2 * kb * (nu/c)**2 * 1e26
    if isinstance(lmax, type(None)):
        lmax = 3*lside

    dtheta = np.abs(beam_model[0,0]-beam_model[1,0])*np.pi/180.
    beam_solid_angle = 2*np.pi*np.sum(beam_model[:,1]*np.sin(beam_model[:,0]*np.pi/180.))*dtheta
    bl = hp.beam2bl(beam_model[:,1],beam_model[:,0]*np.pi/180.,lmax)
    #bl /= bl[0]

    # Calculate l, m and alm indices
    Nalm = hp.Alm.getsize(lmax)
    idx = np.arange(Nalm,dtype=int)
    l, m = hp.Alm.getlm(lmax,idx)

    # This calculates the source locations in Alm space using GSL legendre libraries
    if verbose:
        print('Calculating alms...')

    if use_corr_flux:
        h = h5py.File('AncillaryData/FluxCorrections.hdf5','r')
        sourceInfo = np.array([glon*np.pi/180.,
                               (90-glat)*np.pi/180.,
                               flux*h['flux_corrections'][:]]).T
        h.close()
    else:
        sourceInfo = np.array([glon*np.pi/180.,
                               (90-glat)*np.pi/180.,
                               flux]).T

    d_alms = gsl_funcs.precomp_harmonics(sourceInfo,idx.size,np.max(l))
    if verbose:
        print('...Done')
    alms = d_alms[:,0] - 1j*d_alms[:,1]
    alms[np.isfinite(alms) == False] = 0 + 1j*0

    # Apply beam here:
    if verbose:
        print('Smoothing')
    if isinstance(beam_model,type(None)):
       alms = hp.smoothalm(alms,fwhm=fwhm*np.pi/180.)
    else:
       alms = hp.smoothalm(alms,beam_window=bl)
       #pass
    # Normalise and tranform into map space
    galmap = hp.alm2map(alms,nside)/calfactor/beam_solid_angle # Jy/beam 

    return galmap

def Cat2Alm(Catalogue,lmax,verbose=False):
    # Calculate l, m and alm indices
    Nalm = hp.Alm.getsize(lmax)
    idx = np.arange(Nalm,dtype=int)
    l, m = hp.Alm.getlm(lmax,idx)

    # This calculates the source locations in Alm space using GSL legendre libraries
    if verbose:
        print('Calculating alms...')
    sourceInfo = np.array([Catalogue['GLON']*np.pi/180.,
                           (90-Catalogue['GLAT'])*np.pi/180.,
                           Catalogue['FLUX']]).T
    d_alms = gsl_funcs.precomp_harmonics_onethread(sourceInfo,idx.size,np.max(l))
    if verbose:
        print('...Done')
    alms = d_alms[:,0] - 1j*d_alms[:,1]
    alms[np.isfinite(alms) == False] = 0 + 1j*0
    del sourceInfo
    del d_alms

    return alms


def main(params,catalogues,use_corr_flux=False):
    """
    """

    galmap = AlmSpace(catalogues[params['catalogue']],nside=int(params['nside']),
                      fwhm=params['fwhm'],
                      beam_model=np.loadtxt(params['beam_model']))


    if use_corr_flux:
        prefix = 'flux_corrected_'
    else:
        prefix = ''


    if not os.path.exists(f'{params["output_map_dir"]}'):
        os.makedirs(f'{params["output_map_dir"]}')
        
    hp.write_map(f'{params["output_map_dir"]}/{prefix}{params["output_map_file"]}', galmap,
                 overwrite=True)
