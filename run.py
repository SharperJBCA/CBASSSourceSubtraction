import matplotlib 
matplotlib.use('Agg')
import numpy as np
import sys

from SourceSubtraction.Tools import healpix_tools
from SourceSubtraction import Mapper, Catalogues
#, Mapper, FluxCorrection,Subtract
#from cbass.Tools import Parser
import h5py
import healpy as hp
from matplotlib import pyplot
from tqdm import tqdm 


def haversine(theta1, phi1, theta2, phi2):
    """
    Calculate the great circle distance between two points
    on a sphere
    phi = longitude
    theta = latitude 
    """
    # Convert latitude and longitude to spherical coordinates in radians.
    theta1 = np.radians(theta1)
    phi1 = np.radians(phi1)
    theta2 = np.radians(theta2)
    phi2 = np.radians(phi2)

    # Haversine formula
    dtheta = theta2 - theta1
    dphi = phi2 - phi1
    a = np.sin(dtheta/2)**2 + np.cos(theta1) * np.cos(theta2) * np.sin(dphi/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c 

def common_sources(catalogue1, catalogue2,nside_low=64):
    # Do a low resolution pass to find common sources
    m = np.zeros(12*nside_low**2)
    pixels1 = hp.ang2pix(nside_low, *catalogue1.thetaphi)
    m[pixels1] += 1
    pixels2 = hp.ang2pix(nside_low, *catalogue2.thetaphi)
    m[pixels2] += 1

    # Now, compare sources that are within pixels that have more than one source
    common_pixels = np.where(m > 1)[0]
    common_sources1 = np.where(np.in1d(pixels1, common_pixels))[0]
    common_sources2 = np.where(np.in1d(pixels2, common_pixels))[0]

    # Now, find sources in catalogue1 that match sources in catalogue2
    # This is done by finding the closest source in catalogue2 to each source in catalogue1

    # Create mask to remove sources from catalogue1
    mask1 = np.ones(catalogue1.size, dtype=bool)
    # Create mask to update sources from catalogue2
    mask2 = np.ones(catalogue2.size, dtype=bool)
    # Create an array for modified fluxes for catalogue2 
    flux2_array = np.zeros(catalogue2.size)
    eflux2_array = np.zeros(catalogue2.size)
    for i,(glon1,glat1,flux1,eflux1) in enumerate(zip(catalogue1.glon[common_sources1],catalogue1.glat[common_sources1],catalogue1.flux[common_sources1],catalogue1.eflux[common_sources1])):
        dist = haversine(glat1,glon1,catalogue2.glat[common_sources2],catalogue2.glon[common_sources2])
        if np.min(dist)*60 > 1:
            continue
        match = np.argmin(dist)
        flux2 = catalogue2.flux[common_sources2[match]]
        eflux2 = catalogue2.eflux[common_sources2[match]]

        mask1[common_sources1[i]] = False
        mask2[common_sources2[match]] = False 
        flux2_array[common_sources2[match]] = (flux1/eflux1**2 + flux2/eflux2**2)/(1/eflux1**2 + 1/eflux2**2)
        eflux2_array[common_sources2[match]] = np.sqrt(1/(1/eflux1**2 + 1/eflux2**2))

    # Now, remove sources from catalogue1 and update sources from catalogue2
    catalogue1.remove_sources(mask1)
    catalogue2.update_sources(~mask2,flux2_array[~mask2],eflux2_array[~mask2])

    hp.mollview(m)
    hp.graticule()
    pyplot.savefig(f'figures/{catalogue1.name}_{catalogue2.name}_common_sources.png')
    pyplot.close()

    return catalogue1, catalogue2

def merge_catalogues(catalogue1, catalogue2, fwhm=1.,nside_low=64,nside=1024):
    """
    For this we need to calculate the weight to each source in catalogue1 relative to catalogue2
    this weight is a 2D gaussian with a FWHM 
    """

    pixel_area = 4*np.pi/(12*nside**2)
    sigma = fwhm/2.355*np.pi/180.
    for i in tqdm(range(catalogue1.flux.size)):

        dist = haversine(catalogue1.glat[i],catalogue1.glon[i],catalogue2.glat,catalogue2.glon)
        weights = np.exp(-0.5*dist**2/sigma**2)
        catalogue2.flux = catalogue2.flux*(1-weights)


    return catalogue1, catalogue2

def main():
    """
    """

    mask_map = ~hp.read_map('../Masks/scripts/masks/CBASS_80pc_G_1024.fits').astype(bool) 


    defaults = ''
    nside = 1024
    fwhm = 1 
    beam_model_filename = 'AncillaryData/cbass_beam2_nodupes.txt'

    cbass = Catalogues.CBASS()
    cbass('AncillaryData/Catalogues/cbassNorth_sourceCat_v1.3.txt')
    cbass.remove_sources(cbass.flux > 1)
    cbass.mask_declinations(declination_min=-5,declination_max=90)

    mingaliev = Catalogues.Mingaliev()
    mingaliev('AncillaryData/Catalogues/mingaliev2001_RATAN600.fits')
    gb6 = Catalogues.GB6()
    gb6('AncillaryData/Catalogues/gregory1996_gb6.fits')
    pmn = Catalogues.PMN()
    pmn('AncillaryData/Catalogues/griffith1993_pmne.fits')
    pmnt = Catalogues.PMN()
    pmnt('AncillaryData/Catalogues/griffith1993_pmnt.fits')

    mingaliev,gb6 = common_sources(mingaliev,gb6)
    pmn,gb6 = common_sources(pmn,gb6)

    total_catalogue = mingaliev + gb6 + pmn + pmnt
    total_catalogue.clean_nan()
    merge_catalogues(cbass,total_catalogue, fwhm=1.)
    total_catalogue = total_catalogue + cbass
    total_catalogue.clean_nan()
    total_catalogue.mask_map(mask_map)

    galmap = Mapper.pixel_space(total_catalogue.flux, total_catalogue.glon, total_catalogue.glat,nside=nside,
                    fwhm=fwhm)
                    #beam_model=np.loadtxt(beam_model_filename))
    hp.write_map('figures/galmap_test_with_cbass.fits',galmap,overwrite=True)
    hp.mollview(galmap,norm='hist')
    hp.graticule()
    pyplot.savefig('figures/galmap_test_with_cbass.png')
    pyplot.close()
    print(total_catalogue)


if __name__ == "__main__":

    #params = Parser.Parser(sys.argv[1])
    #main()
    
    cbass = hp.read_map('AncillaryData/AWR1_xND12_xAS14_1024_NM20S3M1_G_Offmap_deconvolution.fits')

    srcs  = hp.read_map('figures/galmap_test_with_cbass.fits')
    mask_map = hp.read_map('../Masks/scripts/masks/CBASS_80pc_G_1024.fits')
    mask_map[cbass == hp.UNSEEN] = hp.UNSEEN
    residual = cbass - srcs
    residual[cbass == hp.UNSEEN] = hp.UNSEEN
    residual[residual != hp.UNSEEN] *= 1e3 
    cbass[cbass != hp.UNSEEN] *= 1e3
    mollview = healpix_tools.Mollview()
    mollview(cbass,vmin=-2e1, vmax=1e1)
    mollview.add_colorbar(unit_label='mK')
    mollview.add_grid()
    mollview.contourf(mask_map,levels=[0.5,1],cmap=pyplot.get_cmap('Greys'))
    pyplot.title('Original CBASS')
    pyplot.savefig('figures/with_cbass_original.png')
    #pyplot.show()
    pyplot.close()

    mollview = healpix_tools.Mollview()
    mollview(residual,vmin=-2e1, vmax=1e1)
    mollview.add_colorbar(unit_label='mK')
    mollview.add_grid()
    mollview.contourf(mask_map,levels=[0.5,1],cmap=pyplot.get_cmap('Greys'))
    pyplot.title('Source-subtracted CBASS')
    pyplot.savefig('figures/with_cbass_residual.png')
    #pyplot.show()
    pyplot.close()

    gnomview = healpix_tools.Gnomview(crval=[305.5,57.5])
    gnomview(residual)
    gnomview.add_colorbar(unit_label='K')
    gnomview.add_grid() 
    
    pyplot.savefig('figures/with_cbass_residual_gnom.png')

    sys.exit()
    gls,gbs = np.meshgrid(np.linspace(0,360,10),np.linspace(-90,90,9))
    gls = gls.flatten()
    gbs = gbs.flatten()
    for (gl,gb) in zip(tqdm(gls),gbs):
        gnomview = healpix_tools.Gnomview(crval=[gl,gb])

        try:
            img = gnomview(cbass)
        except ValueError:
            continue
        gnomview.contourf(srcs,levels=[1e-3,2e-3,3e-3,4e-3],cmap=pyplot.get_cmap('Reds'),alpha=0.25)
        gnomview.add_colorbar(unit_label='K')
        pyplot.savefig(f'figures/cutouts/with_cbass_gl{gl:03.0f}_gb{gb:02.0f}_original.png')
        pyplot.close() 

        vmin,vmax = img.get_clim()

        gnomview(residual,vmin=vmin,vmax=vmax)
        gnomview.add_colorbar(unit_label='K')
        pyplot.savefig(f'figures/cutouts/with_cbass_gl{gl:03.0f}_gb{gb:02.0f}_residual.png')
        pyplot.close()
        del gnomview