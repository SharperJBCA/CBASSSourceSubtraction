import numpy as np
from SourceSubtraction.Tools import Coordinates
from SourceSubtraction.SourceSubtraction import Mapper
from scipy.optimize import minimize
import healpy as hp
from matplotlib import pyplot
import h5py

frequency = 4.76 # GHz
kb = 1.3806503e-23
c = 2.99792458e8
nu = frequency*1e9
beam_solid_angle = 1.133 * (np.pi/180.)**2
calfactor = 2 * kb * (nu/c)**2 * 1e26

nside = 512
beam_model = np.loadtxt('AncillaryData/cbass_beam2.txt')
lmax = 4*nside
dtheta = np.abs(beam_model[0,0]-beam_model[1,0])*np.pi/180.
beam_solid_angle = 2*np.pi*np.sum(beam_model[:,1]*np.sin(beam_model[:,0]*np.pi/180.))*dtheta
bl = hp.beam2bl(beam_model[:,1],beam_model[:,0]*np.pi/180.,lmax)

def model(P,x):
    alms = Mapper.Cat2Alm({'GLON':x[0],
                            'GLAT':x[1],
                            'FLUX':x[2]*P[:-1]},lmax)
    alms = hp.smoothalm(alms,beam_window=bl)
    gmap = hp.alm2map(alms,nside)/calfactor/beam_solid_angle # Jy/beam 
    gmap = gmap + P[-1]
    
    return gmap

def error(P,x,y,pixels):
    chi2 = np.sum((y[pixels]-model(P,x)[pixels])**2)
    return chi2
    


from tqdm import tqdm
def main(params,catalogue):

    #cbass_map = hp.ud_grade(hp.read_map(params['General']['cbass_sourcemap']),nside)/1000.    
    cbass_map = Mapper.AlmSpace(catalogue[params['FluxCorrection']['cbass_catalogue']],
                                nside=int(params['Mapper']['nside']),
                                fwhm=params['Mapper']['fwhm'],
                                beam_model=np.loadtxt(params['Mapper']['beam_model']))
    srcx = catalogue['COMMON']['GLON']
    srcy = catalogue['COMMON']['GLAT']
    srcf = catalogue['COMMON']['FLUXES']
    flux_corrections = np.ones((srcf.size))


    select = (catalogue['CBASS']['FLUX'] > 1) # 1Jy or brighter only
    for k in catalogue['CBASS'].keys():
        catalogue['CBASS'][k] = catalogue['CBASS'][k][select]


    srt = np.argsort(catalogue['CBASS']['FLUX'])[::-1]
    for i,(x,y,s) in enumerate(zip(tqdm(catalogue['CBASS']['GLON'][srt]),catalogue['CBASS']['GLAT'][srt],catalogue['CBASS']['FLUX'][srt])):
        if np.abs(y) < 15:
            continue
        print(x,y)
        distance = Coordinates.AngularSeperation(x,y,srcx,srcy)

        select = (distance < 0.6)
        pixels = hp.query_disc(nside,hp.ang2vec((90-y)*np.pi/180.,x*np.pi/180.),0.5*np.pi/180.)

        if len(srcx[select]) == 0:
            continue
        initial = [1]*len(srcx[select])+[0]
        soln = minimize(error,initial,args=([srcx[select],srcy[select],srcf[select]], cbass_map, pixels),tol=1e-4)

        flux_corrections[select] = soln.x[:-1]
        
        h = h5py.File('AncillaryData/FluxCorrections.hdf5','a')
        if 'flux_corrections' in h:
            del h['flux_corrections']
        
        h.create_dataset('flux_corrections',data=flux_corrections)
        h.close()
        
