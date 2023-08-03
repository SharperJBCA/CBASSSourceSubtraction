import numpy as np
import healpy as hp
import os
import h5py
from dataclasses import dataclass, field
from astropy.io import fits
from matplotlib import pyplot

#set_num_threads(2)

def rotate(x,y,coord=['G','C']):
    rot = hp.rotator.Rotator(coord=coord)

    yr, xr = rot((90-y)*np.pi/180., x*np.pi/180.)
    xr *= 180./np.pi
    yr  = (np.pi/2. - yr)*180./np.pi

    return xr, yr

@dataclass 
class Catalogue: 

    name : str = ''
    glon : np.ndarray = field(default_factory=lambda : np.zeros(1))
    glat : np.ndarray = field(default_factory=lambda : np.zeros(1))
    flux : np.ndarray = field(default_factory=lambda : np.zeros(1))
    eflux: np.ndarray = field(default_factory=lambda : np.zeros(1))
    flag : np.ndarray = field(default_factory=lambda : np.zeros(1,dtype=bool))

    def __add__(self,other):
        """Add two catalogues together"""
        new = Catalogue()
        new.name = self.name + '_' + other.name
        new.glon = np.concatenate([self.glon,other.glon])
        new.glat = np.concatenate([self.glat,other.glat])
        new.flux = np.concatenate([self.flux,other.flux])
        new.eflux = np.concatenate([self.eflux,other.eflux])
        new.flag = np.concatenate([self.flag,other.flag])
        return new
    
    def __repr__(self) -> str:
        s = f'Catalogue: {self.name}\n'
        s += f'Number of sources: {self.size}\n'
        s += f'Flagged sources: {self.flag.sum()}\n'
        return s
        
    def __call__(self,fitsfile):
        self.run(fitsfile)
        self.clean_nan() 
        self.remove_flagged()

    def clean_nan(self):
        mask = np.isnan(self.glon) | np.isnan(self.glat) | np.isnan(self.flux) | np.isnan(self.eflux) | (self.flux <= 0)
        self.glon = self.glon[~mask]
        self.glat = self.glat[~mask]
        self.flux = self.flux[~mask]
        self.eflux = self.eflux[~mask]
        self.flag = self.flag[~mask]

    def remove_flagged(self):
        """Remove flagged sources from catalogue"""
        mask = self.flag == False
        self.glon = self.glon[mask]
        self.glat = self.glat[mask]
        self.flux = self.flux[mask]
        self.eflux = self.eflux[mask]
        self.flag = self.flag[mask]

    def run(self,fitsfile):
        pass

    @property 
    def size(self):
        return len(self.glon)
    
    @property
    def thetaphi(self):
        print(np.nanmin(self.glat),np.nanmax(self.glat))
        return (90-self.glat)*np.pi/180., self.glon*np.pi/180.

    def remove_sources(self,mask):
        """Remove sources from catalogue"""
        self.glon = self.glon[mask]
        self.glat = self.glat[mask]
        self.flux = self.flux[mask]
        self.eflux = self.eflux[mask]
        self.flag = self.flag[mask]

    def update_sources(self,mask,flux,eflux):
        """Update fluxes of sources"""
        self.flux[mask] = flux
        self.eflux[mask] = eflux

    def write_file(self,filename):
        """Write to HDF5 file"""
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        h = h5py.File(filename,'w')
        grp = h.create_group(self.name)
        grp.create_dataset('GLON',data=self.glon)
        grp.create_dataset('GLAT',data=self.glat)
        grp.create_dataset('FLUX',data=self.flux)
        grp.create_dataset('eFLUX',data=self.eflux)
        grp.create_dataset('FLAG',data=self.flag)
        h.close() 

    def load_file(self,filename, name='none'):
        """Read from HDF5 File"""
        h = h5py.File(filename,'r')
        grp = h[name]
        self.glon = grp['GLON'][...]
        self.glat = grp['GLAT'][...]
        self.flux = grp['FLUX'][...]
        self.eflux = grp['eFLUX'][...]
        self.flag = grp['FLAG'][...]
        h.close()

@dataclass 
class Mingaliev(Catalogue):

    name : str = 'Mingaliev'

    def run(self,fitsfile):

        hdu = fits.open(fitsfile,memmap=False)
        data = hdu[1].data

        self.glon = data['_Glon']
        self.glat = data['_Glat']
        
        self.flux = data['S3_9GHz']*(3.9/4.76)**data['Sp-Index']
        self.eflux = data['e_S3_9GHz']*(3.9/4.76)**data['Sp-Index']
        self.flag = np.zeros_like(self.glon,dtype=bool)
        hdu.close()

@dataclass 
class GB6(Catalogue):

    name : str = 'GB6'

    def run(self,fitsfile):

        hdu = fits.open(fitsfile,memmap=False)
        data = hdu[1].data
        self.glon = data['_Glon']
        self.glat = data['_Glat']
        
        self.flux = data['Flux']*1e-3
        self.eflux = data['e_Flux']*1e-3

        self.flag = np.array([True if (data['Eflag'][i] == 'E') else False for i in range(len(data['Eflag']))])

        hdu.close()

@dataclass 
class PMN(Catalogue):

    name : str = 'PMN'

    def run(self,fitsfile):

        hdu = fits.open(fitsfile,memmap=False)
        data = hdu[1].data
        self.glon = data['_Glon']
        self.glat = data['_Glat']
        
        self.flux = data['Flux']*1e-3
        self.eflux = data['e_Flux']*1e-3
        self.flag = np.array([True if (data['Xflag'][i] == 'X') else False for i in range(len(data['Xflag']))])
        hdu.close()

@dataclass
class CBASS(Catalogue):

    name : str = 'CBASS'

    def run(self,fitsfile):

        data = np.loadtxt(fitsfile,skiprows=1,usecols=[3,4,7,8])
        self.glon = data[:,0]
        self.glat = data[:,1]
        self.flux = data[:,2]*0.7
        self.eflux = data[:,3]
        rot = hp.rotator.Rotator(coord=['C','G'])
        self.glat, self.glon = rot(*self.thetaphi) 
        self.glon = np.mod(self.glon*180/np.pi,360)
        self.glat = (np.pi/2.-self.glat)*180/np.pi
        self.flag = np.zeros_like(self.glon,dtype=bool)

def mingaliev(filename):
    """
    """
    columns = {
        'GLON':[], 
        'GLAT':[],
        'FLUX':[],
        'eFLUX':[]
    }
        
        
    with open(filename,'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            glon, glat, flux, eflux = line.split()
            columns['GLON'] += [float(glon)]
            columns['GLAT'] += [float(glat)]
            columns['FLUX'] += [float(flux)/1000.]
            columns['eFLUX']+= [float(eflux)]

    columns = {k:np.array(v) for k,v in columns.items()}
    return columns

    
def RichardCBASS(filename):
    """
    """
    columns = {'GLON':[],
               'GLAT':[],
               'FLUX':[],
               'eFLUX':[],
               'SNR':[],
               'CRATES':[],
               'MASKED':[],
               'NAME':[]}

    with open(filename,'r') as f:
        for line in f:
            line = line.strip()
            if line[0] == '#':
                continue
            name,ra, dec, gl, gb, flux, eflux ,dflux,deflux, snr,matches,masked = line.split()

            masked = int(masked)
            
            ra, dec = float(ra), float(dec)
            if ra < 0:
                ra = 360+ra
            if float(flux) < 0:
                continue
            ra,dec = float(ra), float(dec)
            glon,glat = rotate(ra,dec,coord=['C','G'])
            columns['GLON'] += [float(glon)]
            columns['GLAT'] += [float(glat)]
            columns['FLUX'] += [float(dflux)/1000.]
            columns['eFLUX']+= [float(deflux)/1000.]
            columns['SNR']+= [float(snr)]
            columns['CRATES'] += ['C' in matches]
            columns['MASKED'] += [masked == 1]
            columns['NAME'] += [name]

    columns = {k:np.array(v) for k,v in columns.items()}
    return columns
    
def PCCS(filename):
    """
    """
    columns = {
            'GLON':[], 
            'GLAT':[],
            'FLUX':[],
            'eFLUX':[]
        }
    with open(filename,'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            glon, glat, flux, eflux = line.split()
            if float(flux) < 0:
                continue
            columns['GLON'] += [float(glon)]
            columns['GLAT'] += [float(glat)]
            columns['FLUX'] += [float(flux)/1e3]
            columns['eFLUX']+= [float(eflux)/1e3]

    columns = {k:np.array(v) for k,v in columns.items()}
    return columns

def gb6(filename):
    """
    """
    columns = {'GLON':[],
               'GLAT':[],
               'FLUX':[],
               'BEAM':[]}
    with open(filename,'r') as f:
        for line in f:
            if line[0] == '#':
                continue

            glat = float(line[0:5])
            glon = float(line[6:6+5])
            flux = float(line[12:12+5])
            eflux = float(line[18:18+4])
            if line[23] == 'E':
                extended = True
            else:
                extended = False
            if line[25] == 'W':
                weak = True
            else:
                weak = False
            if line[27] == 'C':
                confused = True
            else:
                confused = False
            Major = float(line[29:29+5])
            Minor = float(line[35:35+5])

            if extended | confused:
                continue
            columns['GLON'] += [float(glon)]
            columns['GLAT'] += [float(glat)]
            columns['FLUX'] += [float(flux)*float(Major)*float(Minor)/1000.]
            columns['BEAM']+= [float(Major)*float(Minor)]

        columns = {k:np.array(v) for k,v in columns.items()}
        return columns

def pmn(filename):
    """
    """
    columns = {
            'GLON':[], 
            'GLAT':[],
            'RA':[],
            'DEC':[],
            'FLUX':[],
            'eFLUX':[]
        }
        
    with open(filename,'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            glon, glat, ra, dec, flux, eflux = line.split()
            columns['GLON'] += [float(glon)]
            columns['GLAT'] += [float(glat)]
            columns['RA']   += [float(ra)]
            columns['DEC']  += [float(dec)]
            columns['FLUX'] += [float(flux)/1000.]
            columns['eFLUX']+= [float(eflux)/1000.]

    columns = {k:np.array(v) for k,v in columns.items()}
    return columns

def save_catalogues(filename,catalogues):

    ddir = os.path.basename(filename)
    if not os.path.exists(ddir):
        os.makedirs(ddir)

    h = h5py.File(filename,'a')

    for catname, cat in catalogues.items():
        if catname in h:
            grp = h[catname]
        else:
            grp = h.create_group(catname)
        for dname,data in cat.items():
            if dname in grp:
                del grp[dname]
            try:
                dset = grp.create_dataset(dname,data=data)
            except TypeError:
                continue

    h.close()

def load_catalogues(filename):
    h = h5py.File(filename,'r')

    catalogues = {}
    for catname, cat in h.items():
        catalogue = {}
        for dname, data in cat.items():
            catalogue[dname] = data[...]
        catalogues[catname] = catalogue

    h.close()
    return catalogues

functions = {'Mingaliev':Mingaliev,
             'GB6':GB6,
             'PMN':PMN,
             'PCCS':PCCS,
             'CBASS':RichardCBASS}

##@njit(nogil=True)
def AngularSeperation(phi0,theta0,phi1,theta1):
    """
    phi - longitude parameters
    theta- latitude parameters
    """    
    c = np.pi/180.

    mid = np.sin(theta0*c)*np.sin(theta1*c) + np.cos(theta0*c)*np.cos(theta1*c)*np.cos((phi1-phi0)*c)
    
    return np.arccos(mid)/c

#@njit(nogil=True,parallel=True)
def distance_calc(positions,ncatalogues,min_dist):

    counts = np.zeros((get_num_threads(),positions.shape[1]),dtype=np.int64)
    distance = np.zeros((get_num_threads()))
    d = np.zeros((ncatalogues, positions.shape[1]),dtype=np.int64)
    d += -1
    z = np.zeros((ncatalogues, positions.shape[1]))
    for i in prange(positions.shape[1]):
        for j in range(i,positions.shape[1]):
            if i == j:
                continue
            distance[_get_thread_id()] = AngularSeperation(positions[0,i],
                                                           positions[1,i],
                                                           positions[0,j],
                                                           positions[1,j])
            if distance[_get_thread_id()] < min_dist:
                if (counts[_get_thread_id(),i] > (ncatalogues-1)):
                    continue
                d[counts[_get_thread_id(),i],i] = j
                z[counts[_get_thread_id(),i],i] = distance[_get_thread_id()]
                counts[_get_thread_id(),i] += 1
    return d,z

def cross_check_catalogues(catalogues):

    N = 0
    for i,(k,v) in enumerate(catalogues.items()):
        N += len(v['GLON'])

    positions = np.zeros((2,N))
    fluxes = np.zeros((N))
    catids = np.zeros(N)
    low = 0
    for i,(k,v) in enumerate(catalogues.items()):
        high = low + len(v['GLON'])
        positions[:,low:high] = v['GLON'],v['GLAT']
        fluxes[low:high] = v['FLUX']
        catids[low:high] = i 
        low += len(v['GLON'])

    min_dist = 30./60.**2
    d,z = distance_calc(positions[:,:],len(catalogues.keys()),min_dist)
    

    common_sources = np.sum((d >= 0),axis=0)
    common = np.where((common_sources > 0))[0] # Everywhere there is a matching pair
    not_common = np.where((common_sources == 0))[0]
    data = {k:np.zeros(len(common)+len(not_common)) for k in catalogues['GB6'].keys()}
    data['GLON'][:len(common)]

    for ii,i in enumerate(common):
        select = (d[:,i] != -1) & (catids[d[:,i]] != catids[i])
        if np.sum(select) == 0:
            continue
        fluxes[i] = np.mean([fluxes[i], fluxes[d[select,i]]])
        for j in range(2):
            positions[j,i] = np.mean([positions[j,i],positions[j,d[select,i]]])


    fluxes,idx = np.unique(fluxes,return_index=True)
    positions = positions[:,idx]

    
    return {'COMMON':{'FLUX':fluxes,'GLON':positions[0],'GLAT':positions[1]}}
def main(params):
    """
    Run through the catalogues.
    """

    if params['create_catalogues']:
        catalogues = {}
        for catalogue in params['catalogues']:
            filename = params[catalogue]
            catalogues[catalogue]=functions[catalogue](filename)
        save_catalogues(params['catalogues_database'],catalogues)
    
        common_catalogue = cross_check_catalogues({k:catalogues[k] for k in params['common_catalogues']})
        save_catalogues(params['catalogues_database'],common_catalogue)

    common_catalogue = load_catalogues(params['catalogues_database'])

    return common_catalogue
    
    
    
