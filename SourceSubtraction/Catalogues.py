import numpy as np
import healpy as hp
import os
import h5py
from dataclasses import dataclass, field
from astropy.io import fits

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
    galactic_flag : np.ndarray = field(default_factory=lambda : np.zeros(1,dtype=bool))

    def __add__(self,other):
        """Add two catalogues together"""
        new = Catalogue()
        new.name = self.name + '_' + other.name
        new.glon = np.concatenate([self.glon,other.glon])
        new.glat = np.concatenate([self.glat,other.glat])
        new.flux = np.concatenate([self.flux,other.flux])
        new.eflux = np.concatenate([self.eflux,other.eflux])
        new.galactic_flag = np.concatenate([self.galactic_flag,other.galactic_flag])
        return new

    @property 
    def size(self):
        return len(self.glon)
    
    def __call__(self,fitsfile):
        pass 

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
        grp.create_dataset('GALACTIC_FLAG',data=self.galactic_flag)
        h.close() 

    def load_file(self,filename, name='none'):
        """Read from HDF5 File"""
        h = h5py.File(filename,'r')
        grp = h[name]
        self.glon = grp['GLON'][...]
        self.glat = grp['GLAT'][...]
        self.flux = grp['FLUX'][...]
        self.eflux = grp['eFLUX'][...]
        self.galactic_flag = grp['GALACTIC_FLAG'][...]
        h.close()

@dataclass 
class Mingaliev(Catalogue):

    name : str = 'Mingaliev'

    def __call__(self,fitsfile):

        hdu = fits.open(fitsfile,memmap=False)
        data = hdu[1].data
        print(hdu[1].columns)

        self.glon = data['_Glon']
        self.glat = data['_Glat']
        s3flux = data['S3_9']*(3.9/4.76)**data['Sp-Index']
        s7flux = data['S7_7']*(7.7/4.76)**data['Sp-Index']
        es3flux = data['e_S3_9']*(3.9/4.76)**data['Sp-Index']
        es7flux = data['e_S7_7']*(7.7/4.76)**data['Sp-Index']
        self.flux = (s3flux/es3flux**2+s7flux/es7flux**2)/(1./es3flux**2+1./es7flux**2)

        hdu.close()


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

def GB6(filename):
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

def PMN(filename):
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
    
    
    
