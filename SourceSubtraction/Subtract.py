import numpy as np
import healpy as hp
import os

def main(params):
    """
    """
    cbass_map,hdr = hp.read_map(params['Subtract']['cbass_map'],[0,1,2],h=True)
    cbass_nside = hp.npix2nside(cbass_map[0].size)
    source_map = hp.read_map('{}/{}'.format(params['Mapper']['output_map_dir'],
                                            params['Mapper']['output_map_file']))

    source_nside = hp.npix2nside(source_map.size)

    if source_nside != cbass_nside:
        source_map = hp.ud_grade(source_map,cbass_nside)

    resid_map = cbass_map[0]-source_map

    mask = (cbass_map[0] == hp.UNSEEN)
    resid_map[mask] = hp.UNSEEN

    if not os.path.exists(os.path.dirname(params['Subtract']['output_map'])):
        os.makedirs(os.path.dirname(params['Subtract']['output_map']))
        
    hp.write_map(params['Subtract']['output_map'],[resid_map,cbass_map[1],cbass_map[2]])
