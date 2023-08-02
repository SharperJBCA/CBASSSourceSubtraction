import numpy as np
import sys

from SourceSubtraction import Mapper, Catalogues
#, Mapper, FluxCorrection,Subtract
#from cbass.Tools import Parser
import h5py
def main():
    """
    """

    defaults = ''

    mingaliev = Catalogues.Mingaliev()
    mingaliev('AncillaryData/Catalogues/mingaliev2001_RATAN600.fits')
    sys.exit()
    if params['General']['do_catalogues']:
        catalogue = Catalogues.main(params['Catalogues'])
    else:
        catalogue = Catalogues.load_catalogues(params['Catalogues']['catalogues_database'])

    if params['General']['do_mapping']:
        Mapper.main(params['Mapper'],catalogue)
    
    if params['General']['do_fluxcorrection']:
        FluxCorrection.main(params['FluxCorrection'],catalogue)
        
    if params['General']['do_sourcesubtraction']:
        Subtract.main(params['Subtract'])



if __name__ == "__main__":

    #params = Parser.Parser(sys.argv[1])
    main()
