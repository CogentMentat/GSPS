"""
Test the generatime system using the GSPS Applied Linear Regression case study
data.

Author: Alexander TJ Barron
Date Created: 2014-09-24

"""

import numpy as np
from argparse import ArgumentParser

import GSPS.generative_system.generative_system as gs
from GSPS.generative_system.data_handling import load_data

def main(args):

    datapath = args.datapath

    data = load_data(datapath, transpose=True)
    vnum, supportnum = data.shape

    print data
    print data.shape

    # The example calls for a mask depth of 1.
    maskinds = [range(vnum),[0]*vnum]

    Gs = gs.Generative_system(maskinds)

    # probabilistic behavior function
    pbf = Gs.prob_bf(data)
    print '\nprobabilistic behavior function:\n'
    print pbf

if __name__ == "__main__":
    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('datapath',
                        help='Path to data.')
    args = parser.parse_args()

    main(args)
