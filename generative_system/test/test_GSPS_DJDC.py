"""
Test the generative system using the GSPS Dow Jones Index case study data.

Author: Alexander TJ Barron
Date Created: 2014-09-24

"""

from argparse import ArgumentParser

import GSPS.generative_system.generative_system as gs
from GSPS.generative_system.data_handling import load_data

def main(args):

    datapath = args.datapath

    data = load_data(datapath, transpose=True)
    print data.shape
    vnum, supportnum = data.shape

    # The example used a mask depth of 5, but the submasks were limited to the
    # full-depth case.
    depth = 5
    maskinds = [[0]*depth,range(depth)]

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
