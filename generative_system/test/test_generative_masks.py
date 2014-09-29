"""
Testing generative submasks on a toy data set and mask.

Author: Alexander TJ Barron
Date Created: 2014-09-29

"""

import numpy as np

import GSPS.generative_system.generative_system as gs

def test(M, data):
    # TODO: I've checked these generated and generating submasks manually by
    # looking at the below printed output, but maybe a comparison of some
    # counting result Klir most likely presents would also be good.

    for inds_gtd, inds_gng in M.mask_generator(increasing=True):
        print '\n===============\n'
        print 'data:'
        print data
        print '\n'
        print 'generated inds:'
        print inds_gtd
        print 'applied to data:'
        print data[inds_gtd]
        print '\n'
        print 'generating inds:'
        print inds_gng
        print 'applied to data:'
        print data[inds_gng]

def main():

    # Toy data
    data = np.array([[0,1,2],[10,11,12]])
    print '\nToy data:\n'
    print data

    print '\nPrinting generated and generating submasks for each ' \
            'full mask over the toy data set:\n'

    M = gs.M_matrix(2,3)

    test(M, data)

if __name__ == "__main__":
    main()
