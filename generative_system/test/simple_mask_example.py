"""
Example of a probabilistic behavior function on a toy data set and mask.

Author: Alexander TJ Barron
Date Created: 2014-09-17

"""

import numpy as np

import GSPS.generative_system.generative_system as gs

def main():

    # triomino-shaped mask
    maskinds = [[0,0,1],[0,1,1]]

    # Toy data of 2 variables and 7 support instances, engineered to have one
    # sample from the triomino-shaped mask be twice as likely as the rest.
    data = np.array([range(5) + [1, 2], [10 + i for i in range(5)] + [0,12]])
    print '\ndata:\n'
    print data

    print '\nleftmost sample using triomino mask:\n'
    print data[maskinds]

    Gs = gs.Generative_system(maskinds)

    # probabilistic behavior function
    pbf = Gs.prob_bf(data)
    print '\nprobabilistic behavior function:\n'
    print pbf

if __name__ == "__main__":
    main()
