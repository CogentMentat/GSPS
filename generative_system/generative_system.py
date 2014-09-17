"""
Functions for creating a generative system on a curated data set.

Author(s): Alexander TJ Barron (and potentially others)
Date created: 2014-09-07
"""

import itertools as it
from collections import defaultdict
import pandas as pd
import numpy as np
import pdb

class M_matrix(object):
    """
    Full M-matrix defined by desired maximum mask depth and number of
    variables in a data system.  Introduced on p. 100 in ASPS.

    Args:
      vnum (int): number of variables in the data system
      maxdepth (int): number of consecutive support states the M-matrix should
        span on the data system

    """


    def __init__(self, vnum, maxdepth):
        self.vnum = vnum
        self.maxdepth = maxdepth
    
    def _powerset(self):
        for r in range(1,self.maxdepth + 1):
            for c in it.combinations(xrange(self.maxdepth), r):
                yield c
        
    def gen_inds(self, gen_stack, ind_stack, rind):
        """
        Generates lists of indices for masking a numpy array of data.
        Intended to be called from the mask_generator method.

        Index lists correspond to meaningful submasks as defined on p. 100 of
        ASPS.

        Args:
          gen_stack (list): list of _powerset generators, length equal to
            the number of variables in the generative system
          ind_stack (list): list of filler values to be replaced by tuples
            of indices, length equal to gen_stack's length

        Yields:
          tuple: indices in the 1st and 2nd dimensions

        """

        while rind > -1:
            try:
                ind_stack[rind] = gen_stack[rind].next()
                if rind == self.vnum - 1:
                    iindspre = [[i]*len(inds) for i, inds in enumerate(ind_stack)]
                    iinds = reduce(lambda x, y: list(x) + list(y), iindspre)
                    jinds = reduce(lambda x, y: list(x) + list(y), ind_stack)
                    if self.maxdepth - 1 not in jinds:
                        pass # for condition m2 on p. 100
                    else:
                        yield [iinds, jinds]
                else:
                    rind += 1
                self.gen_inds(gen_stack, ind_stack, rind)
            except StopIteration:
                gen_stack[rind] = self._powerset()
                rind -= 1
                self.gen_inds(gen_stack, ind_stack, rind)

    def mask_generator(self):
        """
        Generate all possible mask indices.
        
        Returns:
          generator: yielding tuples of indices for masking a numpy array
        
        """

        gen_stack = [self._powerset() for j in range(self.vnum)]
        ind_stack = [0]*self.vnum
        rind = 0
        return self.gen_inds(gen_stack, ind_stack, rind)

class Generative_system(object):
    """
    Generative system for crisp data.

    Args:
      maskinds (list): i and j indices defining a mask on the data system

    """

    def __init__(self, maskinds):

        self.maskinds = maskinds

    def prob_bf(self, data):
        """
        Probabilistic behavior function.

        Args:
          data (numpy.ndarray): curated data of n rows by t columns, where n
            is the number of variables in the data system and t is the number
            of support states.

        Returns:
          pandas.dataframe: a s by m+1 df, where s is the number of observed
            sampling variable tuples covered by the mask and m is the number
            of sampling variables in the mask.  The final column is the
            observed probability of the sampling variable tuple in the data
            using the maximum likelihoood estimator.

        """

        mwidth = max(self.maskinds[1]) - min(self.maskinds[1]) + 1
        support_len = data.shape[1]
        sample_num = support_len - mwidth + 1

        # the defaultdict will learn occurring samples as the mask scans
        # across the data
        d_samp_cnt = defaultdict(int)
        for s in xrange(sample_num):
            # scan the mask across the ordered support
            sampinds = [self.maskinds[0],
                        [j+s for j in self.maskinds[1]]]
            sample = data[sampinds]
            d_samp_cnt[tuple(sample)] += 1

        tot_count = 0
        for c in d_samp_cnt.itervalues():
            tot_count += c
        tot_count = float(tot_count)
        sample_len = len(self.maskinds[0])
        sample_cnt = len(d_samp_cnt)
        # probabilistic behavior function array
        dtype = [('s{}'.format(i),'u2') for i in range(sample_len)] + \
                [('f', 'f4')]
        a_bf = np.array(np.zeros(shape=(sample_cnt,),
                        dtype=dtype))
        for s, (sample, count) in enumerate(d_samp_cnt.iteritems()):
            a_bf[s] = tuple(list(sample) + [count/tot_count])

        return pd.DataFrame(a_bf)
