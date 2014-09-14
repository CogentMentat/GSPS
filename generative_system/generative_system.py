"""
Functions for creating a generative system on a curated data set.

Author(s): Alexander TJ Barron (and potentially others)
Date created: 2014-09-07
"""

import itertools as it
import pdb

class Mask(object):

    def __init__(self, vnum, maxdepth):
        self.vnum = vnum
        self.maxdepth = maxdepth
    
    def _powerset(self):
        for r in range(1,self.maxdepth + 1):
            for c in it.combinations(xrange(self.maxdepth), r):
                yield c
        
    def gen_rowinds(self, gen_stack, ind_stack, rind):
        """
        Generates tuples of indices for masking a numpy array of data.
        Intended to be called from the mask_generator method.

        Args:
          gen_stack (list): list of _powerset generators, length equal to
            the number of variables in the generative system
          ind_stack (list): list of filler values to be replaced by tuples
            of indices, length equal to gen_stack's length

        Yields:
          tuple: indices in the 1st and 2nd dimensions, respectively

        """

        while rind > -1:
            try:
                ind_stack[rind] = gen_stack[rind].next()
                if rind == self.vnum - 1:
                    iindspre = [[i]*len(inds) for i, inds in enumerate(ind_stack)]
                    iinds = reduce(lambda x, y: list(x) + list(y), iindspre)
                    jinds = reduce(lambda x, y: list(x) + list(y), ind_stack)
                    yield iinds, jinds
                else:
                    rind += 1
                self.gen_rowinds(gen_stack, ind_stack, rind)
            except StopIteration:
                gen_stack[rind] = self._powerset()
                rind -= 1
                self.gen_rowinds(gen_stack, ind_stack, rind)

    def mask_generator(self):
        """
        Generate all possible mask indices.
        
        Returns:
          generator: yielding tuples of indices for masking a numpy array
        
        """

        gen_stack = [self._powerset() for j in range(self.vnum)]
        ind_stack = [0]*self.vnum
        rind = 0
        return self.gen_rowinds(gen_stack, ind_stack, rind)
