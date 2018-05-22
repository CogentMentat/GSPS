# coding: utf-8
"""
Functions for creating a generative system on a curated data set.

Author(s): Alexander TJ Barron
Date created: 2014-09-07
"""

from collections import defaultdict
from itertools import combinations, groupby
from functools import reduce
import numpy as np

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
            for c in combinations(range(self.maxdepth), r):
                yield c

    def gen_inds(self, gen_stack, ind_stack, rind, increasing):
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
          increasing (bool, optional): Mask generation direction (False for
            decreasing)

        Yields:
          if increasing is not specified:
            tuple: indices in the 1st and 2nd dimensions
          if increasing is True or False:
            tuple: generating indices in the 1st and 2nd dimensions
            tuple: generated indices in the 1st and 2nd dimensions

        """

        # Recursive generator.
        while rind > -1:
            try:
                ind_stack[rind] = next(gen_stack[rind])
                if rind == self.vnum - 1:

                    iindspre = [[i]*len(inds) for i, inds in \
                                enumerate(ind_stack)]
                    jindspre = [list(inds) for inds in ind_stack]

                    if not any([self.maxdepth - 1 in inds for inds in jindspre]):
                        continue # For condition m2 on p. 100.

                    if increasing == None: # non-generative system
                        iinds = reduce(lambda x, y: list(x) + list(y),
                                       iindspre)
                        jinds = reduce(lambda x, y: list(x) + list(y),
                                       jindspre)
                        yield [iinds, jinds]

                    if increasing != None: # Generative system

                        if not any([len(inds) > 1 for inds in iindspre]):
                            continue # Won't create generating/ed indices.

                        # Split mask into generating and generated indices.
                        generating_iindspre = []
                        generated_iindspre = []
                        generating_jindspre = []
                        generated_jindspre = []
                        for rowinds, colinds in zip(iindspre, jindspre):
                            if len(rowinds) > 1:
                                # This row has more than one entry, so should
                                # be split.
                                if increasing: # Increasing system over backdrop.
                                    generating_iindspre.append(rowinds[:-1])
                                    generating_jindspre.append(colinds[:-1])
                                    generated_iindspre.append([rowinds[-1]])
                                    generated_jindspre.append([colinds[-1]])
                                else: # Decreasing system over backdrop.
                                    generating_iindspre.append(rowinds[1:])
                                    generating_jindspre.append(colinds[1:])
                                    generated_iindspre.append([rowinds[0]])
                                    generated_jindspre.append([colinds[0]])
                            else:
                                # This row has only one entry, so has no
                                # generating states.
                                generated_iindspre.append(rowinds)
                                generated_jindspre.append(colinds)

                        try:
                            generating_iinds = reduce(lambda x, y: list(x) + list(y),
                                           generating_iindspre)
                        except:
                            raise Exception("generating_iindspre is []")
                        generating_jinds = reduce(lambda x, y: list(x) + list(y),
                                       generating_jindspre)
                        generated_iinds = reduce(lambda x, y: list(x) + list(y),
                                       generated_iindspre)
                        generated_jinds = reduce(lambda x, y: list(x) + list(y),
                                       generated_jindspre)

                        yield [[generated_iinds, generated_jinds], \
                                [generating_iinds, generating_jinds]]


                else:
                    rind += 1
                    self.gen_inds(gen_stack, ind_stack, rind, increasing)

            except StopIteration:
                gen_stack[rind] = self._powerset()
                rind -= 1
                self.gen_inds(gen_stack, ind_stack, rind, increasing)

    def mask_generator(self, increasing=None):
        """
        Generate all possible mask indices.

        Args:
          increasing (bool, optional): True for yielding generating and
            generated states while increasing the ordered support index of the
            mask or False for decreasing.  Default is None for making
            non-generative masks.
        Returns:
          generator: yielding tuples of indices for masking a numpy array

        """

        gen_stack = [self._powerset() for j in range(self.vnum)]
        ind_stack = [0]*self.vnum
        rind = 0
        return self.gen_inds(gen_stack, ind_stack, rind, increasing)


class Generative_system(object):
    """
    Generative system for crisp data.

    Args:
      maskinds (list): i and j indices defining a mask on the data system

    """

    def __init__(self, generated_maskinds, generating_maskinds):

        maskinds = [generating_maskinds[0] + generated_maskinds[0],
                    generating_maskinds[1] + generated_maskinds[1]]

        self.maskinds = maskinds
        self.generating_maskinds = generating_maskinds
        self.generated_maskinds = generated_maskinds

        generatednum = len(generated_maskinds[0])
        generatingnum = len(generating_maskinds[0])
        self.generating_sampinds = range(generatingnum)
        self.generated_sampinds = range(generatingnum, generatingnum + generatednum)

    def get_generation_probabilities(self, data):
        """
        Get the probabilistic behavior functions.

        NOTE: if the mask variable dimension (0) is not as large as the data
        variable dimension, the mask will scan across the top, then drop down
        a row and scan across again, etc. until the bottom of the mask covers
        the bottom of the data matrix in a final scan.

        Args:
          data (numpy.ndarray): curated data of n rows by t columns, where n
            is the number of variables in the data system and t is the number
            of support states.

        """

        colsupport_len = data.shape[1]
        maxcoldepth = max(self.maskinds[1]) + 1
        colscan_num = colsupport_len - maxcoldepth + 1

        rowsupport_len = data.shape[0]
        maxrowdepth = max(self.maskinds[0]) + 1
        rowscan_num = rowsupport_len - maxrowdepth + 1

        # Record all sampling variables for the mask.
        d_samp_cnt = defaultdict(int)
        gtng_sampvars = set()
        gtd_sampvars = set()
        for rs in range(rowscan_num):
            for cs in range(colscan_num):

                # Full mask counts.
                sampinds = [[j+rs for j in self.maskinds[0]],
                            [j+cs for j in self.maskinds[1]]]
                sample = data[sampinds]
                d_samp_cnt[tuple(sample)] += 1

                ## Generating and generated variables observed.
                generatinginds = [[j+rs for j in self.generating_maskinds[0]],
                                    [j+cs for j in self.generating_maskinds[1]]]
                gtng_sampvars.add(tuple(data[generatinginds]))
                generatedinds = [[j+rs for j in self.generated_maskinds[0]],
                                    [j+cs for j in self.generated_maskinds[1]]]
                gtd_sampvars.add(tuple(data[generatedinds]))

        tot_count = 0
        for c in d_samp_cnt.values():
            tot_count += c
        tot_count = float(tot_count)

        # Probabilistic behavior function array, split into generating,
        # generated, and probability lists (same order).  This keeps variable
        # states as tuples with integer values.  Placing generated and
        # generating states in the same array as their probabilities forces
        # state values to be floats.  This is one solution.

        generated_varstates = []
        generating_varstates = []
        maskstate_probs = []
        for sample, count in d_samp_cnt.items():
            #generated_varstates.append(tuple(sample[self.generated_sampinds]))
            generated_varstates.append(tuple([sample[si] for si in self.generated_sampinds]))
            #generating_varstates.append(tuple(sample[self.generating_sampinds]))
            generating_varstates.append(tuple([sample[si] for si in self.generating_sampinds]))
            maskstate_probs.append(count/tot_count)

        # Get generating marginal probabilities.
        d_gtngsampvar_prob = {}
        for gen_sampvar in gtng_sampvars:
            p_gen_sampvar = 0.
            for gtng, maskstate_prob in \
                    zip(generating_varstates, maskstate_probs):
                if gtng == gen_sampvar:
                    p_gen_sampvar += maskstate_prob
            d_gtngsampvar_prob[gen_sampvar] = p_gen_sampvar

        # Get conditional probabilities for generated given generating states.
        d_gtd_gtng_prob = {}
        for gen_sampvar, gtd_sampvar, maskstate_prob in \
                zip(generating_varstates, generated_varstates, maskstate_probs):
            d_gtd_gtng_prob[(gtd_sampvar, gen_sampvar)] = \
                    maskstate_prob/d_gtngsampvar_prob[gen_sampvar]

        self.generated_varstates = generated_varstates
        self.generating_varstates = generating_varstates
        self.maskstate_probs = maskstate_probs
        self.generated_sampvars = gtd_sampvars
        self.generating_sampvars = gtng_sampvars
        self.d_gtngsampvar_prob = d_gtngsampvar_prob
        self.d_gtd_gtng_prob = d_gtd_gtng_prob

    def get_uncertainty(self):
        """Calculate generative uncertainty."""
        uncertainty = 0
        for gtng in self.generating_sampvars:

            inner_summands = []
            for gtd in self.generated_sampvars:
                gtd_gtng = (gtd, gtng)
                if gtd_gtng in self.d_gtd_gtng_prob:
                    inner_summands.append(self.d_gtd_gtng_prob[gtd_gtng])
            innersum = sum([k*np.log2(k) for k in inner_summands])

            uncertainty -= self.d_gtngsampvar_prob[gtng]*innersum

        self.uncertainty = uncertainty

    def get_complexity(self):
        self.complexity = len(self.maskinds[0])

    def compute_behavior_system(self, data):
        self.get_generation_probabilities(data)
        self.get_uncertainty()
        self.get_complexity()

def find_admissible_behavior_systems(generative_systems):
    """
    Isolate admissible behavior systems, using the methodology in pp. 115-120
    in ASPS.

    Args:
      generative_systems: list of things that can be indexed to get complexity
        and uncertainty, using `thing.complexity`, etc.

    """

    complex_sorted = groupby(sorted(generative_systems,
                                    key=lambda x: x.complexity, reverse=True),
                             lambda x: x.complexity)

    admissible = []
    admissible_uncertainties = [np.inf]
    for complexity, complexity_block in complex_sorted:

        unc_sorted = sorted(complexity_block, key=lambda x: x.uncertainty,
                reverse=False)
        lowest_unc = unc_sorted[0].uncertainty

        ## Establish upper bound of complexity in complexity-uncertainty front.
        if any([lowest_unc <= unc for unc in admissible_uncertainties]):
            admissible = [unc_sorted[0]]
            admissible_uncertainties = [lowest_unc]
        else:
            admissible.append(unc_sorted[0])
            admissible_uncertainties.append(lowest_unc)

    return admissible
