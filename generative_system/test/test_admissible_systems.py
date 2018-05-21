# coding: utf-8
"""
Test admissible systems procedure.

Author: Alexander TJ Barron
Date created: 2018-03-26

"""

import pdb

import numpy as np

from GSPS.generative_system.generative_system import \
        M_matrix, Generative_system, find_admissible_behavior_systems

# Random data...
#data = np.random.randint(6, size=120).reshape(3, 40)
#
# ...with some regularity added.
#for k in [0, 10, 15, 25, 35]:
#    deg1inds = [k+j for j in [0, 1, 2, 3]]
#    data[[0, 2, 2, 2], deg1inds] = [0, 1, 2, 3]

data = np.loadtxt('../../test_data/male_gull_class_data_ex2pt5.txt',
                  delimiter=',')

# Mask generator

vnum = 2
maxdepth = 3

MM = M_matrix(vnum, maxdepth)

# Handle generative systems, one per mask.

gensyslist = []
for m in MM.mask_generator(increasing=True):
    
    GSys = Generative_system(*m)
    GSys.compute_behavior_system(data)
    gensyslist.append(GSys)

# Find admissible systems.

admissible = find_admissible_behavior_systems(gensyslist)

pdb.set_trace()
