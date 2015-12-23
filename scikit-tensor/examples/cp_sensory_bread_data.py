#!/usr/bin/env python

import logging
from scipy.io.matlab import loadmat

import sys
sys.path.append('..')
from sktensor import dtensor, cp_als, tucker_hooi

# Set logging to DEBUG to see CP-ALS information
logging.basicConfig(level=logging.DEBUG)

# Load Matlab data and convert it to dense tensor format
mat = loadmat('../data/sensory-bread/brod.mat')
T = dtensor(mat['X'])

# Decompose tensor using CP-ALS
# P, fit, itr, exectimes = cp_als(T, 3, init='random')
core, U = tucker_hooi(T, 3, init='random')
print U