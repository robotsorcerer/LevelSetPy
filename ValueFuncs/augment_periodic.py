__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

import numpy as np
from LevelSetPy.BoundaryCondition import addGhostPeriodic
from LevelSetPy.Utilities import isfield, expand

def augmentPeriodicData(g, data):

    # Dealing with periodicity
    for i  in range(g.dim):
        if isfield(g, 'bdry') and (id(g.bdry[i])==id(addGhostPeriodic)):
            # Grid points
            g.vs[i] = np.concatenate((g.vs[i], expand(g.vs[i][-1] + g.dx[i], 1)), 0)
            
            to_app = expand(data[i,...], i)
            data = np.concatenate((data, to_app), i)

    return g, data
