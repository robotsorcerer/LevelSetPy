__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

import cupy as cp
from BoundaryCondition import addGhostPeriodic
from Utilities import isfield, expand

def augmentPeriodicData(g, data):

    # Dealing with periodicity
    for i  in range(g.dim):
        if isfield(g, 'bdry') and (id(g.bdry[i])==id(addGhostPeriodic)):
            # Grid points
            # print(f'g.vs[{i}]: b4 {g.vs[i].shape}, {g.vs[i][-1].shape}, {g.dx[i]}')
            g.vs[i] = cp.concatenate((g.vs[i], expand(g.vs[i][-1] + g.dx[i], 1)), 0)
            # print(f'g.vs[{i}]: aft {g.vs[i].shape}')
            # Icp.t data eg. data = cat(:, data, data(:,:,1))
            # indices = cp.arange(g.dim, dtype=cp.intp)
            # indices[i] = 0
            # print('indices: ', indices, ' data: ', data.shape)
            # to_app = data[cp.ix_(indices)]
            to_app = expand(data[i,...], i)
            # print('to_app ', to_app.shape, ' data b4: ', data.shape)
            data = cp.concatenate((data, to_app), i)
            # print(i, ' data aft: ', data.shape)

    return g, data
