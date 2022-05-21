__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, A 2nd Order Reachable Sets Computation Scheme via a  Cauchy-Type Variational Hamilton-Jacobi-Isaacs Equation."
__comment__ 	= "Returns trajectories that belong to a given state space."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Ongoing"

import numpy as np
from .system import VaHJIApprox
from LevelSetPy.Grids import *
from LevelSetPy.Utilities import *

def generate_init_conds(dx = .1, grid=None):
    
    """        
        This part accumulates all initial trajectories within the
        given bounds of the state space.

        Inputs
        ------
         .dx: delta between every discretization between trajectories that emerge on 
            the state space. 
         .grid: Optional. Grid properties as already created by createGrid in the LevelSetPy Toolbox.
        
        Output
        -------
         .X: A Numpy array of all trajectories that emanate as initial conditions from 
          the state space.
    """
    # contants
    a = 64 # ft/sec^2
    g = 32 # ft/sec^2
    dx = .1  # state space resolution

    if grid is not None:
        grid_min  = grid.min
        grid_max  = grid.max
    else:
        grid_min = np.array(([0,0, 64, g]))
        grid_max = np.array(([1e5, 1e5, 64, g+20]))

    X = [grid_min]

    while np.any(X[-1]<grid_max):
        X += [X[-1]+dx]

    X = np.array(X)



    # # fix the inconsistencies in (x1, x2, x3, x4)
    indices = [np.nan for idx in range(X.shape[0])]
    for dim in range(X.shape[-1]):
        indices[dim] = X[:,dim]>grid_max[dim]

        # replace trajectories with bounds that exceed max 
        # values along respective dimensions
        X[indices[dim],dim] = grid_max[dim]   

        
    return X

def DDPReach(pol_r, eta=.1, rho=.9):
    """
        eta: Stopping criteria for backward pass
        rho: Cost improvement parameter
    """
    value_buff = np.zeros((4,4))
    X = generate_init_conds()
    
    T = 100
    dX = 4
    dU = 4
    dV = 4
           
    ur, vr = pol_r
    
    value_func_r = np.zeros((T, dX))
    
    for x_i in X:         
        xdot, fx, fu, fv = system(x_i, pol_r)   
        xr, pol_star = backward_pass(xdot, pol_r)  # check that vr is correct
        value_full_state, pol_ri = forward_pass(xr, pol_star)
        value_buff = np.max(value_buff, value_full_state)           
            