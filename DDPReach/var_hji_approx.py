__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, A 2nd Order Reachable Sets Computation Scheme via a  Cauchy-Type Variational Hamilton-Jacobi-Isaacs Equation."
__license__ 	= "Molux Licence"
__comment__ 	= "The algorithm in the LCSS paper."
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__date__ 		= "May 20, 2022"
__status__ 		= "Ongoing"

import numpy as np
from LevelSetPy.Utilities import *
from .rocket_system import RocketSystem


class VarHJIApprox(RocketSystem):
    def __init__(self, eta=.5, rho=.99, dx=.1, grid=None, X=None):
        super().__init__()
        """
            This class uses iterative dynamic game to compute the level sets of the
            value function for all scheduled trajectories of a dynamical system. For
            details see the following bibliography:

            "A Second-Order  Reachable Sets Computation Scheme via a  Cauchy-Type
            Variational Hamilton-Jacobi-Isaacs Equation." IEEE Letter to Control
            Systems Society, by Lekan Molu, Ian Abraham, and Sylvia Herbert.

            Parameters:
            -----------
            Cost improvement params:
                .eta: Stopping condition for backward pass;
                .rho: Regularization condition.
                .dx: delta between every discretization between trajectories that emerge on 
                    the state space. 
                .grid: Optional. Grid properties as already created by createGrid in the LevelSetPy Toolbox.
            
            .X: All initial state trajectories that emanate from the state space.
        """
        self.eta = eta
        self.rho = rho

        self.X = X if X else  self.generate_init_conds(dx, grid)


        self.system = RocketSystem()  

        # Buffer for all values
        self.value_buf = list()

    def generate_init_conds(self, dx = .1, grid=None):        
        """        
            This method accumulates all initial trajectories within the
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
