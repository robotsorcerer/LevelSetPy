__all__ = ["DubinsFlock"]

__author__ = "Lekan Molux"
__date__ = "Dec. 21, 2021"
__comment__ = "Two Dubins Vehicle in Relative Coordinates"

import cupy as cp
import numpy as np
from .dubins_absolute import DubinsVehicleAbs

class DubinsFlock(DubinsVehicleAbs):
    def __init__(self, grid, u_bound=5, w_bound=5, num_agents=7):
        """
            A flock of Dubins Vehicles. These are patterned after the 
            behavior of starlings which self-organize into local flocking patterns.

            Note that here, we must work in absolute coordinates.

            The inspiration for this is the following paper:

                "Interaction ruling animal collective behavior depends on topological 
                rather than metric distance: Evidence from a field study." 
                ~ Ballerini, Michele, Nicola Cabibbo, Raphael Candelier, 
                Andrea Cavagna, Evaristo Cisbani, Irene Giardina, Vivien Lecomte et al. 
                Proceedings of the national academy of sciences 105, no. 4 
                (2008): 1232-1237. 

            Dynamics:
                \dot{x}_1 = -v_e + v_p cos x_3 + w_e x_2
                \dot{x}_2 = -v_p sin x_3 - w_e x_1
                \dot{x}_3 = -w_p - w_e

            Parameters
            ==========
                grid: 2 possible types of grids exist for resolving vehicular dynamics:
                    .single_grid: an np.meshgrid that homes all these birds
                    .multiple grids: a collection of possibly intersecting grids 
                    where agents interact.
                
                u_bound: absolute value of the linear speed of the vehicle.

                w_bound: absolute value of the angular speed of the vehicle.

                num_agents: number of agents in this flock of vehicles.
        """

        self.v           = lambda u: u*u_bound
        self.w           = lambda w: w*w_bound
        # Number of vehicles in this flock
        self.N           = num_agents

        # birds could be on different subspaces of an overall grid
        if isinstance(grid, list):
            self.vehicles = []
            for each_grid in grid:
                self.vehicles.append(DubinsVehicleAbs(each_grid, u_bound, w_bound))
        else: # all birds are on the same grid
            self.vehicles = [DubinsVehicleAbs(grid, u_bound, w_bound) for _ in range(num_agents)]

        self.grid        = grid
        """
             Define the anisotropic parameter for this flock.
             This gamma parameter controls the degree of interaction among 
             the agents in this flock. Interaction decays with the distance, and 
             we can use the anisotropy to get information about the interaction.
             Note that if nc=1 below, then the agents 
             exhibit isotropic behavior and the aggregation is non-interacting by and large.
        """
        self.gamma = lambda nc: (1/3)*nc

        # for the nearest neighors in this flock, they should have an anisotropic policy
        # set linear speeds
        if not np.isscalar(u_bound) and len(u_bound) > 1:
            self.v_e = self.v(1)
            self.v_p = self.v(-1)
        else:
            self.v_e = self.v(1)
            self.v_p = self.v(1)

        # set angular speeds
        if not np.isscalar(w_bound) and len(w_bound) > 1:
            self.w_e = self.w(1)
            self.w_p = self.w(-1)
        else:
            self.w_e = self.w(1)
            self.w_p = self.w(1)

    def hamiltonian(self, t, data, value_derivs, finite_diff_bundle):
        """
            H = p_1 [v_e - v_p cos(x_3)] - p_2 [v_p sin x_3] \
                   - w | p_1 x_2 - p_2 x_1 - p_3| + w |p_3|

            Parameters
            ==========
            value: Value function at this time step, t
            value_derivs: Spatial derivatives (finite difference) of
                        value function's grid points computed with
                        upwinding.
            finite_diff_bundle: Bundle for finite difference function
                .innerData: Bundle with the following fields:
                    .partialFunc: RHS of the o.d.e of the system under consideration
                        (see function dynamics below for its impl).
                    .hamFunc: Hamiltonian (this function).
                    .dissFunc: artificial dissipation function.
                    .derivFunc: Upwinding scheme (upwindFirstENO2).
                    .innerFunc: terminal Lax Friedrichs integration scheme.
        """
        p1, p2, p3 = value_derivs[0], value_derivs[1], value_derivs[2]
        p1_coeff = self.v_e - self.v_p * cp.cos(self.grid.xs[2])
        p2_coeff = self.v_p * cp.sin(self.grid.xs[2])

        Hxp = p1 * p1_coeff - p2 * p2_coeff - self.w(1)*cp.abs(p1*self.grid.xs[1] - \
                p2*self.grid.xs[0] - p3) + self.w(1) * cp.abs(p3)

        return Hxp

    def dissipation(self, t, data, derivMin, derivMax, \
                      schemeData, dim):
        """
            Parameters
            ==========
                dim: The dissipation of the Hamiltonian on
                the grid (see 5.11-5.12 of O&F).

                t, data, derivMin, derivMax, schemeData: other parameters
                here are merely decorators to  conform to the boilerplate
                we use in the levelsetpy toolbox.
        """
        assert dim>=0 and dim <3, "Dubins vehicle dimension has to between 0 and 2 inclusive."

        if dim==0:
            return cp.abs(self.v_e - self.v_p * cp.cos(self.grid.xs[2])) + cp.abs(self.w(1) * self.grid.xs[1])
        elif dim==1:
            return cp.abs(self.v_p * cp.sin(self.grid.xs[2])) + cp.abs(self.w(1) * self.grid.xs[0])
        elif dim==2:
            return self.w_e + self.w_p

    def dynamics(self):
        """
            Computes the Dubins vehicular dynamics in relative
            coordinates (deterministic dynamics).

            \dot{x}_1 = -v_e + v_p cos x_3 + w_e x_2
            \dot{x}_2 = -v_p sin x_3 - w_e x_1
            \dot{x}_3 = -w_p - w_e
        """
        x1 = self.grid.xs[0]
        x2 = self.grid.xs[1]
        x3 = self.grid.xs[2]

        xdot = [
                -self.ve + self.vp * np.cos(x3) + self.we * x2,
                -self.vp * np.sin(x3) - self.we * x1,
                -self.wp - self.we # pursuer minimizes
        ]

        return xdot
