__all__ = ["BirdFlock"]

__author__ = "Lekan Molux"
__date__ = "Dec. 21, 2021"
__comment__ = "Two Dubins Vehicle in Relative Coordinates"

import hashlib
import cupy as cp
import numpy as np
import random
from .bird_single import BirdSingle
from LevelSetPy.Utilities.matlab_utils import *

class BirdFlock(BirdSingle):
    def __init__(self, grids, u_bound=5, w_bound=5, num_agents=7, 
                center=None, radius=1):
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
                grids: 2 possible types of grids exist for resolving vehicular dynamics:
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
        if isinstance(grids, list):
            self.vehicles = []
            #reference bird must be at origin of the grid
            bird_pos = [ 
                         np.mean(grids[0].vs[0]),
                         np.mean(grids[0].vs[1]),
                         np.mean(grids[0].vs[2])
                         ]
            lab = 0
            for each_grid in grids:
                self.vehicles.append(BirdSingle(each_grid, u_bound, w_bound, \
                                        bird_pos, random.random(), label=lab))
                # randomly initialize position of other birds
                bird_pos = [np.random.sample(each_grid.vs[0], 1), \
                            np.random.sample(each_grid.vs[1], 1), \
                            np.random.sample(each_grid.vs[2], 1)]
                lab += 1
        else: # all birds are on the same grid
            self.vehicles = [BirdSingle(grids, u_bound, w_bound, \
                                [np.random.sample(grids.vs[0], 1), \
                                 np.random.sample(grids.vs[1], 1), \
                                 np.random.sample(grids.vs[2], 1)], \
                                random.random(), label=_+1) for _ in range(num_agents)]

        self.grid = grids
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

        """create the target set for this local flock"""
        self.flock_payoff = self.get_target(reach_rad=1.0, avoid_rad=1.0)

        # graph entities: this from Jadbabaie's paper
        self.Ap = np.zeros((self.N, self.N)) #adjacency matrix
        self.Dp = np.zeros((self.N, self.N)) #diagonal matrix of valencies
        self.Fp = np.zeros_like(self.Ap) # transition matrix for all the headings in this flock
        self.θs = np.zeros((self.N, 1)).fill(self.w(1)) # agent headings

        # recursively update each agent's position and neighbors in the state
        for agent in self.vehicles:
            self.update_agent_single(agent)
        

    def update_agent_single(self, agent, t=None):
        """
            Compute the # of neighbors of this `agent` at time t.
            In addition, update the number of neighbors of this agent
             as labels in a list, and compute the average heading of
             this robot as well. 

            Update the number of nearest neighbors of this agent
            and then the labels of its neighbors for bookkeeping.

            Parameters:
            ==========
            agent: This agent as a BirdsSingle object.
            t: Time at which we are updating this agent's dynamics.
        """
        # TODO: get this agent's position; how to better valc an agent's psosition
        pos = agent.update_position() 
        # neighbors are those agents within a normed distance to this agent's position
        n    = agent.n
        label = agent.label # we do not expect this to change

        # update headings and neighbors (see eqs 1 and 2 in Jadbabaie)
        for other_agent in self.vehicles:
            if other_agent == agent:
                # we only compare with other agents
                continue 

            # TODO: how to better find a vehicle's position on the state space at t? 
            dist = np.norm(other_agent.position()[:2], 2) - np.norm(agent.position()[:2], 2)
            if dist <= agent.neigh_rad:
                n += 1 # increase neighbor count if we are within the prespecified radius
                agent.neighbors.append(other_agent.label) # label will be integers
        
        # update heading for this agent
        if np.any(agent.neighbors):
            neighbor_headings = [self.vehicles[agent.neighbors[i]].w \
                for i in range(len(agent.neighbors)) \
                    if agent!=self.vehicles[agent.neighbors[i]]]
        else:
            neighbor_headings = 0
        # this maps headings w/values in [0, 2\pi) to [0, \pi)
        θr = (1/(1+n))*(agent.w + np.sum(neighbor_headings))        
        agent.update_agent_params(t, n, label, θr)
        
    

    def get_target(self, reach_rad=1.0, avoid_rad=1.0):
        """
            Make reference bird the evader and every other bird the pursuer
            owing to the lateral visual anisotropic characteric of starlings.
        """
        # first bird is the evader, so collect its position info
        cur_agent = 0
        evader = self.vehicles[cur_agent]
        target_set = np.zeros((self.N-1,)+(evader.grid.shape), dtype=np.float64)
        payoff_capture = np.zeros((evader.grid.shape), dtype=np.float64)
        # first compute the any pursuer captures an evader
        for pursuer in self.vehicles[1:]:
            if not np.any(pursuer.center):
                pursuer.center = np.zeros((pursuer.grid.dim, 1))
            elif(numel(pursuer.center) == 1):
                pursuer.center = pursuer.center * np.ones((pursuer.grid.dim, 1), dtype=np.float64)

            #---------------------------------------------------------------------------
            #axis_align must be same for all agents in a flock
            # any pursuer can capture the reference bird
            for i in range(pursuer.grid.dim):
                if(i != pursuer.axis_align):
                    target_set[cur_agent] += (pursuer.grid.xs[i] - evader.grid.xs[i])**2
            target_set[cur_agent] = np.sqrt(target_set[cur_agent])

            # take an element wise min of all corresponding targets now
            if cur_agent >= 1:
                payoff_capture = np.minimum(target_set[cur_agent], target_set[cur_agent-1], dtype=np.float64)
            cur_agent += 1
        payoff_capture -= reach_rad

        # compute the anisotropic value function: this maintains the gap between the pursuers
        target_set = np.zeros((self.N-1,)+(evader.grid.shape), dtype=np.float64)
        payoff_avoid = np.zeros((evader.grid.shape), dtype=np.float64)
        cur_agent = 0
        for vehicle_idx in range(1, len(self.vehicles)-1):
            this_vehicle = self.vehicles[vehicle_idx]
            next_vehicle = self.vehicles[vehicle_idx+1]
            for i in range(this_vehicle.grid.dim):
                if(i != this_vehicle.axis_align):
                    target_set[cur_agent] += (this_vehicle.grid.xs[i] + next_vehicle.grid.xs[i])**2
            target_set[cur_agent] = np.sqrt(target_set)

            # take an element wise min of all corresponding targets now
            if cur_agent >= 1:
                payoff_avoid = np.minimum(target_set[cur_agent], target_set[cur_agent-1], dtype=np.float64)
            cur_agent += 1
        
        payoff_avoid -= avoid_rad

        # now do a union of both the avoid and capture sets
        combo_payoff = np.minimum(payoff_avoid, payoff_capture)

        return combo_payoff



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