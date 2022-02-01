__all__ = ["Murmurs"]

__author__ = "Lekan Molux"
__date__ = "Dec. 21, 2021"
__comment__ = "Two Dubins Vehicle in Relative Coordinates"

import random
import hashlib
import cupy as cp
import numpy as np
from .bird import Bird
from LevelSetPy.Grids import *
from LevelSetPy.Utilities.matlab_utils import *

class Murmurs(Bird):
    def __init__(self, grids, vehicles, label=1,
                reach_rad=1.0, avoid_rad=1.0):
        """
            Introduction:
            =============
                A flock of Dubins Vehicles. These are patterned after the
                behavior of starlings which self-organize into local flocking patterns.
                The inspiration for this is the following paper:
                    "Interaction ruling animal collective behavior depends on topological
                    rather than metric distance: Evidence from a field study."
                    ~ Ballerini, Michele, Nicola Cabibbo, Raphael Candelier,
                    Andrea Cavagna, Evaristo Cisbani, Irene Giardina, Vivien Lecomte et al.
                    Proceedings of the national academy of sciences 105, no. 4
                    (2008): 1232-1237.

            Parameters:
            ===========
                .grids: 2 possible types of grids exist for resolving vehicular dynamics:
                    .single_grid: an np.meshgrid that homes all these birds
                    .multiple grids: a collection of possibly intersecting grids
                        where agents interact.
                .vehicles: Bird Objects in a list.
                .id (int): The id of this flock.
                .reach_rad: The reach radius that defines capture by a pursuer.
                .avoid_rad: The avoid radius that defines the minimum distance between
                agents.
        """
        self.N         = len(vehicles)  # Number of vehicles in this flock
        self.label     = label      # label of this flock
        self.avoid_rad = avoid_rad  # distance between each bird.
        self.reach_rad = reach_rad  # distance between birds and attacker.
        self.vehicles  = vehicles   # # number of birds in the flock
        self.init_random = False

        self.grid = grids
        """
             Define the anisotropic parameter for this flock.
             This gamma parameter controls the degree of interaction among
             the agents in this flock. Interaction decays with the distance, and
             we can use the anisotropy to get information about the interaction.
             Note that if nc=1 below, then the agents
             exhibit isotropic behavior and the aggregation is non-interacting by and large.
        """
        #update neighbors+headings now based on topological distance
        self._housekeeping()

    def _housekeeping(self):
        """
            Update the neighbors and headings based on topological
            interaction.
        """
        # Update neighbors first
        for i in range(self.N):
            # look to the right and update neighbors
            for j in range(i+1,self.N):
                self._compare_neighbor(self.vehicles[i], self.vehicles[j])

            # look to the left and update neighbors
            for j in range(i-1, -1, -1):
                self._compare_neighbor(self.vehicles[i], self.vehicles[j])

        # recursively update each agent's headings based on neighbors
        for idx, agent in enumerate(self.vehicles):
            self._update_headings(agent, idx)

    def _compare_neighbor(self, agent1, agent2):
        "Check if agent1 is a neighbor of agent2."
        if np.abs(agent1.label - agent2.label) < agent1.neigh_rad:
            agent1.update_neighbor(agent2)

    def _update_headings(self, agent, idx, t=None):
        """
            Update the average heading of this flock.

            Parameters:
            ===========
            agent: This agent as a BirdsSingle object.
            t (optional): Time at which we are updating this agent's dynamics.
        """
        # update heading for this agent
        if agent.has_neighbor:
            neighbor_headings = [neighbor.w_e for neighbor in (agent.neighbors)]
        else:
            neighbor_headings = 0

        # this maps headings w/values in [0, 2\pi) to [0, \pi)
        θr = (1/(1+agent.valence))*(agent.w_e + np.sum(neighbor_headings))
        agent.w_e = θr

        # bookkeeing on the graph
        self.graph.θs[idx,:] =  θr

    def hamiltonian(self, t, data, value_derivs, finite_diff_bundle):
        """
            By definition, the Hamiltonian is the total energy stored in
            a system. If we have a team of agents moving along in a state
            space, it would inform us that the total Hamiltonian is a union
            (sum) of the respective Hamiltonian of each agent.

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
        # do housekeeping: update neighbors and headings
        self._housekeeping()

        # randomly drop one agent from the flock for the pursuer to attack
        attacked_idx = np.random.choice(len(self.vehicles))

        # update vehicles not under attack
        vehicles = [x for x in self.vehicles if x is not self.vehicles[attacked_idx]]
        
        # get hamiltonian of non-attcked agents
        unattacked_hams  = []
        for vehicle in vehicles:
            ham_x = vehicle.hamiltonian_abs(t, data, value_derivs, finite_diff_bundle)
            unattacked_hams.append(ham_x)
            # update orientations etc
            # self._housekeeping()
        unattacked_hams = cp.sum(cp.asarray(unattacked_hams), axis=0)

        # try computing the attack of a pursuer against the targeted agent
        attacked_ham = self.vehicles[attacked_idx].hamiltonian(t, data, value_derivs, finite_diff_bundle)

        # sum all the energies of the system
        ham = attacked_ham + unattacked_hams
        
        return ham

    def dissipation(self, t, data, derivMin, derivMax, \
                      schemeData, dim):
        """
            Just add the respective dissipation of all the agent's dissipation
        """
        assert dim>=0 and dim <3, "Dubins vehicle dimension has to between 0 and 2 inclusive."

        alphas = [vehicle.dissipation(t, data, derivMin, derivMax, schemeData, dim).take(0) for vehicle in self.vehicles]
        
        alphas = max(alphas)
        
        return cp.asarray(alphas)

    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False

    def __repr__(self):
        parent=f"Flock: {self.label}"
        return parent