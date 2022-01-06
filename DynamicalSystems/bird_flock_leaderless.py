__all__ = ["BirdFlock"]

__author__ = "Lekan Molux"
__date__ = "Dec. 21, 2021"
__comment__ = "Two Dubins Vehicle in Relative Coordinates"

import hashlib
import cupy as cp
import numpy as np
import random
from LevelSetPy.Grids import *
from .bird_single_leaderless import BirdSingle
from LevelSetPy.Utilities.matlab_utils import *

class Graph():
    def __init__(self, n, grids, vertex_set, edges=None):
        """A graph (an undirected graph that is) that models 
        the update equations of agents positions on a state space 
        (defined as a grid).

        The graph has a vertex set {1,2,...,n} so defined such that 
        (i,j) is one of the graph's edges in case i and j are neighbors.
        This graph changes over time since the relationship between neighbors
        can change.

        Paramters
        =========
            .grids
            n: number of initial birds (vertices) on this graph.
            .V: vertex_set, a set of vertices {1,2,...,n} that represent the labels
            of birds in a flock. Represent this as a list (see class vertex).
            .E: edges, a set of unordered pairs E = {(i,j): i,j \in V}.
                Edges have no self-loops i.e. i≠j or repeated edges (i.e. elements are distinct).
        """
        self.N = n
        if vertex_set is None:            
            self.vertex_set = {f"{i+1}":BirdSingle(grids[i], 1, 1,\
                    None, random.random(), label=f"{i}") for i in range(n)}
        else:
            self.vertex_set = {f"{i+1}":vertex_set[i] for i in range(n)}
        
        # edges are updated dynamically during game
        self.edges_set = edges 

        # obtain the graph params
        self.reset(self.vertex_set[list(self.vertex_set.keys())[0]].w_e)

    def reset(self, w):
        # graph entities: this from Jadbabaie's paper
        self.Ap = np.zeros((self.N, self.N)) #adjacency matrix
        self.Dp = np.zeros((self.N, self.N)) #diagonal matrix of valencies
        self.θs = np.ones((self.N, 1))*w # agent headings
        self.I  = np.ones((self.N, self.N))
        self.Fp = np.zeros_like(self.Ap) # transition matrix for all the headings in this flock

    def insert_vertex(self, vertex):
        if isinstance(vertex, list):
            assert isinstance(vertex, BirdSingle), "vertex to be inserted must be instance of class Vertex."
            for vertex_single in vertex:
                self.vertex_set[vertex_single.label] = vertex_single.neighbors
        else:
            self.vertex_set[vertex.label] = vertex

    def insert_edge(self, from_vertices, to_vertices):
        if isinstance(from_vertices, list) and isinstance(to_vertices, list):
            for from_vertex, to_vertex in zip(from_vertices, to_vertices):
                self.insert_edge(from_vertex, to_vertex)
            return
        else:
            assert isinstance(from_vertices, BirdSingle), "from_vertex to be inserted must be instance of class Vertex."
            assert isinstance(to_vertices, BirdSingle), "to_vertex to be inserted must be instance of class Vertex."
            from_vertices.update_neighbor(to_vertices)
            self.vertex_set[from_vertices.label] = from_vertices.neighbors
            self.vertex_set[to_vertices.label] = to_vertices.neighbors

    def adjacency_matrix(self, t):
        for i in range(self.Ap.shape[0]):
            for j in range(self.Ap.shape[1]):
                for verts in sorted(self.vertex_set.keys()):
                    if str(j) in self.vertex_set[verts].neighbors:
                        self.Ap[i,j] = 1 
        return self.Ap

    def diag_matrix(self):
        "build Dp matrix"
        i=0
        for vertex, egdes in self.vertex_set.items():
            self.Dp[i,i] = self.vertex_set[vertex].valence
        return self.Dp

    def update_headings(self, t):
        return self.adjacency_matrix(t)@self.θs

class BirdFlock(BirdSingle):
    def __init__(self, grids, vehicles, label=1,
                reach_rad=1.0, avoid_rad=1.0):
        """        
        Intro:
        =====    
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
            .vehicles: BirdSingle Objects in a list.
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
        self.graph = Graph(self.N, self.grid, self.vehicles, None)

        """
            update neighbors now based on topological distance
            update headings too: note that headings do not 
            contribute to neighbors;  only linear positions do.
        """
        self._housekeeping()

   
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
        for idx in range(len(self.vehicles)):
            self._update_headings(self.vehicles[idx], idx)

    def _compare_neighbor(self, agent1, agent2):
        "Check if agent1 is a neighbor of agent2."
        if np.abs(agent1.label - agent2.label) < agent1.neigh_rad:
            agent1.update_neighbor(agent2)
        
    def _update_headings(self, agent, idx, t=None):
        """
            Update the average heading of this flock. 

            Parameters:
            ==========
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

    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False

    def __repr__(self):
        parent=f"Flock: {self.label}"
        return parent 