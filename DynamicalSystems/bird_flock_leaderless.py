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
        self.reset(self.vertex_set[list(self.vertex_set.keys())[0]].w)

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
    def __init__(self, grids, num_agents=7, grid_nodes=101,
                reach_rad=1.0, avoid_rad=1.0):
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
                .grids: 2 possible types of grids exist for resolving vehicular dynamics:
                    .single_grid: an np.meshgrid that homes all these birds
                    .multiple grids: a collection of possibly intersecting grids 
                    where agents interact.
                
                .num_agents: number of agents in this flock of vehicles.

                .grid_nodes: number of nodes in grid.
        """
        # Number of vehicles in this flock
        self.N  = num_agents
        self.grid_nodes = grid_nodes

        if grids is None and num_agents==1:
            # for every agent, create the grid bounds
            grid_mins = [[-1, -1, -np.pi]]
            grid_maxs = [[1, 1, np.pi]]   
            grids = flockGrid(grid_mins, grid_maxs, dx=.1, num_agents=num_agents, N=grid_nodes)
        elif grids is None and num_agents>1:
            gmin = np.array(([[-1, -1, -np.pi]]),dtype=np.float64).T
            gmax = np.array(([[1, 1, np.pi]]),dtype=np.float64).T
            grid = createGrid(gmin, gmax, grid_nodes, 2)

        # birds could be on different subspaces of an overall grid
        if isinstance(grids, list):
            self.vehicles = []
            #reference bird must be at origin of the grid
            # bird_pos = [ 
            #              np.mean(grids[0].vs[0]),
            #              np.mean(grids[0].vs[1]),
            #              np.mean(grids[0].vs[2])
            #              ]
            lab = 0
            for each_grid in grids:
                self.vehicles.append(BirdSingle(each_grid,1,1,None, \
                                         random.random(), label=lab))
                # # randomly initialize position of other birds
                # bird_pos = [np.random.sample(each_grid.vs[0], 1), \
                #             np.random.sample(each_grid.vs[1], 1), \
                #             np.random.sample(each_grid.vs[2], 1)]
                lab += 1
        else: # all birds are on the same grid
            ref_bird = BirdSingle(grids[0], 1, 1, None, \
                                random.random(), label=0) 
            self.vehicles = [BirdSingle(grids[i], 1, 1, None, \
                    random.random(), label=i) for i in range(1,num_agents)]

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
        """create the target set for this local flock"""
        self.flock_payoff = self.get_target(reach_rad=1.0, avoid_rad=1.0)
            
        self.graph = Graph(num_agents, self.grid, self.vehicles, None)

    def update_dynamics(self):
        """
            Update the dynamics of the agents on this grid whose positionings are
            determined by a graph self.graph.
        """
        
        # recursively update each agent's position and neighbors in the state
        for idx in range(len(self.vehicles)):
            self.graph.θs[idx,:] = self.update_agent_single(self.vehicles[idx])

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
        # neighbors are those agents within a normed distance to this agent's position
        n    = agent.n
        label = agent.label # we do not expect this to change

        # update headings and neighbors (see eqs 1 and 2 in Jadbabaie)
        for other_agent in self.vehicles:
            if other_agent == agent:
                # we only compare with other agents
                continue 

            # TODO: how to better find a vehicle's position on the state space at t? 
            dist = np.linalg.norm(other_agent.position()[:2], 2) - np.linalg.norm(agent.position()[:2], 2)
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
        #agent.update_agent_params(t, n, label, θr)
        return θr
           
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
        # note this is also the avoid set
        target_set = np.zeros((self.N-1,)+(evader.grid.shape), dtype=np.float64)
        payoff_avoid = np.zeros((evader.grid.shape), dtype=np.float64)
        cur_agent = 0
        for vehicle_idx in range(1, len(self.vehicles)-1):
            this_vehicle = self.vehicles[vehicle_idx]
            next_vehicle = self.vehicles[vehicle_idx+1]
            for i in range(this_vehicle.grid.dim):
                if(i != this_vehicle.axis_align):
                    target_set[cur_agent] += (this_vehicle.grid.xs[i] + next_vehicle.grid.xs[i])**2
            target_set[cur_agent] = np.sqrt(target_set[cur_agent])

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
