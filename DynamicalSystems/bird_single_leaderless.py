__all__ = ["BirdSingle"]

__author__ = "Lekan Molux"
__date__ = "Dec. 25, 2021"
__comment__ = "Single Dubins Vehicle under Leaderless Coordination."

import time
import random
import hashlib
import cupy as cp
import numpy as np
from LevelSetPy.Utilities import eps

class BirdSingle():
    def __init__(self, grid, u_bound=+1, w_bound=+1, \
                 init_state=None, rw_cov=None, \
                 axis_align=2, center=None, 
                 neigh_rad=.3, init_random=False,
                 label="0", neighbors=[]):
        """
            Dubins Vehicle Dynamics in absolute coordinates.
            Please consult Merz, 1972 for a detailed reference.

            Dynamics:
            ==========
                \dot{x}_1 = v cos x_3
                \dot{x}_2 = v sin x_3
                \dot{x}_3 = w

            Parameters:
            ===========
                grid: an np.meshgrid state space on which we are
                resolving this vehicular dynamics. This grid does not have
                a value function (yet!) until it's part of a flock
                u_bound: absolute value of the linear speed of the vehicle.
                w_bound: absolute value of the angular speed of the vehicle.
                init_state: initial position and orientation of a bird on the grid
                rw_cov: random covariance scalar for initiating the stochasticity
                on the grid.
                center: location of this bird's value function on the grid
                axis_align: periodic dimension on the grid to be created
                neigh_rad: sets of neighbors of agent i
                label: label of this bird; must be a unique integer
                n: number of neighbors of this agent at time t
                init_random: randomly initialize this agent's position on the grid?

            BirdSingle Parameters
            =================
                .label (str): The label of this BirdSingle drawn from the set {1,2,...,n} 
                .valence(int): number of edges incident on a BirdSingle V_i
                .indicant_edge(tuple): an edge (i,j) such that i=v or j=v.
                .neighbors: neighbors of this BirdSingle (as labels).

            Test
            ====
                b0 = BirdSingle(.., label="0")
                b1 = BirdSingle(.., "1")
                b2 = BirdSingle(.., "2")
                b3 = BirdSingle(.., "3")
                b0.update_neighbor(b1)
                b0.update_neighbor(b2)
                b2.update_neighbor(b3)
                print(b0)
                print(b1)
                print(b2)
                print(b3)

                Prints: BirdSingle: 0 | Neighbors: ['1', '2']
                        BirdSingle: 1 | Neighbors: ['0']
                        BirdSingle: 2 | Neighbors: ['0', '3']
                        BirdSingle: 3 | Neighbors: ['2']

                Multiple neighbors test
                -----------------------
                neighs_2 = [BirdSingle(f"{i}") for i in range(3, 9)]
                b2.update_neighbors(neighs_2)
                print(b2)

                Prints
        """

        assert label is not None, "label of an agent cannot be empty"
        # BirdSingle Params
        self.label = label
        # set of labels of those agents whicvh are neighbors of this agent
        self.neighbors   = neighbors
        self.inidicant_edge = []       
        # minimum L2 distance that defines a neighbor 
        self.neigh_rad = neigh_rad
        self.init_random = init_random
        self.n  = 0

        # grid params
        self.grid        = grid
        self.center      = center
        self.axis_align  = axis_align

        # for the nearest neighors in this flock, they should have an anisotropic policy
        self.v = lambda v: u_bound
        self.w = lambda w: w_bound

        # set actual linear speeds: 
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

        # this is a vector defined in the direction of its nearest neighbor
        self.u = None
        self.deltaT = eps # use system eps for a rough small start due to in deltaT
        self.rand_walk_cov = random.random if rw_cov is None else rw_cov

        # position this bird at the center of the state space
        self.position(init_state)
        
    def position(self, cur_state=None, t_span=None):
        """
            Birds use an optimization scheme to keep
            separated distances from one another.

            'even though vision is the main mechanism of interaction,
            optimization determines the anisotropy of neighbors, and
            not the eye's structure. There is also the possibility that
            each individual keeps the front neighbor at larger distances
            to avoid collisions. This collision avoidance mechanism is
            vision-based but not related to the eye's structure.'

            Parameters
            ==========
            cur_state: position and orientation.
                i.e. [x1, x2, θ] at this current position
            t_span: time_span as a list [t0, tf] where
                .t0: initial integration time
                .tf: final integration time
            
            Parameters
            ==========
            .init_state: current state of a bird in the state space
                (does not have to be an initial state/could be a current
                state during simulation). If it is None, it is initialized
                on the center of the state space.
        """
        if not np.any(cur_state):
            #put cur_state at origin if not specified
            cur_state = [np.mean(self.grid.vs[0]),
                         np.mean(self.grid.vs[1]),
                         np.mean(self.grid.vs[2])
                         ]
        
        X = self.do_runge_kutta4(cur_state, t_span)

        if self.init_random:
            # Simulate each agent's position in a flock as a random walk
            W = np.asarray(([self.deltaT**2/2])).T
            WW = W@W.T
            rand_walker = np.ones((len(cur_state))).astype(float)*WW*self.rand_walk_cov**2

            X += rand_walker

        return X

    def dynamics(self, cur_state):
        """
            Computes the Dubins vehicular dynamics in relative
            coordinates (deterministic dynamics).

            \dot{x}_1 = -v_e + v_p cos x_3 + w_e x_2
            \dot{x}_2 = -v_p sin x_3 - w_e x_1
            \dot{x}_3 = -w_p - w_e
        """
        xdot = [
                self.v_e * np.cos(cur_state[2]),
                self.v_e * np.sin(cur_state[2]),
                self.w_e * np.ones_like(cur_state[2])
        ]

        return np.asarray(xdot)

    def update_neighbor(self, neigh):
        if isinstance(neigh, list):
            for neigh_single in neigh:
                self.update_neighbor(neigh_single)
            return
        assert isinstance(neigh, BirdSingle), "Neighbor must be a BirdSingle member function."
        # assert neigh not in self.neighbors, "No repeated neighbors allowed."
        # assert neigh!=self, "Cannot assign a BirdSingle as its own neighbor"

        if neigh in self.neighbors or neigh==self:
            return self.neighbors 

        self.neighbors.append(neigh)

        # this neighbor must be a neighbor of this parent
        neigh.neighbors.append(self) 

    @property
    def valence(self):
        """
            By how much has the number of edges incident
            on v changed?

            Parameter
            =========
            delta: integer (could be positive or negative).

            It is positive if the number of egdes increases at a time t.
            It is negative if the number of egdes decreases at a time t.
        """
        return len(self.neighbors)
    
    def update_inidicant_edges(self, edges):
        """
            Update the number of edges (i,j) of the graph for which either
            i=j or j=v.
        """   
        pass     

    def do_runge_kutta4(self, cur_state, t_span, M=4, h=2):
        """
            .cur_state: state at time space 
            .t_span
            .M: RK steps per interval 
            .h: time step        
        """
        # integrate the dynamics with 4th order RK scheme
        X = np.array(cur_state) if isinstance(cur_state, list) else cur_state

        for j in range(M):
            if np.any(t_span): # integrate for this much time steps
                hh = (t_span[1]-t_span[0])/10/M
                for h in np.arange(t_span[0], t_span[1], hh):
                    k1 = self.dynamics(X)
                    k2 = self.dynamics(X + h/2 * k1)
                    k3 = self.dynamics(X + h/2 * k2)
                    k4 = self.dynamics(X + h * k3)

                    X  = X+(h/6)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                k1 = self.dynamics(X)
                k2 = self.dynamics(X + h/2 * k1)
                k3 = self.dynamics(X + h/2 * k2)
                k4 = self.dynamics(X + h * k3)

                X  = X+(h/6)*(k1 +2*k2 +2*k3 +k4)

        return X

    def __hash__(self):
        # hash method to distinguish agents from one another    
        return int(hashlib.md5(str(self.label).encode('utf-8')).hexdigest(),16)

    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False

    def __repr__(self):
        s="Bird {}."\
        .format(self.label)
        return s  
        parent=f"BirdSingle: {self.label} | "
        children="Neighbors: 0" if not self.neighbors \
                else f"Neighbors: {sorted([x.label for x in self.neighbors])}"
        return parent + children  