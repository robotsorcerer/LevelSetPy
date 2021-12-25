__all__ = ["BirdSingle"]

__author__ = "Lekan Molux"
__date__ = "Dec. 21, 2021"
__comment__ = "Two Dubins Vehicle in Absolute Coordinates"

import time
import hashlib
import cupy as cp
import numpy as np
from LevelSetPy.Utilities import eps

class BirdSingle():
    def __init__(self, grid, u_bound=+5, w_bound=+5, \
                 init_state=[0,0,0], rw_cov=0.0, \
                 axis_align=2, center=None, label=None,
                 neigh_rad=.4, n=0, init_random=False):
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
        """

        assert label is not None, "label of an agent cannot be empty"

        self.grid        = grid
        self.v = lambda u: u*u_bound
        self.w = lambda w: w*w_bound

        # this is a vector defined in the direction of its nearest neighbor
        self.u = None
        self.deltaT = eps # use system eps for a rough small start due to in deltaT
        self.rand_walk_cov = rw_cov

        self.center = center
        self.axis_align = axis_align

        # position this bird at in the state space
        self.position(init_state)
        

        # this from Jadbabie's paper
        self.label = label  # label of this bird in the flock (an integer)
        self.neigh_rad = neigh_rad
        self.n  = n
        self.init_random = init_random
        # set of labels of those agents whicvh are neighbors of this agent
        self.neighbors   = [] 

    def update_agent_params(self, t, n, label, w):
        """
            Update the parameters of this agent.
            t: continuous time, t.
            n: number of agents within a circle of raduius r 
                about the current position of this agent.
            label: label (as a natural number) of this agent.

            w: heading of this agent averaged over that of 
                its neighbors.

        """
        self.n     = n 
        self.w     = w     # averaged over the neighors that surround this agent
        self.label = label # update the label of this agent if it has not changed


    def position(self, init_state=None):
        """
            simulate each agent's position in a flock as a random walk
            Parameters
            ==========
            .init_state: current state of a bird in the state space
                (does not have to be an initial state/could be a current
                state during simulation). If it is None, it is initialized
                on the center of the state space.
        """
        W = np.asarray(([self.deltaT**2/2])).T
        WW = W@W.T

        rand_walker = np.ones((len(init_state))).astype(float)*WW*self.rand_walk_cov**2

        pos = self.update_position(init_state) 

        if self.init_random:
            pos += rand_walker
        
        return pos

    def dynamics(self, cur_state):
        """
            Computes the Dubins vehicular dynamics in relative
            coordinates (deterministic dynamics).

            \dot{x}_1 = -v_e + v_p cos x_3 + w_e x_2
            \dot{x}_2 = -v_p sin x_3 - w_e x_1
            \dot{x}_3 = -w_p - w_e
        """
        xdot = [
                self.v * np.cos(cur_state[2]),
                self.v * np.sin(cur_state[2]),
                self.w
        ]

        return np.asarray(xdot)

    def update_position(self, cur_state, t_span=None):
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
        """
        if not np.any(cur_state):
            #put cur_state at origin if not specified
            cur_state = [np.mean(self.grid.vs[0]),
                         np.mean(self.grid.vs[1]),
                         np.mean(self.grid.vs[2])
                         ]

        M, h = 4,  0.2 # RK steps per interval vs time step
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
