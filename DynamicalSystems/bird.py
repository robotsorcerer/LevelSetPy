__all__ = ["Bird"]

__author__ = "Lekan Molux"
__date__ = "Dec. 25, 2021"
__comment__ = "Single Dubins Vehicle under Leaderless Coordination."

import time
import random
import hashlib
import cupy as cp
import numpy as np
from LevelSetPy.Utilities import eps

class Bird():
    def __init__(self, grid, u_bound=+1, w_bound=+1, \
                 init_xyw=None, rw_cov=None, \
                 axis_align=2, center=None,
                 neigh_rad=3, init_random=False,
                 label=0):
        """
            Parameters
            ----------
            .grid: an np.meshgrid state space on which we are
            resolving this vehicular dynamics. This grid does not have
            a value function (yet!) until it's part of a flock
            .u_bound: absolute value of the linear speed of the vehicle.
            .w_bound: absolute value of the angular speed of the vehicle.
            .init_xyz: initial position and orientation of a bird on the grid
            .rw_cov: random covariance scalar for initiating the stochasticity
            on the grid.
            .center: location of this bird's value function on the grid
            axis_align: periodic dimension on the grid to be created
            .neigh_rad: sets of neighbors of agent i
            .label (int): The label of this BirdSingle drawn from the set {1,2,...,n}
            .init_random: randomly initialize this agent's position on the grid?

            Tests
            -----
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
            Test1:
                # for every agent, create the grid bounds
                grid_mins = [[-1, -1, -np.pi]]
                grid_maxs = [[1, 1, np.pi]]
                grids = flockGrid(grid_mins, grid_maxs, dx=.1, N=101)
                ref_bird = BirdSingle(grids[0], 1, 1, None, \
                                    random.random(), label=0)
                print(ref_bird.position())

                print(ref_bird)

                num_agents=10
                neighs = [BirdSingle(grids[i], 1, 1, None, \
                                    random.random(), label=i) for i in range(1, num_agents)]
                ref_bird.update_neighbor(neighs)
                print(ref_bird, ' || valence: ', ref_bird.valence)

            Prints:
                array([0.99558554, 1.15271013, 8.        ])

                Agent: 0 | Neighbors: 0 || valence: 0.
                Agent: 0 | Neighbors: [1, 2, 3, 4, 5, 6, 7, 8, 9] || valence: 9.

            Test2:
                ref_bird = BirdSingle(grids[0], 1, 1, None, \
                        random.random(), label=21)
                print(ref_bird)
                neighs = [BirdSingle(grids[i], 1, 1, None, \
                                    random.random(), label=random.randint(10, 100)) for i in range(1, num_agents)]
                ref_bird.update_neighbor(neighs)
                print(ref_bird)

            Prints:
                Agent: 21 | Neighbors: 0 || valence: 0.
                Agent: 21 | Neighbors: [10, 39, 45, 61, 66, 67, 85, 90] || valence: 8.

            Author: Lekan Molux.
            December 2021
        """

        assert label is not None, "label of an agent cannot be empty"
        # BirdSingle Params
        self.label = label

        # set of labels of those agents whicvh are neighbors of this agent
        self.neighbors   = []
        self.indicant_edge = []

        # minimum L2 distance that defines a neighbor
        self.neigh_rad = neigh_rad
        # print(f'init_random: {init_random}')
        self.init_random = init_random
        # print(f'self.init_random: {self.init_random}')

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

        assert isinstance(init_xyw, np.ndarray), "initial state must either be a numpy or cupy array."
        r, c = init_xyw.shape
        if r<c:
            # turn to column vector
            init_xyw = init_xyw.T

        self.cur_state   = self.position(init_xyw )

        # adhoc functiomn for payoff
        self.payoff = None

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
            .init_xyz: current state of a bird in the state space
                (does not have to be an initial state/could be a current
                state during simulation). If it is None, it is initialized
                on the center of the state space.
        """

        if self.init_random:
            # Simulate each agent's position in a flock as a random walk
            W = np.asarray(([self.deltaT**2/2])).T
            WW = W@W.T
            rand_walker = np.ones((len(cur_state))).astype(float)*WW*self.rand_walk_cov**2

            cur_state += rand_walker

        return cur_state

    def reset(self):
        self.neighbors=[]

    def has_neighbor(self):
        """
            Check that this agent has a neighbor on the
            state space.
        """
        if self.neighbors is not None:
            return True
        return False

    def update_neighbor(self, neigh):
        """
            Neigh: A BirdSingle Instance.
        """
        if isinstance(neigh, list): # multiple neighbors.
            for neigh_single in neigh:
                self.update_neighbor(neigh_single)
            return
        assert isinstance(neigh, Bird), "Neighbor must be a BirdSingle member function."

        if neigh in self.neighbors or neigh==self:
            return self.neighbors
        self.neighbors.append(neigh)

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
        cur_state = cp.asarray(self.cur_state)

        p1_coeff = self.v_e - self.v_p * cp.cos(cur_state[2])
        p2_coeff = self.v_p* cp.sin(cur_state[2])

        # find lower and upper bound of orientation of vehicles that are neighbors
        w_e_upper_bound = max([state.cur_state[2] for state in self.neighbors]).item(0)
        w_e_lower_bound = min([state.cur_state[2] for state in self.neighbors]).item(0)

        Hxp = (p1 * p1_coeff - p2 * p2_coeff + cur_state[2]) + \
               w_e_upper_bound*cp.abs(p2 * self.cur_state[0] - p1*self.cur_state[1]+p3) -\
                self.w(-1) * cp.abs(p3)         
        
        # Note the sign of w

        return Hxp

    def dissipation(self, t, data, derivMin, derivMax, schemeData, dim):
        """
            Parameters
            ==========
                dim: The dissipation of the Hamiltonian on
                the grid (see 5.11-5.12 of O&F). Robust cohesion is simulated 
                against a worst-possible attacker.

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

    def __hash__(self):
        # hash method to distinguish agents from one another
        return int(hashlib.md5(str(self.label).encode('utf-8')).hexdigest(),16)

    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False

    def __repr__(self):
        parent=f"Agent: {self.label} | "
        children="Neighbors: 0" if not self.neighbors \
                else f"Neighbors: {sorted([x.label for x in self.neighbors])}"
        valence=f" || valence: {self.valence}."
        return parent + children  + valence
