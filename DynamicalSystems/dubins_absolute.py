__all__ = ["DubinsVehicleAbs"]

__author__ = "Lekan Molux"
__date__ = "Dec. 21, 2021"
__comment__ = "Two Dubins Vehicle in Absolute Coordinates"

import cupy as cp
import numpy as np

class DubinsVehicleAbs():
    def __init__(self, grid, u_bound=+5, w_bound=+5, init_state=[0,0,0]):
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
                resolving this vehicular dynamics.
                u_bound: absolute value of the linear speed of the vehicle.
                w_bound: absolute value of the angular speed of the vehicle.
        """
        self.grid        = grid
        self.v = lambda u: u*u_bound
        self.w = lambda w: w*w_bound

        # this is a vector defined in the direction of its nearest neighbor
        self.u = None

        # position this bird at such and such position in the state space
        self.position(init_state)

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

    def position(self, cur_state, t_span=None):
        """
            Given the speed and orientation of this bird in the 
            flock, what is its position and orientation at a time t_o-->t_f?
            Here, we solve a basic RK4 integration scheme.

            Parameters
            ==========
            cur_state: position and orientation, 
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

        return list(X)