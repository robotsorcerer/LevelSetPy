__all__ = ["DubinsVehicleAbs"]

__author__ = "Lekan Molux"
__date__ = "Dec. 21, 2021"
__comment__ = "Two Dubins Vehicle in Absolute Coordinates"

import cupy as cp
import numpy as np

class DubinsVehicleAbs():
    def __init__(self, grid, u_bound=+5, w_bound=+5):
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
                self.v * np.cos(x3),
                self.vp * np.sin(x3),
                self.w
        ]

        return xdot
