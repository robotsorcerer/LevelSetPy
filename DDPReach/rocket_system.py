__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, A 2nd Order Reachable Sets Computation Scheme via a  Cauchy-Type Variational Hamilton-Jacobi-Isaacs Equation."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Ongoing"

import numpy as np
from scipy.linalg import cholesky
from LevelSetPy.Utilities import *


class RocketSystem():
    def __init__(self, r = 100, \
                  T=100, DX=4, DU=4, DV=4):
        """
            This class sets up the dynamics, value function and Hamiltonian as
            well as their partial derivatives for the differential game between
            two rockets over a shared space in $\mathbb{R}^n$.

            Parameters:
            -----------
              Capture params:
                r: Capture radius.
              Integration/State/Control Parameters:
                .T: Horizon of time in which to integrate the for-back equations.
                .DX: Dimension of the state.
                .DU: Dimension of the control.
                .DV: Dimension of the disturbance.
        """
        # gravitational acceleration, 32.17ft/sec^2
        self.g = 32.17
        # acceleration of the rocket <-- eq 2.3.70 Dreyfus
        self.a = 64.0 #ft/sec^2
        # capture radius
        self.capture_rad = r # ft

        self.pre_allocations(T, DX, DU, DV)


    def pre_allocations(self, T, DX, DU, DV):
        """
            Pre-allocate all buffers for every V, H, f and their derivatives.
            Arguments:  Dimensions along time, x u, v respectively.
        """
        # Dimensions along time, x u, v
        self.T, self.DX, self.DU, self.DV = T, DX, DU, DV

        # Allocations
        self.fx = np.zeros((self.T, self.DX,self.DX))
        self.fu = np.zeros((self.T, self.DV))
        self.fv = np.zeros((self.T, self.DV))

        # Hamiltonian allocations
        self.H   = np.zeros((self.T))
        self.Hx  = np.zeros((self.T, self.DX))
        self.Hu  = np.zeros((self.T, self.DU))
        self.Hv  = np.zeros((self.T, self.DV))
        self.Hvu = np.zeros((self.T, self.DV, self.DU))
        self.Huv = np.zeros((self.T, self.DU, self.DV))
        self.Hvv = np.zeros((self.T, self.DV, self.DV))
        self.Huu = np.zeros((self.T, self.DU, self.DU))
        self.Hux = np.zeros((self.T, self.DU, self.DX))
        self.Hvx = np.zeros((self.T, self.DV, self.DX))
        self.Hxx = np.zeros((self.T, self.DX, self.DX))

        # Value functions and their derivatives
        self.V  = np.zeros((self.T))
        self.Vx = np.zeros((self.T, self.DX))
        self.Vxx = np.zeros((self.T, self.DX, self.DX))

        # Gains
        self.ku = np.zeros((self.T, self.Du, self.Du))
        self.kv = np.zeros((self.T, self.Dv, self.Dv))

    def dynamics(self, x, pol):
        """
            Form:   dx/dt = f(x, u, v)

            The dynamics of the rocket is obtained from Dreyfus' formulation using a first-order Calculus
            of Variation model in 1962 in a RAND Report. This same model was used by S.K. Mitter in
            solving a 2nd-order gradient-based variational problem in 1966 in an Automatica paper.

            Here, we use the Mitter problem as stipulated by Jacobson and Mayne in their
            DDP book in 1970. See equations 2.3.68 through 2.3.75.

            The equations of motion are adopted from Dreyfus' construction as follows:

                &\dot{y}_1 = y_3,          &\dot{y}_5 = y_7, \\
                &\dot{y}_2 = y_4,          &\dot{y}_6 = y_8, \\
                &\dot{y}_3 = a\cos(u),     &\dot{y}_7 = a\cos(v), \\
                &\dot{y}_4 = a\sin(u) - g, &\dot{y}_8 = a\sin(v) - g.

            where $u(t), t \in [-T,0]$ is the controller under the coercion of the evader and
             $v(t), t \in [-T,0]$ is the controller under the coercion of the pursuer i.e.
             the pursuer is minimizing while the evader is maximizing. The full state dynamics
             is given by

            \dot{y} = \left(\begin{array}{c}
                            \dot{y}_1 & \dot{y}_2 & \dot{y}_3 & \dot{y}_4 \\
                            \dot{y}_5 & \dot{y}_6 & \dot{y}_7 &\dot{y}_8 \\
                            \end{array}
                        \right)^T =
                            \left(\begin{array}{c}
                            y_3 & y_4 & a\cos(u) & a\sin(u) - g &
                            y_7 & y_8 & a\cos(v) & a\sin(v) - g
                            \end{array}\right).

            In relative coordinates between the two rockets, we have
                    &\dot{x}_1 = x_3,\\
                    &\dot{x}_2 = x_4,\\
                    &\dot{x}_3 = a\cos(u) - a\cos(v),\\
                    &\dot{x}_4 = a\sin(u) - a\sin(v)

        Parameters
        ==========
            Input:
                .x - state as an initial condition
                .pol: A tuple of maximizing (u) and minimizing (v) controllers where:
                    .u - control input (evader)
                    .v - disturbance input (pursuer)
            Output:
                xdot - System dynamics
        """
        assert isinstance(x, list) or isinstance(x, np.ndarray), "x must be a List or ND-Array."
        assert isinstance(pol, tuple), "Policy passed to <dynamics> must be a Tuple."
        u, v = pol

        xdot = np.array(([x[2], x[3], \
                         self.a*np.cos(u)-self.a*np.cos(v), \
                         self.a*np.sin(u)-self.a*np.sin(v)])).T

        return xdot

    def rk4_integration(self, x0, pol, steps = 50, time_step = 0.2):
        """
            Compute an approximation to the rhs of an ode using a 4th-order Runge-Kutta
            integration technique/scheme. Equations from: https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html
            Note that this assumes that the system dynamics is of a non-autonomous system.

            Input:
                .x0 -- \equiv x(t_0). Initial condition of the ode to be integrated. Must be a list
                    or Numpy array.
                .steps -- How many steps for RK.
                .time_step -- The number of time per step.
            Output:
                .x  -- The resulting value(s) of the integrated system.
        """
        X = np.asarray(x0) if isinstance(x0, list) else x0

        for j in range(steps):
            k1 = self.dynamics(X, pol)                    # {k_1} = f({\hat{x}}({t_0}),{t_0})
            k2 = self.dynamics(X + time_step/2 * k1, pol) # f\left( {\hat{x}({t_0}) + {k_1}{h \over 2},{t_0} + {h \over 2}} \right)
            k3 = self.dynamics(X + time_step/2 * k2, pol) # f\left( {\hat{x}({t_0}) + {k_2}{h \over 2},{t_0} + {h \over 2}} \right)
            k4 = self.dynamics(X + time_step * k3, pol)   # f\left( {\hat{x}({t_0}) + {k_3}h,{t_0} + h} \right)

            # compute a weighted sum of the slopes to obtain the final estimate of \hat{x}(t_0 + h)
            X  = X+(time_step/6)*(k1 +2*k2 +2*k3 +k4)

        return X

    def f_derivs(self, f, pol, t=0):
        """
            This function computes the partials of f w.r.t the state,
            control, and disturbance. The definition of f is in the __doc__ of
            member function <dynamics> above.

            Inputs:
                .f -- System Dynamics
                .pol: A tuple of maximizing (u) and minimizing (v) controllers where:
                    .u - Control input (evader).
                    .v - Disturbance input (pursuer).
                .t - Time step at which to compute f.
            Output: Bundle of all partials of f i.e.:
                    .Bundle(fu, fv, fx)
        """
        assert isinstance(pol, tuple), "pol must be a tuple."
        u, v = pol

        self.fx[:,0,2] = 1
        self.fx[t,1,3] = 1

        self.fu[t,2,0] = -self.a*np.sin(u)
        self.fu[t,3,0] = self.a*np.cos(u)

        self.fv[t,2,0] = self.a*np.sin(v)
        self.fv[t,3,0] = -self.a*np.cos(v)

        # return Bundle(dict(fx=self.fx, fu=self.fu, fv=self.fv))

    def hamiltonian(self, states, pol, p, t=0):
        '''
            Input:
                .states: The State vector.
                .pol: A tuple of maximizing (u) and minimizing (v) controllers where:
                    .u - control input (evader).
                    .v - disturbance input (pursuer).
                    .t - Time step at which to compute H and its derivatives.
                .p: Costate  vector.
            Output: The Hamiltonian and its derivatives.

            **********************************************************************************
            The Hamiltonian:
                H(x,p) &=  p_1 * x_3 + p_2 * x_4 +
                            p_3*a*(\cos(u) - \cos(v)) + p_4*a*(\sin(u) - \sin(v))

            Derivative w.r.t u terms   &H_u = -p_3*a*\sin(u) + p_4*a*cos(u) \\
                                        &H_{uu} = -p_3*a*\cos(u) - p_4*a*\sin(u) \\
                                        &H_{uv} = \textbf{0}_{4\times4}  \\
                                        &H_{ux} = \textbf{0}_{4\times4}

            Derivative w.r.t v terms   &H_v = p_3*a*\sin(v) - p_4*a*\cos(v) \\
                                        &H_{vv} = p_3*a*\cos(v) + p_4*a*\sin(v) \\
                                        &H_{vu} = \textbf{0}_{4\times4} \\
                                        &H_{vx} = \textbf{0}_{4\times4}

            Derivative w.r.t x terms:  &H_x = \begin{bmatrix}0 &0 &p_1 &p_2 \end{bmatrix}^T \\
                                        &H_{xx} = 0_{4\times4} \\
                                        &H_{xu} = \textbf{0}_{4\times4} \\
                                        &H_{xv} = \textbf{0}_{4\times4}
        '''

        assert isinstance(states, np.ndarray), "The states of the dynamical system must be a list instance."
        assert isinstance(pol, tuple), "Policy input for Hamiltonian Function must be a tuple."
        assert isinstance(p, np.ndarray), "The co-state must be a list instance."
        u, v = pol

        # computations  #TODOs: Broadcast to appropriate Dims
        self.H[t] = p[0]*states[2] + p[1]*states[3] + \
             self.a * p[2]*(np.cos(u)-np.cos(v)) + \
             self.a * p[3]*(np.sin(u)-np.sin(v))

        self.Hu[t] = -p[2] * self.a * np.sin(u) + p[3] * self.a * np.cos(u)
        self.Huu[t] = -p[2] * self.a * np.cos(u) - p[3] * self.a * np.sin(u)

        self.Hv[t] = p[2] * self.a * np.sin(v) - p[3] * self.a * np.cos(v)
        self.Hvv[t] = p[2] * self.a * np.cos(v) + p[3] * self.a * np.sin(v)


        self.Hx[t, 2] = p[0]
        self.Hx[t, 3] = p[1]

        # return Bundle(dict(H=self.H, Hx=self.Hx, Hu=self.Hu, Hv=self.Hv, \
        #                     Huu=self.Huu, Hvv=self.Hvv, Huv=self.Huv, Hvx=self.Hvx, \
        #                     Hvu=self.Hvu, Hux=self.Hux, Hxx=self.Hxx))

    def value_funcs(self, x, t):
        """
            Compute the Value function as well as its partial
            derivatives. Note that the value function is the
            target set parameterized by time and state. The co-state and its 2nd order derivatives
            are matrices.

            Equations
            ---------
            V(x,0) &= \sqrt{x_1^2 + x_2^2} - r

            V_x = \dfrac{2 x_1}{\sqrt{x_1^2 + x_2^2}}{x}_3 +
                    \dfrac{2 x_2}{\sqrt{x_1^2 + x_2^2}}{x}_4;

            V_{xx} &= \dfrac{2 \left[x_1^3+x_2^2(x_2+x_3)+
                        x_1 x_2 (x_2 -x_3 - x_4) +
                        x_1^2(x_2+x_4)\right]}{\left(\sqrt{x1^2 + x2^2}\right)^3}

            Input:
              .x: State vector.
              .t - Time step at which to compute H and its derivatives.
            Output: Bundle of
              .V -- The terminal cost.
              .Vx -- The terminal cost's derivative wrt x.
              .Vxx -- The co-state's Hessian wrt x.
        """
        self.V[t] = np.sqrt(x[0]**2+x[1]**2)-self.capture_rad

        # Todos: Fix the broadcast of V's w.r.t to state
        self.Vx[t,0] = 2*x[0]
        self.Vx[t,1] = 2*x[1]

        #second order derivative
        self.Vxx[t,0,0] = 2
        self.Vxx[t,1,1] = 2

    def gains(self):
        """
            Compute the gains for the controller and disturbance
            𝑢∗, 𝑣∗ as given by equation (31).
        """
            #Huu_inv = np.zeros_like(self.Huu)
            try:
                Huu_upper = cholesky(self.Huu, lower=False)
                Huu_lower = Huu_upper.T
            except:
                raise LinAlgError("Had trouble computing the Huu inverse matrix.", LinAlgError)

            try:
                Hvv_upper = cholesky(self.Hvv, lower=False)
                Hvv_lower = Hvv_upper.T
            except:
                raise LinAlgError("Had trouble computing the Hvv inverse matrix.", LinAlgError)
            """
                Recall A = U^* U ==> A^{-1}=U^{-1} U*{-1}
                Solve for X in AX = I ==> U^* U X = I ==> U^* B = I
            """
            Huu_B = solve_triangular(Huu_upper, np.eye(self.DU)) # B above
            Hvv_B = solve_triangular(Hvv_upper, np.eye(self.DV)) # B above
