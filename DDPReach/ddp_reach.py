__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, A 2nd Order Reachable Sets Computation Scheme via a  Cauchy-Type Variational Hamilton-Jacobi-Isaacs Equation."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Ongoing"

import numpy as np
from LevelSetPy.Utilities import *


class VaHJIApprox():
    def __init__(self, X, eta=.5, rho=.99, r = 100, \
                  T=100, DX=4, DU=4, DV=4):
        """
            This functions uses iterative dynamic game to compute the level sets of the 
            value function for all scheduled trajectories of a dynamical system. 
            
            At issue is a differential game between two rockets over a shared space in 
            $\mathbb{R}^n$.

            Parameters:
            -----------
                .X: All states that emanate from the trajectory.
              Cost improvement params:
                .eta: Stopping condition for backward pass;
                .rho: Regularization condition.
              Capture params:
                r: Capture radius.
              Integration/State/Control Parameters:
                .T: Horizon of time in which to integrate the for-back equations.
                .DX: Dimension of the state.
                .DU: Dimension of the control.
                .DV: Dimension of the disturbance.
        """
        self.eta = eta
        self.rho = rho
        # gravitational acceleration, 32.17ft/sec^2
        self.g = 32.17 
        # acceleration of the rocket <-- eq 2.3.70 Dreyfus
        self.a = 64.0 #ft/sec^2  
        # capture radius
        self.capture_rad = r # ft

        # Dimensions along time, x u, v
        self.T, self.DX, self.DU, self.DV = T, DX, DU, DV

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
                .x - state
                .pol: A tuple of maximizing (u) and minimizing (v) controllers where:
                    .u - control input (evader)
                    .v - disturbance input (pursuer)
            Output:
                xdot - System dynamics
        """
        assert isinstance(x, list), "x must be a List."
        assert isinstance(pol, tuple), "Policy passed to <dynamics> must be a Tuple."
        u, v = pol
        
        xdot = np.array(([x[2], x[3], \
                         self.a*np.cos(u)-self.a*np.cos(v), \
                         self.a*np.sin(u)-self.a*np.sin(v)])).T
        
        return xdot
 
    def rk4_integration(self, x0, steps = 50, time_step = 0.2):
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
            k1 = self.dynamics(X)                    # {k_1} = f({\hat{x}}({t_0}),{t_0})
            k2 = self.dynamics(X + time_step/2 * k1) # f\left( {\hat{x}({t_0}) + {k_1}{h \over 2},{t_0} + {h \over 2}} \right)
            k3 = self.dynamics(X + time_step/2 * k2) # f\left( {\hat{x}({t_0}) + {k_2}{h \over 2},{t_0} + {h \over 2}} \right)
            k4 = self.dynamics(X + time_step * k3)   # f\left( {\hat{x}({t_0}) + {k_3}h,{t_0} + h} \right)

            # compute a weighted sum of the slopes to obtain the final estimate of \hat{x}(t_0 + h)
            X  = X+(time_step/6)*(k1 +2*k2 +2*k3 +k4)

        return X

    def f_derivs(self, f, pol):
        """
            This function computes the partials of f w.r.t the state,
            control, and disturbance.

            Inputs: 
                .f -- System Dynamics
                .pol: A tuple of maximizing (u) and minimizing (v) controllers where:
                    .u - control input (evader)
                    .v - disturbance input (pursuer)
            Output: Bundle of all partials of f i.e.:
                    .Bundle(fu, fv, fx)
        """
        assert isinstance(pol, tuple), "pol must be a tuple."
        u, v = pol

        # Allocations
        fx = np.zeros((4,4))
        fu = np.zeros((4,1))
        fv = np.zeros((4,1))

        fx[0,2] = 1 
        fx[1,3] = 1 
        
        fu[2] = -self.a*np.sin(u)
        fu[3] = self.a*np.cos(u)
        
        fv[2] = self.a*np.sin(v)
        fv[3] = -self.a*np.cos(v)
        
        return Bundle(dict(fx=fx, fu=fu, fv=fv))

    def hamiltonian(self, states, pol, p):
        '''
            Input:         
                .states: State vector.
                .pol: A tuple of maximizing (u) and minimizing (v) controllers where:
                    .u - control input (evader).
                    .v - disturbance input (pursuer).
                .p: Costate  vector.
            Output: The Hamiltonian and its derivatives.

            **********************************************************************************
            The Hamiltonian:
                H(x,p) &=  p_1x_3 + p_2x_4 + p_3a(\cos(u) - \cos(v)) + p_4a(\sin(u) - \sin(v))

            Derivative w.r.t u terms   &H_u = -p_3a\sin(u) + p_4a\cos(u) \\
                                        &H_{uu} = -p_3a\cos(u) - p_4a\sin(u) \\
                                        &H_{uv} = \textbf{0}_{4\times4}  \\
                                        &H_{ux} = \textbf{0}_{4\times4} 

            Derivative w.r.t v terms   &H_v = p_3a\sin(v) - p_4a\cos(v) \\
                                        &H_{vv} = p_3a\cos(v) + p_4a\sin(v) \\
                                        &H_{vu} = \textbf{0}_{4\times4} \\
                                        &H_{vx} = \textbf{0}_{4\times4} 

            Derivative w.r.t x terms:  &H_x = \begin{bmatrix}0 &0 &p_1 &p_2 \end{bmatrix}^T \\
                                        &H_{xx} = 0_{4\times4} \\
                                        &H_{xu} = \textbf{0}_{4\times4} \\
                                        &H_{xv} = \textbf{0}_{4\times4}                                         
        '''
        
        assert isinstance(pol, tuple), "Policy input for Hamiltonian Function must be a tuple."
        assert isinstance(p, list), "The co-state must be a list instance."
        u, v = pol
        
        # allocations
        H   = np.zeros((self.T))
        Hx  = np.zeros((self.T, self.DX))
        Hu  = np.zeros((self.T, self.DU))
        Hv  = np.zeros((self.T, self.DV))
        Hvu = np.zeros((self.T, self.DV, self.DU))
        Huv = np.zeros((self.T, self.DU, self.DV))
        Hvv = np.zeros((self.T, self.DV, self.DV))
        Huu = np.zeros((self.T, self.DU, self.DU))
        Hux = np.zeros((self.T, self.DU, self.DX))
        Hvx = np.zeros((self.T, self.DV, self.DX))
        Hxx = np.zeros((self.T, self.DX, self.DX))


        # computations  #TODOs: Broadcast to appropriate Dims
        H = p[0]*states[2] + p[1]*states[3] + \
             self.a * p[2]*(np.cos(u)-np.cos(v)) + \
             self.a * p[3]*(np.sin(u)-np.sin(v)) 
        
        Hu = -p[2] * self.a * np.sin(u) + p[3] * self.a * np.cos(u)         
        Huu = -p[2] * self.a * np.cos(u) - p[3] * self.a * np.sin(u)    
        
        Hv = p[2] * self.a * np.sin(v) - p[3] * self.a * np.cos(v)         
        Hvv = p[2] * self.a * np.cos(v) + p[3] * self.a * np.sin(v)    
        

        Hx[2,0] = p[0]
        Hx[3,0] = p[1]

        return Bundle(dict(H=H, Hx=Hx, Hu=Hu, Hv=Hv, \
                            Huu=Huu, Hvv=Hvv, Huv=Huv, Hvx=Hvx, \
                            Hvu=Hvu, Hux=Hux, Hxx=Hxx))

    def value_funcs(self, x):
        """
            Compute the Value function as well as its partial
            derivatives. Note that the value function is the 
            target set parameteized by time and state.

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
            Output:
              .cost -- The terminal cost.
        """
        V = x[0]**2+x[1]**2-self.capture_rad**2
        
        #first order derivative
        V  = np.zeros((self.T))
        Vx = np.zeros((self.T, self.DX))
        Vxx = np.zeros((self.T, self.DX, self.DX))

        # Todos: Fix the broadcast of V's w.r.t to state
        Vx[0] = 2*x[0]
        Vx[1] = 2*x[1]

        #second order derivative
        Vxx[0,0] = 2
        Vxx[1,1] = 2
        
        return V, Vx, Vxx
    