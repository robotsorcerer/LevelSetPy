__comment__ =  "Old Cancer treatment planning code. Use at thine own risk."
__author__  =  "Olalekan Ogunmolu, UTSW MAIA Lab, 2018"

import time
import logging
import scipy as sp
import numpy as np

from scipy import linalg as LA
from scipy.sparse import linalg as LAS

logger = logging.getLogger(__name__)


class ADMM(object):

    """Solve for arg min || Ax - b ||_2"""

    def __init__(self, rhs, disp=False):
        super(ADMM, self).__init__()
        self.ABSTOL   = 1e-4
        self.RELTOL   = 1e-2
        self.A = None
        self.fontdict = {'fontsize':14, 'fontweight':'bold'}

        self.obj_hist       = list()
        self.lhs            = None
        self.rhs            = rhs
        self.options        = {'disp': disp, 'maxiter': 1500}
        self.AtA            = None

    def optimize(self, value_rom=None, rho=1.0, alpha=1.6, \
                 gamma=(np.sqrt(5)+1)/2, \
                 lambder=0.7, A=None):

        print('getting dijs')

        if A is None:
            self.A = self.A
        else:
            self.A = A

        self.AtA = self.A.T @ self.A

        x, z, u = [np.zeros(self.A.shape[-1])]*3
        # store away the ridge regression coefficient
        x_left = self.AtA + rho * cp.eye(self.AtA.shape[0])
        # store away A^Tb
        Atb = self.A.T.dot(self.rhs)

        m,n = self.A.shape

        if self.options['disp']:
            print('%s\t%s\t\t%s\t\t%s\t\t%s\t\t%s\n' %( 'iter #', \
                'r_norm',  'eps_pri', 's_norm', 'eps_dual', 'obj_val'))
        for k in range(self.options['maxiter']):
            # see pg 43; admm paper by boyd
            x_right = Atb + cp.multiply(rho, (z - u))
            cg_res = LA.cg(x_left, x_right, x0=x, tol=1e-05, maxiter=1500, M=None, callback=None)
            if cg_res[1] == 0:
                x = cg_res[0]
            else:
                print('Fatal: unsuccessful inversion with conjugate gradient\nCheck AtA matrix is Hermitian and positive semi definite')
                break

            zold = z
            x_hat = x#alpha*x + (1 - alpha)*zold;

            # project z onto the non-negative orthant # see pg 37 of badmm paper
            z = self.shrinkage(x_hat + u, lambder/rho)
            # u-update: scaled dual
            u = u - gamma * (z - x_hat)

            f = self.objective(x, z)

            r_norm  = LA.norm(x - z, 2)
            s_norm  = LA.norm(-rho*(z - zold), 2)

            eps_pri = np.sqrt(rho)*self.ABSTOL + self.RELTOL*max(LA.norm(x, 2), LA.norm(-z, 2))
            eps_dual= np.sqrt(n)*self.ABSTOL + self.RELTOL*LA.norm(u, 2)

            if self.options['disp']:
                print('%3d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.2f\n' %( k, \
                    r_norm,  eps_pri, s_norm, eps_dual, f))

            if (r_norm < eps_pri and  s_norm < eps_dual):
                break

            self.x = x
            self.z = z

        self.lhs = self.A@x

        return x, z

    def objective(self, x, z):
        self.lhs = self.A @ x

        obj = np.array(self.lhs - self.rhs)
        obj = float(LA.norm(obj))

        # p = float(LA.norm((self.overPenalty * oDose.clip(0)),ord=2) ** 2 +
        #                 LA.norm((self.underPenalty * uDose.clip(-1e-10,0)),ord=2) ** 2)

        return obj

    def shrinkage(self, x, kappa):
        z = np.maximum( 0, x - kappa ) - np.maximum( 0, -x - kappa )
        return z