import cupy as cp
import numpy as np

class chambollepock(object):
    def __init__(self):
        """
          This is a first order primal-dual hybrid-gradient method for non-smooth convex optimization problems with known saddle-point structure

            \max_{y \in Y} \min_{x \in X} \big( \langle L x, y\rangle_Y + g(x) - f^*(y) \big) ,

            where X and Y are Hilbert spaces with inner product \langle\cdot,\cdot\rangle and norm \|.\|_2 = \langle\cdot,\cdot\rangle^{1/2}, L is a continuous linear operator L: X \to Y, g: X \to [0,+\infty] and f: Y \to [0,+\infty] are proper, convex and lower semi-continuous functionals, and f^* is the convex (or Fenchel) conjugate of f, (see convex conjugate).

            The saddle-point problem is a primal-dual formulation of the primal minimization problem

            \min_{x \in X} \big( g(x) + f(L x) \big).

            The corresponding dual maximization problem is

            \max_{y \in Y} \big( g^*(-L^* x) - f^*(y) \big)

            with L^* being the adjoint of the operator L.
        """
        
        print('Type chambollepock.help() for info on how to use this class!')

        self.proxops = proxops()
        self.variables = cpk_variables()
        self.applyK = cpk_applyK()
        self.optimizer = cpk_optimizer(self)
        self.define = cpk_define(self)

    def help(self):
        print('A first-order primal-dual algorithm for convex problems'
              'with applications to imaging by Antonin Chambolle, Thomas Pock'
               'Authors: Dan Nguyen, Lekan Molux, 2018.'
               'UT Southwestern Medical Center, MAIA Lab.')

class cpk_variables(object):
    def __init__(self):
        self.K = []
        self.Kc = []
        self.pprox = None
        self.dprox = []
        self.shapex = None
        self.shapez = []
        self.tau = None
        self.itau = None
        self.sig = []
        self.isig = []

        self.costfunc = None
        self.maxiter = 500

        self.x = None
        self.z = []

        self.dtype = cp.float32
        self.theta = cp.array([1],dtype=self.dtype)

        self.gpudevice = 0
        cp.cuda.Device(self.gpudevice).use()

    def reset_variables(self):
        self.K = []
        self.Kc = []
        self.pprox = None
        self.dprox = []
        self.shapex = None
        self.shapez = []
        self.tau = None
        self.itau = None
        self.sig = []
        self.isig = []

        self.costfunc = None
        self.maxiter = 500

        self.x = None
        self.z = []
        self.theta = cp.array([1],dtype=self.dtype)

    def set_gpu(self,gpudevice=0):
        self.gpudevice = gpudevice
        cp.cuda.Device(self.gpudevice).use()

    def check_variables(self):
        if not self.K:
            print('K        : NOT Defined')
        else:
            print('K        : Defined')

        if not self.Kc:
            print('Kc       : NOT Defined')
        else:
            print('Kc       : Defined')

        if self.pprox is None:
            print('pprox    : NOT Defined')
        else:
            print('pprox    : Defined')

        if not self.dprox:
            print('dprox    : NOT Defined')
        else:
            print('dprox    : Defined')

        if self.shapex is None:
            print('shapex   : NOT Defined')
        else:
            print('shapex   : Defined')

        if not self.shapez:
            print('shapez   : NOT Defined')
        else:
            print('shapez   : Defined')

        if self.tau is None:
            print('tau      : NOT Defined')
        else:
            print('tau      : Defined')

        if self.itau is None:
            print('itau     : NOT Defined')
        else:
            print('itau     : Defined')

        if not self.sig:
            print('sig      : NOT Defined')
        else:
            print('sig      : Defined')

        if not self.isig:
            print('isig     : NOT Defined')
        else:
            print('isig     : Defined')

        if self.costfunc is None:
            print('costfunc : NOT Defined')
        else:
            print('costfunc : Defined')

        print('gpu dev  :', self.gpudevice)
        print('maxiter  :', self.maxiter)

    def init_opt_variables(self):

        self.x=cp.zeros(self.shapex,dtype=self.dtype)
        self.z=[cp.zeros(self.shapez[ii],dtype=self.dtype) for ii in range(len(self.shapez))]

class cpk_applyK(object):
    def matrix_multiply(self, A):
        def buff(x):
            return A.dot(x)
        return buff

class cpk_define(object):
    def __init__(self,parentself):
        self.parent = parentself

    def K_datamatrix(self,Kmat,sparsemat=False,sparsetype='csr'):
        if not isinstance(Kmat,list):
            Kmat = [Kmat]

        #this extra loop is added so that tau and sigma are calculaed with the cpu K
        #this is done because of certain instability issues on CuPy Sparse
        sig=[]
        isig=[]
        itau=np.zeros(Kmat[0].shape[1])
        self.parent.variables.shapex = Kmat[0].shape[1]
        for ii in range(len(Kmat)):
            self.parent.variables.shapez.append(Kmat[ii].shape[0])
            itau += np.array(np.sum(np.abs(Kmat[ii]),axis=0)).reshape(-1)
            isig.append(np.array(np.sum(np.abs(Kmat[ii]),axis=1)).reshape(-1))
            sig.append(np.divide(1, isig[ii], where=(np.array(isig[ii]) != 0)).reshape(-1))

            self.parent.variables.isig.append(cp.array(isig[ii]))
            self.parent.variables.sig.append(cp.array(sig[ii]))

        self.parent.variables.itau=cp.array(itau)
        self.parent.variables.tau = cp.array(np.divide(1,itau,where=(np.array(itau) != 0)).reshape(-1))

        for ii in range(len(Kmat)):
            # if self.parent.variables.shapex is None:
            #     self.parent.variables.shapex = Kmat[ii].shape[1]
            # if self.parent.variables.itau is None:
            #     self.parent.variables.itau = cp.zeros(Kmat[ii].shape[1],dtype=self.parent.variables.dtype)
            # self.parent.variables.shapez.append(Kmat[ii].shape[0])

            if sparsemat:
                if sparsetype is 'coo':
                    K_gpu = cp.sparse.coo_matrix(Kmat[ii],dtype=self.parent.variables.dtype)
                elif sparsetype is 'csc':
                    K_gpu = cp.sparse.csc_matrix(Kmat[ii],dtype=self.parent.variables.dtype)
                elif sparsetype is 'csr':
                    K_gpu = cp.sparse.csr_matrix(Kmat[ii],dtype=self.parent.variables.dtype)
                else:
                    print('invalid sparse type designation. defaulting to csr.')
                    K_gpu = cp.sparse.csr_matrix(Kmat[ii],dtype=self.parent.variables.dtype)
            else:
                K_gpu = cp.array(Kmat[ii],dtype=self.parent.variables.dtype)

            self.parent.variables.K.append(self.parent.applyK.matrix_multiply(K_gpu))

            # if sparsemat:
            #     if sparsetype is 'coo':
            #         self.parent.variables.Kc.append(self.parent.applyK.matrix_multiply(K_gpu.transpose()))
            #     elif sparsetype is 'csc':
            #         self.parent.variables.Kc.append(self.parent.applyK.matrix_multiply(K_gpu.transpose().tocsc()))
            #     else:
            #         self.parent.variables.Kc.append(self.parent.applyK.matrix_multiply(K_gpu.transpose().tocsr()))
            # else:
            #     self.parent.variables.Kc.append(self.parent.applyK.matrix_multiply(K_gpu.transpose()))

            #Rewritten this way because of instability issues in CuPy Sparse. Doubles RAM.
            if sparsemat:
                if sparsetype is 'coo':
                    K_gpu_t = cp.sparse.coo_matrix(Kmat[ii].transpose(), dtype=self.parent.variables.dtype)
                elif sparsetype is 'csc':
                    K_gpu_t = cp.sparse.csc_matrix(Kmat[ii].transpose(), dtype=self.parent.variables.dtype)
                else:
                    K_gpu_t = cp.sparse.csr_matrix(Kmat[ii].transpose(), dtype=self.parent.variables.dtype)
                self.parent.variables.Kc.append(self.parent.applyK.matrix_multiply(K_gpu_t))
            else:
                self.parent.variables.Kc.append(self.parent.applyK.matrix_multiply(K_gpu.transpose()))

        #     self.parent.variables.itau += abs(K_gpu).sum(axis=0).reshape(-1)
        #     self.parent.variables.isig.append(abs(K_gpu).sum(axis=1).reshape(-1))
        #     self.parent.variables.sig.append(cp.divide(1, self.parent.variables.isig[-1]).reshape(-1))
        #     self.parent.variables.sig[-1][cp.nonzero(cp.isinf(self.parent.variables.sig[-1]))] = 0
        # self.parent.variables.tau = cp.divide(1,self.parent.variables.itau).reshape(-1)
        # self.parent.variables.tau[cp.nonzero(cp.isinf(self.parent.variables.tau))] = 0


    def K(self):
        pass

    def dualprox(self,operation=None,b=None,w=None,mode='append'):
        if (mode != 'append') and (mode != 'overwrite'):
            print('invalid argument! mode can either be set as "append" or "overwrite')
        elif mode == 'overwrite':
            self.parent.variables.dprox = []

        if not isinstance(operation,list):
            operation = [operation]

        if b is None:
            b = [None] * len(operation)
        if w is None:
            w = [None] * len(operation)

        for ii in range(len(operation)):
            if isinstance(operation[ii], str):
                if operation[ii] in self.parent.proxops.avail_ops:
                    self.parent.variables.dprox.append(self.parent.proxops.get_operation(operation[ii],b[ii],w[ii]))
                else:
                    print('Built-in operation dose not exist. Existing built-in operations are:')
                    print(self.parent.proxops.avail_ops)
            elif callable(operation[ii]):
                self.parent.variables.dprox.append(operation[ii])
            else:
                print('Input operation not a string or function. Skipping ' + ii + 'th item.')

    def primalprox(self,operation=None,b=None,w=None):
        if not isinstance(operation,list):
            operation = [operation]

        if self.parent.variables.pprox is not None:
            print('Warning! Primal prox operator was already defined. Overwriting with new prox operator.')

        if b is None:
            b = [None] * len(operation)
        if w is None:
            w = [None] * len(operation)

        if not isinstance(b,list):
            b = [b]
        if not isinstance(w,list):
            w = [w]

        for ii in range(len(operation)):
            if isinstance(operation[ii], str):
                if operation[ii] in self.parent.proxops.avail_ops:
                    self.parent.variables.pprox=self.parent.proxops.get_operation(operation[ii],b[ii],w[ii])
                else:
                    print('Built-in operation dose not exist. Existing built-in operations are:')
                    print(self.parent.proxops.avail_ops)
            elif callable(operation[ii]):
                self.parent.variables.pprox=operation[ii]
            else:
                print('Input operation not a string or function. Setting primal prox operator to None.')
                self.parent.variables.pprox = None

class cpk_optimizer(object):
    def __init__(self,parentself):
        self.parent = parentself


    def lazy_cpk(self,continue_opt=False):
        # initialize primal and dual variables

        if (not continue_opt) or (self.parent.variables.x is None) or (not self.parent.variables.z):
            self.parent.variables.x = cp.zeros(self.parent.variables.shapex,dtype=self.parent.variables.dtype)
            self.parent.variables.z = [cp.zeros(self.parent.variables.shapez[ii],dtype=self.parent.variables.dtype) for ii in range(len(self.parent.variables.shapez))]

        xp = cp.zeros_like(self.parent.variables.x,dtype=self.parent.variables.dtype)
        xhat = cp.zeros_like(self.parent.variables.x,dtype=self.parent.variables.dtype)
        buffx = cp.zeros_like(self.parent.variables.x,dtype=self.parent.variables.dtype)

        # initalize cost vector (just keeping track)
        if self.parent.variables.costfunc is not None:
            cost = cp.zeros(self.parent.variables.maxiter,dtype=self.parent.variables.dtype)
        else:
            cost = None

        # run algorithm
        for ii in range(self.parent.variables.maxiter):
            # update x
            buffx.fill(0)
            for jj in range(len(self.parent.variables.z)):
                buffx += self.parent.variables.Kc[jj](self.parent.variables.z[jj])
            xp[:] = self.parent.variables.pprox(self.parent.variables.x - cp.multiply(self.parent.variables.tau, buffx), self.parent.variables.tau)

            # overshoot x
            xhat[:] = xp + self.parent.variables.theta * (xp - self.parent.variables.x)

            # update z
            for jj in range(len(self.parent.variables.z)):
                self.parent.variables.z[jj][:] = self.parent.variables.dprox[jj](self.parent.variables.z[jj] + cp.multiply(self.parent.variables.sig[jj], self.parent.variables.K[jj](xhat)), self.parent.variables.sig[jj])

            # overwrite last iteration x
            self.parent.variables.x[:] = xp

            if self.parent.variables.costfunc is not None:
                cost[ii] = self.parent.variables.costfunc(self.parent.variables.K,self.parent.variables.x)

        return (self.parent.variables.x,cost)

class proxops(object):
    def __init__(self):
        self.avail_ops = ['l2s','l2sp','l2sm',
                          'cc_l2s','cc_l2sp','cc_l2sm',
                          'l1','l1p','l1m',
                          'cc_l1','cc_l1p','cc_l1m']

    def get_operation(self,str_op,b=None,w=None):
        if (str_op=='l2s'):
            return self.prox_l2s(b=b,w=w)
        if (str_op=='l2sp'):
            return self.prox_l2sp(b=b,w=w)
        if (str_op=='l2sm'):
            return self.prox_l2sm(b=b,w=w)

        if (str_op=='cc_l2s'):
            return self.prox_cc_l2s(b=b,w=w)
        if (str_op=='cc_l2sp'):
            return self.prox_cc_l2sp(b=b,w=w)
        if (str_op=='cc_l2sm'):
            return self.prox_cc_l2sm(b=b,w=w)

        if (str_op=='l1'):
            return self.prox_l1(b=b,w=w)
        if (str_op=='l1p'):
            return self.prox_l1p(b=b,w=w)
        if (str_op=='l1m'):
            return self.prox_l1m(b=b,w=w)

        if (str_op=='cc_l1'):
            return self.prox_cc_l1(b=b,w=w)
        if (str_op=='cc_l1p'):
            return self.prox_cc_l1p(b=b,w=w)
        if (str_op=='cc_l1m'):
            return self.prox_cc_l1m(b=b,w=w)

    def moreau_decomposition(self,prox_operation=None):
        def buff(x,t):
            return x - cp.multiply(t,prox_operation(cp.divide(x,t),cp.divide(1,t)))
        return buff

    def prox_l2s(self,b=None,w=None):
        if (b is not None) and (w is not None):
            # print('Using assigned b and w')
            def buff(x,t):
                return cp.divide(x + w*cp.multiply(t,b),1+w*t)
            return buff
        elif (b is not None) and (w is None):
            # print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return cp.divide(x + cp.multiply(t,b),1+t)
            return buff
        elif (b is None) and (w is not None):
            # print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return cp.divide(x,1+w*t)
            return buff
        else:
            # print('Defaulting b = 0 and w = 1')
            def buff(x, t):
                return cp.divide(x,1+t)
            return buff
    def prox_l2sp(self,b=None,w=None):
        if (b is not None) and (w is not None):
            # print('Using assigned b and w')
            def buff(x,t):
                return cp.minimum(x, cp.divide(x + w*cp.multiply(t,b),1+w*t))
            return buff
        elif (b is not None) and (w is None):
            # print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return cp.minimum(x, cp.divide(x + cp.multiply(t,b),1+t))
            return buff
        elif (b is None) and (w is not None):
            # print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return cp.minimum(x, cp.divide(x,1+w*t))
            return buff
        else:
            # print('Defaulting b = 0 and w = 1')
            def buff(x, t):
                return cp.minimum(x, cp.divide(x,1+t))
            return buff
    def prox_l2sm(self,b=None,w=None):
        if (b is not None) and (w is not None):
            # print('Using assigned b and w')
            def buff(x,t):
                return cp.maximum(x, cp.divide(x + w*cp.multiply(t,b),1+w*t))
            return buff
        elif (b is not None) and (w is None):
            # print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return cp.maximum(x, cp.divide(x + cp.multiply(t,b),1+t))
            return buff
        elif (b is None) and (w is not None):
            # print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return cp.maximum(x, cp.divide(x,1+w*t))
            return buff
        else:
            # print('Defaulting b = 0 and w = 1')
            def buff(x, t):
                return cp.maximum(x, cp.divide(x,1+t))
            return buff

    def prox_cc_l2s(self,b=None,w=None):
        if (b is not None) and (w is not None):
            # print('Using assigned b and w')
            def buff(x,t):
                return w*cp.divide(x - cp.multiply(t,b),w+t)
            return buff
        elif (b is not None) and (w is None):
            # print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return cp.divide(x - cp.multiply(t, b),1+t)
            return buff
        elif (b is None) and (w is not None):
            # print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return w * cp.divide(x, w + t)
            return buff
        else:
            # print('Defaulting b = 0 and w = 1')
            def buff(x, t):
                return cp.divide(x, 1 + t)
            return buff
    def prox_cc_l2sp(self,b=None,w=None):
        if (b is not None) and (w is not None):
            # print('Using assigned b and w')
            def buff(x,t):
                return cp.maximum(0, w*cp.divide(x - cp.multiply(t,b),w+t))
            return buff
        elif (b is not None) and (w is None):
            # print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return cp.maximum(0, cp.divide(x - cp.multiply(t, b),1+t))
            return buff
        elif (b is None) and (w is not None):
            # print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return cp.maximum(0, w*cp.divide(x,w+t))
            return buff
        else:
            # print('Defaulting b = 0 and w = 1')
            def buff(x, t):
                return cp.maximum(0, cp.divide(x,1+t))
            return buff
    def prox_cc_l2sm(self,b=None,w=None):
        if (b is not None) and (w is not None):
            # print('Using assigned b and w')
            def buff(x,t):
                return cp.minimum(0, w*cp.divide(x - cp.multiply(t,b),w+t))
            return buff
        elif (b is not None) and (w is None):
            # print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return cp.minimum(0, cp.divide(x - cp.multiply(t, b),1+t))
            return buff
        elif (b is None) and (w is not None):
            # print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return cp.minimum(0, w*cp.divide(x,w+t))
            return buff
        else:
            # print('Defaulting b = 0 and w = 1')
            def buff(x, t):
                return cp.minimum(0, cp.divide(x,1+t))
            return buff

    def prox_l1(self,b=None,w=None):
        if (b is not None) and (w is not None):
            # print('Using assigned b and w')
            def buff(x,t):
                return x - cp.minimum(w*t,cp.maximum(-w*t,x-b))
            return buff
        elif (b is not None) and (w is None):
            print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return x - cp.minimum(t,cp.maximum(-t,x-b))
            return buff
        elif (b is None) and (w is not None):
            print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return x - cp.minimum(w*t,cp.maximum(-w*t,x))
            return buff
        else:
            print('Defaulting b = 0 and w = 1')
            def buff(x, t):
                return x - cp.minimum(t,cp.maximum(-t,x))
            return buff
    def prox_l1p(self,b=None,w=None):
        if (b is not None) and (w is not None):
            print('Using assigned b and w')
            def buff(x,t):
                return x - cp.minimum(w*t,cp.maximum(0,x-b))
            return buff
        elif (b is not None) and (w is None):
            print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return x - cp.minimum(t,cp.maximum(0,x-b))
            return buff
        elif (b is None) and (w is not None):
            print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return x - cp.minimum(w*t,cp.maximum(0,x))
            return buff
        else:
            print('Defaulting b = 0 and w = 1')
            def buff(x, t):
                return x - cp.minimum(t,cp.maximum(0,x))
            return buff
    def prox_l1m(self,b=None,w=None):
        if (b is not None) and (w is not None):
            print('Using assigned b and w')
            def buff(x,t):
                return x - cp.minimum(0,cp.maximum(-w*t,x-b))
            return buff
        elif (b is not None) and (w is None):
            print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return x - cp.minimum(0,cp.maximum(-t,x-b))
            return buff
        elif (b is None) and (w is not None):
            print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return x - cp.minimum(0,cp.maximum(-w*t,x))
            return buff
        else:
            print('Defaulting b = 0 and w = 1')
            def buff(x, t):
                return x - cp.minimum(0,cp.maximum(-t,x))
            return buff

    def prox_cc_l1(self,b=None,w=None):
        if (b is not None) and (w is not None):
            print('Using assigned b and w')
            def buff(x,t):
                return cp.minimum(w,cp.maximum(-w,x-cp.multiply(t,b)))
            return buff
        elif (b is not None) and (w is None):
            print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return cp.minimum(1,cp.maximum(-1,x-cp.multiply(t,b)))
            return buff
        elif (b is None) and (w is not None):
            print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return cp.minimum(w,cp.maximum(-w,x))
            return buff
        else:
            print('Defaulting b = 0 and w = 1')
            def buff(x,t):
                return cp.minimum(1,cp.maximum(-1,x))
            return buff
    def prox_cc_l1p(self,b=None,w=None):
        if (b is not None) and (w is not None):
            print('Using assigned b and w')
            def buff(x,t):
                return cp.minimum(w,cp.maximum(0,x-cp.multiply(t,b)))
            return buff
        elif (b is not None) and (w is None):
            print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return cp.minimum(1,cp.maximum(0,x-cp.multiply(t,b)))
            return buff
        elif (b is None) and (w is not None):
            print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return cp.minimum(w,cp.maximum(0,x))
            return buff
        else:
            print('Defaulting b = 0 and w = 1')
            def buff(x,t):
                return cp.minimum(1,cp.maximum(0,x))
            return buff
    def prox_cc_l1m(self,b=None,w=None):
        if (b is not None) and (w is not None):
            print('Using assigned b and w')
            def buff(x,t):
                return cp.minimum(0,cp.maximum(-w,x-cp.multiply(t,b)))
            return buff
        elif (b is not None) and (w is None):
            print('Using assigned b. Defaulting w = 1')
            def buff(x,t):
                return cp.minimum(0,cp.maximum(-1,x-cp.multiply(t,b)))
            return buff
        elif (b is None) and (w is not None):
            print('Using assigned w. Defaulting b = 0')
            def buff(x,t):
                return cp.minimum(0,cp.maximum(-w,x))
            return buff
        else:
            print('Defaulting b = 0 and w = 1')
            def buff(x,t):
                return cp.minimum(0,cp.maximum(-1,x))
            return buff

    def prox_l0(self):
        pass

    def prox_lhalf(self):
        pass

    def prox_I_lb_ub(self,lb=None,ub=None):
        if (lb is not None) and (ub is not None):
            def buff(x,t):
                return cp.minimum(ub,cp.maximum(lb,x))
            return buff
        elif (lb is not None) and (ub is None):
            def buff(x,t):
                return cp.maximum(lb,x)
            return buff
        elif (lb is None) and (ub is not None):
            def buff(x,t):
                return cp.minimum(ub,x)
            return buff
        else:
            def buff(x,t):
                return x
            return buff