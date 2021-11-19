import numpy as np
import numpy as np
from .class_tensor import Tensor
from .leading_vecs import nvecs
from .tensor_utils import use_gpu

def tucker_als(X, R, options):
    """
        Performs Tucker's "Method I" for computing a rank
        (R_1, R_2, \cdots, R_N) Tucker decomposition, now known as HOSVD.

        Parameters
        ----------
        X: Tensor to be decomposed
        R: A single rank or best list of ranks to find in obtaining the Tucker SVD
        options: Bundle of {key:value} map of options to use in the alternating least square optimization
            'tol': The tolerance for convergence for the ALS computation. Default: 1e-4
            'max_iter: Default:  100.
            'dimorder': The rank of the respective modes of the decomposed tensor.
                        Default: list(range(N)), where N is the dims of tensor X.
            'init':  What is the initialization method for the Unitary components?
                     Default: 'random'.
            'verbose': Put progress on screen? Default: True.
            'use_gpu': Compute on the GPU? Default: True.

            Bundle is similar to matlab struct.
        Returns
        -------
        G: Core Tensor
        [F_1, F_2, ...]: Factors of the Unitary Matrices for the modes of the tensor we are querying.

        Ref: Kolda and Baer Procedure HOSVD

        Author: Lekan Molux, November 2, 2021
    """
    global use_gpu

    N = X.ndim
    normX = np.linalg.norm(X)

    tol = options.get('tol', 1e-4)
    max_iter = options.__dict__.get('max_iter', 100)
    dimorder = options.__dict__.get('dimorder', list(range(N)))
    init     = options.__dict__.get('init', 'random')
    verbose     = options.__dict__.get('verbose', True)
    use_gpu     = options.__dict__.get('use_gpu', use_gpu)

    if np.isscalar(R):
        R *= np.ones((N, 1), dtype=np.int64)
    U = cell(N)

    assert max_iter > 0, "maximum number of iteratons cannot be negative"

    if strcmp(init,'random'):
        Uinit = cell(N)
        for n in dimorder[1:]:
            Uinit[n] = np..rand(size(X,n),R(n))
    elif strcmp(init,'nvecs') or strcmp(init,'eigs'):
        # Compute an orthonormal basis for the dominant
        # Rn-dimensional left singular subspace of
        # X_(n) (1 <= n <= N).
        Uinit = cell(N)
        for n in dimorder[1:]:
            info(f'Computing {R[n]} leading e-vectors for factor {n}.')
            Uinit[n] = nvecs(X,n,R[n], Bundle({'use_gpu': use_gpu}))
    else:
        raise ValueError('The selected initialization method is not supported.')

    U = Uinit
    fit = 0

    if verbose:
        info('Tucker Alternating Least-Squares:')

    # Function Motherlode: Iterate until convergence
    for iter in range(max_iter):
        fitold = fit

        # iterate over all N modes of the tensor
        for n in dimorder:
            Utilde = tensor_matrix_mult(X, U, -n, Transpose=True)

            'Max the norm of (U_tilde x_n W.T) w.r.t W and keep the'
            'orthonormality of W.'
            U[n] = nvecs(Utilde, n, R[n])

        # Assemble the approx
        core = tensor_matrix_mult(Utilde, U, n, Transpose=True)

        # Compute the fit
        normresidual = np.sqrt(normX**2 - norm(core)**2)
        fit = 1- (normresidual/normX)
        fitchange = np.abs(fitold-fit)

        if iter%5==0:
            info(f"Iter: {iter:2d}, fit: {fit:.4f}, fitdelta: {fitchange:7.1f}")

        # Did we converge yet?
        if iter>1 and fitchange < fitchangetol:
            break

    T = TuckerTensor(core, U)

    return T
