__all__ = ['tensor_matrix_mult']

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Tensor Algebra"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Finished"

from .class_tensor import Tensor
import copy
import numpy as np
import numpy as np
from Utilities.matlab_utils import numel
from .tensor_utils import dims_check, use_gpu

def tensor_matrix_mult(X, V, n=None, Transpose=False, use_gpu=True):
    """
    tensor_matrix_mult: Tensor times matrix.

        Parameters
        ----------
        X: Tensor with N modes
        V: 2D array
        n: mode of tensor X to which to multiply the matrix by
        Transpose: Should we rotate the matrix before multiplying? If we do rotate,
                    then the tensor product is carried out from left to right in order;
                    otherwise, it is carried out in order from right to left.

        Returns
        -------
        T: A Tensor class which is the product of the
            Tensor-Matrix multiplication.

       Y = tensor_matrix_mult(X,A,N) computes the n-mode product of tensor X with a
       matrix A; i.e., X x_N A.  The integer N specifies the dimension
       (or mode) of X along which A should be multiplied.  If shape(A) =
       (J,I), then X must have shape(X[N]) = I.  The result will be the
       same order and shape as X except that shape(Y[N]) = J.

       Y = tensor_matrix_mult(X,[A,B,C,...]) computes the n-mode product of tensor X
       with a sequence of matrices in the list of array.  The n-mode
       products are computed sequentially along all dimensions (or modes)
       of X. The list of arrays contain X.ndim matrices.

       Y = tensor_matrix_mult(X,[A,B,C,...],DIMS) computes the sequence tensor-matrix
       products along the dimensions specified by DIMS.

       Y = tensor_matrix_mult(...,'T') performs the same computations as above except
       the matrices are transposed.

       Examples
       import numpy.random as np.
       X = np..rand(5,3,4,2)
       A = np..rand(4,5); B = np..rand(4,3); C = np..rand(3,4); D = np..rand(3,2);
       Y = tensor_matrix_mult(X, A, 1)         <-- computes X times A in mode-1
       Y = tensor_matrix_mult(X, [A,B,C,D], 1) <-- same as above
       Y = tensor_matrix_mult(X, A.T, 1, Transpose)   <-- same as above
       Y = tensor_matrix_mult(X, [A,B,C,D], [1, 2, 3, 4]) <-- 4-way multiply
       Y = tensor_matrix_mult(X, [D,C,B,A], [4, 3, 2, 1]) <-- same as above
       Y = tensor_matrix_mult(X, [A,B,C,D])            <-- same as above
       Y = tensor_matrix_mult(X, [A',B',C',D'], Transpose=True)   <-- same as above
       Y = tensor_matrix_mult(X, [C,D], [3, 4])     <-- X times C in mode-3 & D in mode-4
       Y = tensor_matrix_mult(X, [A,B,C,D], [3, 4]) <-- same as above
       Y = tensor_matrix_mult(X, [A,B,D], [1, 2, 4])   <-- 3-way multiply
       Y = tensor_matrix_mult(X, [A,B,C,D], [1, 2, 4]) <-- same as above
       Y = tensor_matrix_mult(X, [A,B,D], -3)        <-- same as above
       Y = tensor_matrix_mult(X, [A,B,C,D], -3)      <-- same as above

       Author: Lekan Molux, November 1, 2021
    """
    if isinstance(X, Tensor):
        # Do all ops on numpy or np.array
        X = X.data

    if use_gpu:
        # Do it on gpu if available
        X = np.asarray(X)
        'when we are multiplying with multiple arrays'
        'be careful that we do not give cupy object dtypes'
        V = np.asarray(V) if V.dtype!='O' else V

    if n is None:
        n = np.arange(X.ndim)

    if V.dtype=='O':

        dims = n
        dims,vidx = dims_check(dims,X.ndim,numel(V))

        # Calc individual tensor products
        if use_gpu:
            Y = tensor_matrix_mult(X, np.asarray(V[vidx[0]]), dims[0], Transpose)
        else:
            Y = tensor_matrix_mult(X, V[vidx[0]], dims[0], Transpose)

        for k in range(1, numel(dims)):
            if use_gpu:
                Y = tensor_matrix_mult(Y, np.asarray(V[vidx[k]]), dims[k], Transpose)
            else:
                Y = tensor_matrix_mult(Y, V[vidx[k]], dims[k], Transpose)
        return Y

    if V.ndim>2:
        raise ValueError(f'Tensor by Matrix multiplication does not support non-matrix as second argument.')

    if (numel(n)!=1 or (n<0) or n > X.ndim):
        raise ValueError(f'Dimension N: {N} of Tensor, must be between 1 and {X.ndim}.');

    # Get Single n-mode product
    N = X.ndim
    sz = list(X.shape)

    if n==N:
        raise ValueError(f"n: {n} cannot be same size as Tensor dimensions: {N}"
                         f"n  should be at most {X.ndim-1}")

    if Transpose:
        p = V.shape[1]
    else:
        p = V.shape[0]

    if np.isscalar(n) and n==0:
        A = X.reshape(sz[n], -1)
        if Transpose:
            B = V.T@A
        else:
            B = V@A
    elif np.isscalar(n) and n==N-1:
        At = X.reshape(-1, sz[n])
        if Transpose:
            B = At@V
        else:
            B = At@V.T
    else:
        to_tensor = False
        nblocks   = int(np.prod(sz[n+1:]))
        ncols = int(np.prod(sz[:n]))
        nAk       = sz[n] * ncols
        nBk       = p  *  ncols
        B         = np.zeros((p * nblocks * ncols, 1)) if use_gpu else np.zeros((p * nblocks * ncols, 1))

        for k in range(nblocks):
            # Extract k-th sub-block of A (in column-major order)
            Akt_slice = slice((k) * nAk, (k+1)*nAk)
            Akt = X.flatten()[Akt_slice].reshape(ncols, sz[n])

            if Transpose:
                Bkt = Akt @ V
            else:
                Bkt =  Akt @ V.T

            B[k*nBk: (k+1) * nBk] = Bkt.ravel().reshape(-1, 1)

    newsz = copy.copy(sz)
    newsz[n] = p

    # put it back in a tensor format
    if use_gpu:
        Y = Tensor(B.get(), tuple(newsz))
    else:
        Y = Tensor(B, tuple(newsz))

    return Y
