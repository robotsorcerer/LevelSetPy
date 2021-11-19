__all__ = ['matricize', 'tensor_matrix_mult' ]

from .class_tensor import Tensor

import copy
import numpy as np
from Utilities.matlab_utils import *

# class TensorAlgebra():
#     def __init__(self):
        
def matricize(T, mode=1):
    """
        Matricize a tensor T shaped (3,3, 3) into Folds along axes, `mode`

        Parameters
        ----------
            Tensor T: (i,j,k) shaped array
                Tensor to be unfolded

            mode : str
                The mode along which to unfold the tensor
        Returns
        -------
            X: (i, j) or (j, k) or (i, k) array of unfolded tensor T along mode, `mode`

        Ref: Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications. 
        SIAM Review, 51(3), 455–500. https://doi.org/10.1137/07070111X

        Author: Lekan Molux, Nov 1, 2021
    """
    assert T.ndim <= 3, "We do not support higher order tensors >3 at this moment"

    #1-mode unfold
    if mode=='1':
        X = np.concatenate(( [T[...,i] for i in range(T.shape[-1])]),
                                axis=1)

    #2-mode unfold
    elif mode=='2':
        X = np.concatenate(( [T[...,i].T for i in range(T.shape[-1])]),
                                axis=1)

    #3-mode unfold
    elif mode=='3':
        X = np.concatenate(( [ np.expand_dims(T[..., i].flatten(),
                        axis=1).T for i in range(T.shape[-1])]), axis=0)

    else:
        raise NotImplementedError(f"Cannot decompose a {T.size} sized Tensor along mode {mode}.")

    return X

def dims_check(self, dims=None, N=None, M=None):
    """
        This preprocesses dimensions of a tensor

        Signature:
            newdims, _ = dimscheck(dims, N): Check that the specified dimensions are valid for a tensor

    """
    if dims is None:
        dims = np.arange(N)

    if np.max(dims)<0:
        raise ValueError('negative dims are not accounted for now.')
        #dims = set(matlab_array(1, N)).difference(-dims)

    P = len(dims)

    sorted_dims = np.sort(dims)
    sorted_dims_idx = np.argsort(dims)

    assert M < N, "We cannot have more multiplicands than dimensions"


    assert (M != N) and (M != P), "Invalid number of multiplicands"

    if P==M:
        """
            Number of items in dims and number of multiplicands
            are equal; therefore, index in order of how sorted_dims
            was sorted.
        """
        vidx = copy.copy(sorted_dims_idx)
    else:
        """
            Number of multiplicands is equal to the number of
            dimensions in the tensor; therefore, index multiplicands by
            dimensions specified in dims argument.
        """
        vidx = copy.copy(sorted_dims)

    return sorted_dims, vidx




def tensor_matrix_mult(X, V, n=None, Transpose=False):
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
       Y = tensor_matrix_mult(X, {A,B,C,D}, [1 2 3 4]) <-- 4-way multiply
       Y = tensor_matrix_mult(X, {D,C,B,A}, [4 3 2 1]) <-- same as above
       Y = tensor_matrix_mult(X, {A,B,C,D})            <-- same as above
       Y = tensor_matrix_mult(X, {A',B',C',D'}, 't')   <-- same as above
       Y = tensor_matrix_mult(X, {C,D}, [3 4])     <-- X times C in mode-3 & D in mode-4
       Y = tensor_matrix_mult(X, {A,B,C,D}, [3 4]) <-- same as above
       Y = tensor_matrix_mult(X, {A,B,D}, [1 2 4])   <-- 3-way multiply
       Y = tensor_matrix_mult(X, {A,B,C,D}, [1 2 4]) <-- same as above
       Y = tensor_matrix_mult(X, {A,B,D}, -3)        <-- same as above
       Y = tensor_matrix_mult(X, {A,B,C,D}, -3)      <-- same as above
       
       Author: Lekan Molux, November 1, 2021
    """      
    if isinstance(X, Tensor):
        # Do all ops on numpy or np.array
        X = X.data
    
    if n is None:      
        n = np.arange(X.ndim)
        
    if isinstance(V, list) or isinstance(V, tuple):
        dims = n
        dims,vidx = dims_check(dims,X.ndim,numel(V))
        
        # Calc individual tensor products
        Y = tensor_matrix_mult(X, V[idx[0]], dims[0], Transpose)
        
        for k in range(1, numel(dims)):
            Y = tensor_matrix_mult(Y, V[idx[k]], dims[k], Transpose)
        return Y

    if V.ndim>2:
        raise ValueError(f'Tensor by Matrix multiplication does not support non-matrix as second argument.')

    if (numel(n)!=1 or (n<0) or n > X.ndim):
        error(f'Dimension N of Tensor, must be between 1 and {X.ndim}.');

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
        nblocks = np.prod(sz[n+1:N])
        ncols   = np.prod(sz[:n-1])
        nAk = sz[n] * ncols
        nBk = p  *  ncols
        B = np.zeros(p * nblocks * ncols, 1)

        for k in range(nblocks):
            Akt = X[k * nAk: k*nAK].reshape(ncols, sz[n])
            if Transpose:
                Bkt = Akt @ V
            else:
                Bkt = Akt @ V.T
            B[(k-1)*nBk + 1: k * nBk] = np.ravel(Bkt)
    newsz = copy.copy(sz)
    newsz[n] = p
    
    Y = Tensor(B, tuple(newsz))

    return Y