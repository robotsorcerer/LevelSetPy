__all__ [""]
__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Tensor Algebra"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"


import copy
import numpy as np
import numpy as np
from Utilities import *
from .tensor_utils import dims_check
from .class_tensor import KruskalTensor
from .tensor_utils import use_gpu

def  ktensor_matrix_mult(X,V,**args):
    """
    TTM Tensor times matrix of a ktensor.

        Parameters
        ----------
        X: Tensor with N modes
        V: 2D array
        args: extra arguments i.e. dims,

        Returns
        -------
        T: A Kruskal Tensor class which is the product of the
            Tensor-Matrix multiplication.

    A KTENSOR Tensor is stored as a Kruskal operator (decomposed).

       Y = TTM(X,A,N) computes the n-mode product of the ktensor X with a
       matrix A; i.e., X x_N A.  The integer N specifies the dimension
       (or mode) of X along which A should be multiplied.  If size(A) =
       [J,I], then X must have size(X,N) = I.  The result will be a
       ktensor of the same order and size as X except that size(Y,N) = J.

       Y = TTM(X,[A,B,C,...]) computes the n-mode product of the ktensor
       X with a sequence of matrices in the cell array.  The n-mode
       products are computed sequentially along all dimensions (or modes)
       of X. The cell array contains ndims(X) matrices.

       Y = TTM(X,[A,B,C,...],DIMS) computes the sequence tensor-matrix
       products along the dimensions specified by DIMS.

       Y = TTM(...,'t') performs the same computations as above except
       the matrices are transposed.

       Examples
       X = ktensor([rand(5,2),rand(3,2),rand(4,2),rand(2,2)]);
       A = rand(4,5); B = rand(4,3); C = rand(3,4); D = rand(3,2);
       Y = ktensor_matrix_mult(X, A, 1)         #<-- computes X times A in mode-1
       Y = ktensor_matrix_mult(X, [A,B,C,D], 1) #<-- same as above
       Y = ktensor_matrix_mult(X, A', 1, 't')   #<-- same as above
       Y = ktensor_matrix_mult(X, [A,B,C,D], [1 2 3 4]) #<-- 4-way multiply
       Y = ktensor_matrix_mult(X, [D,C,B,A], [4 3 2 1]) #<-- same as above
       Y = ktensor_matrix_mult(X, [A,B,C,D])            #<-- same as above
       Y = ktensor_matrix_mult(X, [A',B',C',D'], 't')   #<-- same as above
       Y = ktensor_matrix_mult(X, [C,D], [3 4])     #<-- X times C in mode-3 & D in mode-4
       Y = ktensor_matrix_mult(X, [A,B,C,D], [3 4]) #<-- same as above
       Y = ktensor_matrix_mult(X, [A,B,D], [1 2 4])   #<-- 3-way multiply
       Y = ktensor_matrix_mult(X, [A,B,C,D], [1 2 4]) #<-- same as above
       Y = ktensor_matrix_mult(X, [A,B,D], -3)        #<-- same as above
       Y = ktensor_matrix_mult(X, [A,B,C,D], -3)      #<-- same as above

    """"

    ######################
    ### ERROR CHECKING ###
    ######################

    if not isinstance(X, KruskalTensor):
        X = KruskalTensor(X, [])

    # Check for transpose option
    isTranspose = False
    if len(args) > 0:
      if isnumeric(args[0]):
        dims = args[0]
      isTranspose =  args[-1]

    # Check for dims argument
    if not('dims' in locals()):
        dims = np.array([])

    # Check that 2nd argument is list array. If not, recall with V as a
    # cell array with one element.
    if not iscell(V):
        X = ktensor_matrix_mult(X,[V],dims,args[-1])
        return X

    # Get sorted dims and index for multiplicands
    dims,vidx = dimscheck(dims, X.ndim, numel(V))

    # Determine correct size index
    if isTranspose:
      j = 0
    else:
      j = 1

    # Check that each multiplicand is the right size.
    for i in numel(dims):
        if (V.ndim!=2) or (size(V[vidx[i]],j) != size(X,dims[i])):
            error(f'Inconsistent multiplicand shapes {size(X)}->{size(V[vidx[i]])}.')

    # Do the multiplications in the specified modes.
    for i in range(numel(dims)):
      if isTranspose:
        X.U[dims[i]] = V[vidx[i]].T @ X.U[dims[i]]
      else
        X.U[dims[i]] = V[vidx[i]] @ X.U[dims[i]]
