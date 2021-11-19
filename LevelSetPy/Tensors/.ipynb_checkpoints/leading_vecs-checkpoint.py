__all__ = ['nvecs']

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Tensor Algebra"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

import numpy as np
import numpy as np
from Utilities import Bundle, isfield
from .tensor_utils import use_gpu
from Tensors.TenMat import matricize_tensor

def nvecs(X,n,r,options=None):
    """
        This function computes the leading mode-n vectors for a tensor.

        Parameters
        ----------
            Xn: The mode-n matricization of X.
            n:  The n-th mode of the tensor X.
            r:  Leading eigenvalue of Xn*Xn.T. This reveals information about the
                mode-n fibers.
            options: A Bundle class options hat packs computation options to be 
                     passed along to the decomposition solver.
                     
                     Parameters
                     ----------
                    .flipsign: make each column's largest element positive: Default: True
                    .svd: use svds on Xn rather than np.linalg.eigs on Xn*Xn'; Default: False

        Returns
        -------
            U: The left unitary (typically) orthogonal  mode-n matrix of X.

        Remarks
        -------
            In two-dimensions, the r leading mode-1 vectors are the same as the r left
            singular vectors and the r leading mode-2 vectors are the same as the r
            right singular vectors. By default, this method computes the top r
            eigenvectors of the matrix Xn*Xn^T. This behavior can be changed per the
            options argument as follows:


        Example:
        --------
           X = Tensor(np..randn(3,2,3))
           nvecs(X,3,2)

        Author: Lekan Molux
        Date: November 2, 2021
    """
    global use_gpu

    options = Bundle({}) if options is None  else options
    options = Bundle(options) if isinstance(options, dict) else options

    options.svd      = options.__dict__.get('svd', False)
    options.flipsign = options.__dict__.get('flipsign', True)
    options.rdims    = options.__dict__.get('rdims', np.arange(n, dtype=np.intp))
    use_gpu          = options.__dict__.get('use_gpu', use_gpu)
    
    print('X: ', X.shape, 'options.rdims: ', options.rdims)
    Xn = matricize_tensor(X, n)
    
    Xn = Xn().T.data
    
    print('Xmat: ', Xn.shape, 'r: ', r)
    if isfield(options, 'svd') and options.svd:
        if use_gpu:
            Xn = np.asarray(Xn) # Do it on gpu if available
            U,_, _ = np.linalg.svd(Xn, full_matrices=False)
        else:
            U,_, _ = np.linalg.svd(Xn, full_matrices=False)
        # we are only interested in the first r values, so copy those
        #U = U[:,:r]
    else:
        Y = Xn @ Xn.T
        if use_gpu:
            Y = np.asarray(Y) # Do it on gpu if available
            try:
                U = np.linalg.eigvalsh(Y, 'U')
            except:
                LinAlgError("Could not find the leading eigen values using"
                            "Numpy's eigvals method.")
            # move array back to host
            U = U.get()
        else:
            try:
                U = np.linalg.eigvalsh(Y, 'U')
            except:
                LinAlgError("Could not find the leading eigen values using"
                            "Numpy's eigvals method.")
        print('U: ', U.shape)
        # return only the leading 'r' values:
        U = U[:,:r]

    if isfield(options, 'flipsign') and options.flipsign:
        # Make the largest magnitude element be positive
        maxi = np.amax(np.abs(U))
        maxi_idx = np.where(np.abs(U)==maxi)

        for i in range(r):
            if U[maxi_idx] < 0:
                U[maxi_idx] *= -1

    return U