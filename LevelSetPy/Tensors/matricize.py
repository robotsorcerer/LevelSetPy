__all__ = ["matricization"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Tensor Algebra"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Finished"

import numpy as np
from Utilities import *
from .class_tensor import Tensor
from TenMat import matricize_tensor

def matricization(T, mode=1): # pragma nocover
    np.warnings.warn("matricization() has been matricize_tensor()",
                   DeprecationWarning, stacklevel=1)
    # return matricize_tensor
    # matricization.__doc__ = "\nDEPRECATED! use matricize_tensor(options).\n\n" \
    #                     + matricize_tensor.__doc__
    # # Deprecated
    """
        Matricize the tensor T shaped (3,3) into Folds along axes

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
    assert T.ndim == 3, "We do not support higher order tensors >3 at this moment"

    if isinstance(T, Tensor):
        T = T.data
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

    return Tensor(X)
