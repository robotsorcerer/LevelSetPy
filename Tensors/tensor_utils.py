__all__ = ['dims_check', 'get_size', 'use_gpu']

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Tensor Algebra"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Finished"

import sys
import copy
import numpy as np
import numpy as np
from Utilities import isscalar, error

use_gpu = True if np.is_available else False

def dims_check(dims=None, N=None, M=None):
    """
        This preprocesses dimensions of a tensor

        Signature:
            newdims, _ = dimscheck(dims, N): Check that the specified dimensions are valid for a tensor

    """
    if dims is None:
        dims = np.arange(N)

    if isscalar(dims):
        dims = np.array([dims])

    if np.max(dims)<0:
        tf = np.isin(-dims, range(N)).astype(np.int64)
        tf = np.array([tf]) if isscalar(tf) else tf


        if  min(tf)==0:
            error("Invalid dimension specified.")
        dims = list(set(range(N)).difference(-dims))

    tf = np.isin(dims, range(N)).astype(np.int64)
    tf = np.array([tf]) if isscalar(tf) else tf

    if min(tf)==0:
        error("Invalid dimension specified.")

    P = len(dims)

    sorted_dims = np.sort(dims)
    sorted_dims_idx = np.argsort(dims)

    if M > N: raise ValueError("We cannot have more multiplicands than dimensions")


    if (M != N) and (M != P):
        raise ValueError("Invalid number of multiplicands")

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

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
