__all__ = ["matricize_tensor"]

import numpy as np
import numpy as np
from Utilities import *

from Tensors import Tensor

def matricize_tensor(options):
    """
    This function provides the boilerpate for matricizing a Tensor.

     Options
    ---------
    options: A list of options with sorted argumentds (in ascending order).
             Note that the ordering of the options argument must be progressive.
         .T: A Tensor < see class_tensor.py /> class with attributes
             .tsize: shape of the tensor as a list.
             .rindices: row indices to map to a matrix (see rdims below).
             .cindices: column indices to map to a matrix (see cdims below).

         .rdims: A numpy/cupy (dtype=np.np.intp) index array which specifies the modes of T to
               which we map the rows of a matrix, and the remaining
               dimensions (in ascending order) map to the columns.

         .cdims:  A numpy/cupy (dtype=np.np.intp) index array which specifies the modes of T to
               which we map the columns of a matrix, and the
               remaining dimensions (in ascending order) map
               to the rows.

         .cyclic: String which specifies the dimension in rdim which
                maps to the rows of the matrix, and the remaining
                dimensions span the columns in an order specified
                by the string argument "cyclic" as follows:

              'T'  - Transpose the matrix

              'fc' - Forward cyclic.  Order the remaining dimensions in the
                   columns by [rdim:T.ndim, 0:rdim-1].  This is the
                   ordering defined by Kiers.

               'bc' - Backward cyclic.  Order the remaining dimensions in the
                   columns by [range(rdim-1, 0, -1), range(T.ndim, rdim,-1)].
                   This is the ordering defined by De Lathauwer, De Moor, and Vandewalle.

    Calling Signatures
    ------------------
    matricize_tensor(T, rdims): Create a matrix representation of a tensor
        T.  The dimensions (or modes) specified in rdims map to the rows
        of the matrix, and the remaining dimensions (in ascending order)
        map to the columns.

    matricize_tensor(T, cdims, 'T'): Similar to rdims, but for column
        dimensions are specified, and the remaining dimensions (in
        ascending order) map to the rows.

    matricize_tensor(T, rdims, cdims): Create a matrix representation of
       tensor T.  The dimensions specified in RDIMS map to the rows of
       the matrix, and the dimensions specified in CDIMS map to the
       columns, in the order given.

    matricize_tensor(T, rdim, STR): Create the same matrix representation as
       above, except only one dimension in rdim maps to the rows of the
       matrix, and the remaining dimensions span the columns in an order
       specified by the string argument STR as follows:
       'T' - Transpose.

      'fc' - Forward cyclic.  Order the remaining dimensions in the
                   columns by [rdim+1:T.ndim, 1:rdim-1].  This is the
                   ordering defined by Kiers.

       'bc' - Backward cyclic.  Order the remaining dimensions in the
                   columns by [rdim-1:-1:1, T.ndim:-1:rdim+1].  This is the
                   ordering defined by De Lathauwer, De Moor, and Vandewalle.

    matricize_tensor(T, options=Bundle({rdims, cdims, tsize})): Create a tenmat from a matrix
           T along with the mappings of the row (rdims) and column indices
           (cdims) and the size of the original tensor (T.shape).

    Example:
    1.  X  = np.arange(1, 28).reshape(3,3,3)
        options = Bundle(dict(T=X, rdims=np.array([2], dtype=np.intp),
                          cdims=[0, 1], tsize=X.shape))
        X_1 = matricize_tensor(options)

    2.  X  = np.arange(1, 28).reshape(3,3,3)
        options = dict(rdims=np.array([0, 1], dtype=np.intp))
        X_1 = matricize_tensor(options)

    Author: Lekan Molux, November 3, 2021
    """
    assert isinstance(options, list), "options argument must be a list of params."

    if len(options)==0:
        T = Bundle(dict(
            tsize = None,
            rindices = None,
            cindices = None,
            data = None,
        ))
        return T

    if len(options)==1:
        P = options[0]
        assert isinstance(T, Tensor), "Supplied field for T must be of class <class Tensor>."
        T.tsize = P.tsize
        T.rindices = P.rindices
        T.cindices = P.cindices
        T.data = P.data
        return T

    # Case I: Convert a matrix to a tensor format
    if len(options)==4:
        data = options[0]
        if not isinstance(data, np.ndarray) or isinstance(data, np.ndarray) or data.ndim!=2:
            raise ValueError("Tensor T must be a 2D Numpy/Cupy array.")
        rdims = options[1]
        cdims = options[2]
        tsize = options[3]

        # Error check
        n = numel(tsize)
        if not np.array_equal(np.arange(n), np.sort(np.stack([rdims, cdims]))):
            raise ValueError('Incorrect specification of dimensions')
        elif (np.prod(tsize[rdims]) != size(data,0)):
            raise ValueError(f'A\'s size along 0-dim, {A.shape[0]} does not match size specified by RDIMS and SIZE.')
        elif (np.prod(tsize[cdims]) != size(data,1)):
            raise ValueError(f'A\'s size along 1-dim, {A.shape[1]} does not match size specified by CDIMS and SIZE.')

        # save tensor variables
        T.tsize     = tsize
        T.rindices  = rdims
        T.cindices  = cdims
        T.data      = data

        return T

    # Case II: MDA format to tenmat//Do we need this for py?
    if isinstance(options[0], np.ndarray) or \
        isinstance(options[0], np.ndarray):
        options[0] = Tensor(options[0])
        #opt_arg = Bundle({x[0]: x[1] for x in options})
        T = matricize_tensor(options)
        return T

    if len(options)<2 or len(options)>3:
        raise ValueError("Incorrect number of arguments.")

    # save T size and num-dims
    T     = options[0]
    tsize = T.shape
    n     = T.ndim

    # Figure out which dimensions get mapped where
    if len(options)==2:
        rdims = options[1]
        tmp = np.ones((n, 1), dtype=bool)
        tmp[rdims] = False
        cdims = np.nonzero(tmp)[0]

    if len(options)==3:
        if isinstance(options[2], str):
            if strcmp(options[2], 'T'):
                cdims   = options[1]
                tmp     = np.ones((n,1), dtype=bool)
                tmp[cdims] = False
                rdims = np.nonzero(tmp)[0]
            elif strcmp(options[2], 'fc'):
                rdims = options[1]
                if numel(rdims)!=1:
                    raise ValueError(f'Only one row dimension if options.cyclic is ''fc''.')
                cdims = np.concatenate((np.arange(rdims, n, dtype=np.intp), \
                                        np.arange(rdims-1, dtype=np.intp)), dtype=np.intp)
            elif strcmp(options[2], 'bc'):
                rdims = options[1]
                if numel(rdims)!=1:
                    raise ValueError('Only one row dimension if third argument is ''bc''.')
                cdims = np.concatenate((np.arange(rdims-1, dtype=np.intp)[::-1],\
                                        np.arange(rdims, n, dtype=np.intp)[::-1]), dtype=np.intp)
            else:
                raise ValueError(f'Unrecognized cyclic option. Cyclic option can only be one of {T},'
                                  '{\'fc\'}, or {\'bc\'}')

        else:
            rdims = options[1]
            cdims = options[2]

    # Error check
    if not np.array_equal(np.arange(n), np.sort( np.hstack((rdims, cdims)))):
        raise ValueError('Incorrect specification of dimensions')

    print(rdims, 'cdims: ', cdims)
    # Permute T so that the dimensions specified by RDIMS come first
    T_Rot = np.transpose(T.data, axes=np.hstack([rdims, cdims]))
    tsize = np.asarray(tsize)

    rprods = np.prod(tsize[rdims])
    np.ods = np.prod(tsize[cdims])

    data     = T_Rot.reshape(rprods, np.ods)
    T.rindices = rdims
    T.cindices = cdims
    T.tsize    = tsize
    T = Tensor(T.data, shape=T.data.shape, rindices = rdims, cindices = cdims)

    return T
