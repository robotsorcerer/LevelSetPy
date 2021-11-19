__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Decomposing Level Sets of PDEs"
__credits__  	= "Sylvia Herbert, Ian Abraham"
__license__ 	= "Lekan License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Fix Tensor Mode Swap in Memory Layout."

import numpy as np
from Utilities import *

class TenMat():
    def __init__(self, T, **options):
        """
            This class provides the boilerpate for matricizing a Tensor.

            # TODO: Why does Numpy flip 0-Mode with 1-Mode?

            Parameters
            ----------
            T:       A Tensor < see class_tensor.py />.
            options: A bundle class. If it is a dictionary, it is converted to a bundle.
                     It contains the following fields:
                rdims: A numpy/cupy (dtype=intp) index array which specifies the modes of T to
                       which we map the rows of a matrix, and the remaining
                       dimensions (in ascending order) map to the columns.
                cdims:  A numpy/cupy (dtype=intp) index array which specifies the modes of T to
                       which we map the   columns of a matrix, and the
                       remaining dimensions (in ascending order) map
                       to the rows.
                cyclic: String which specifies the dimension in rdim which
                        maps to the rows of the matrix, and the remaining
                        dimensions span the columns in an order specified
                        by the string argument "cyclic" as follows:

                      'fc' - Forward cyclic.  Order the remaining dimensions in the
                           columns by [rdim+1:T.ndim, 1:rdim-1].  This is the
                           ordering defined by Kiers.

                       'bc' - Backward cyclic.  Order the remaining dimensions in the
                           columns by [rdim-1:-1:1, T.ndim:-1:rdim+1].
                           This is the ordering defined by De Lathauwer, De Moor, and Vandewalle.

            Calling Signatures
            ------------------
            TenMat(T, options.rdims): Create a matrix representation of a tensor
                T.  The dimensions (or modes) specified in rdims map to the rows
                of the matrix, and the remaining dimensions (in ascending order)
                map to the columns.

            TenMat(T, cdims, Transpose=True): Similar to rdims, but for column
                dimensions are specified, and the remaining dimensions (in
                ascending order) map to the rows.

            TenMat(T, rdims, cdims): Create a matrix representation of
               tensor T.  The dimensions specified in RDIMS map to the rows of
               the matrix, and the dimensions specified in CDIMS map to the
               columns, in the order given.

            TenMat(T, rdim, cyclic): Create the same matrix representation as
               above, except only one dimension in rdim maps to the rows of the
               matrix, and the remaining dimensions span the columns in an order
               specified by the string argument STR as follows:

              'fc' - Forward cyclic.  Order the remaining dimensions in the
                           columns by [rdim+1:T.ndim, 1:rdim-1].  This is the
                           ordering defined by Kiers.

               'bc' - Backward cyclic.  Order the remaining dimensions in the
                           columns by [rdim-1:-1:1, T.ndim:-1:rdim+1].  This is the
                           ordering defined by De Lathauwer, De Moor, and Vandewalle.

            TenMat(T, options=Bundle({rdims, cdims, tsize})): Create a tenmat from a matrix
                   T along with the mappings of the row (rdims) and column indices
                   (cdims) and the size of the original tensor (T.shape).

            Author: Lekan Molux, November 3, 2021
        """

        assert isinstance(T, Tensor), 'T must be a tensor class.'
        assert T.data.ndim !=2, "Inp.t Tensor must be 2D."
        assert isinstance(T, Tensor), "T must be of class tensor type."

        if not isbundle(options) and isinstance(options, dict):
            options = Bundle(options)
        assert isbundle(options), "options must be of Bundle class."

        self.tsize = np.asarray(options.__dict__.get("tsize", T.shape), dtype=np.intp)
        self.rindices = options.__dict__.get("rdims", None)
        self.cindices = options.__dict__.get("cdims", None)
        self.data = T.data
        self.T= T

        tsize = np.asarray(options.__dict__.get("tsize", T.shape), dtype=np.intp)
        rdims = options.__dict__.get("rdims", None)
        cdims = options.__dict__.get("cdims", None)
        data  = T.data

        n = numel(tsize)

        if np.any(rdims) and np.any(cdims):
            dims_joined = np.concatenate((rdims, cdims))
        elif np.any(rdims) and not np.any(cdims):
            dims_joined = rdims
        elif not np.any(rdims) and np.any(cdims):
            dims_joined = cdims

        if not np.allclose(range(n), np.sort(dims_joined)):
            raise ValueError('Incorrect dimension specifications.')
        elif (np.prod(self.tsize[rdims]) != size(self.data, 0)):
            raise ValueError('T.shape[0] does not match size specified by rdims and shape.')
        elif (np.prod(self.tsize[cdims]) != size(self.data, 1)):
            raise ValueError('T.shape[1] does not match size specified by cdims and shape.')

        tsize = T.shape
        n     = T.ndim

        tmp = np.zeros((n), dtype=bool)
        tmp.fill(True)
        if np.any(rdims):
            tmp[np.ix_(*rdims)] = False
            cdims = np.nonzero(tmp)

        if isfield(options, 'cyclic') and options.cyclic=='T':
            cdims = copy.copy(options.rdims)
            tmp = np.zeros((n), dtype=bool)
            tmp.fill(True)
            tmp[np.ix_(*cdims)] = False
            rdims = np.nonzero(tmp)

        elif isfield(options, 'cyclic') and options.cyclic=='fc':
            rdims = options.rdims

            if numel(rdims)!=1:
                raise ValueError('Only one row dimension if third argument is ''fc''.')
            cdims = np.concatenate((np.arange(rdims, n, dtype=np.intp), \
                                    np.arange(rdims-1, dtype=np.intp)), dtype=np.intp)

        elif isfield(options, 'cyclic') and options.cyclic=='bc':
            rdims = options.rdims

            if numel(rdims)!=1:
                raise ValueError('Only one row dimension if third argument is ''bc''.')

            cdims = np.concatenate((np.arange(rdims, -1, 1, dtype=np.intp),\
                                    np.arange(n-1, -1, rdims+1, dtype=np.intp)), dtype=np.intp)
        else:
            raise ValueError('Unrecognized option.')

        rdims = options.rdims
        cdims = options.cdims
