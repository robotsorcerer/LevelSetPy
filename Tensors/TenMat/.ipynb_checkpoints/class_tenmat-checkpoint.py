__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Decomposing Level Sets of PDEs"
__credits__  	= "Sylvia Herbert, Ian Abraham"
__license__ 	= "Lekan License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Fix Tensor Mode Swap in Memory Layout."

import cupy as cp
from Tensors import Tensor
from Utilities import *

class TenMatClass():
    def __init__(self, T, **options):
    #def __init__(self, T, rdims=None, cdims=None, cyclic=None):
        """
        This class provides the boilerpate for matricizing a Tensor.

        Parameters
        ----------
        T:       A Tensor < see class_tensor.py />.
        options: A bundle class. If it is a dictionary, it is converted to a bundle.
                 It contains the following fields:
            rdims: A numpy/cupy (dtype=cp.cp.intp) index array which specifies the modes of T to 
                   which we map the rows of a matrix, and the remaining 
                   dimensions (in ascending order) map to the columns.
            cdims:  A numpy/cupy (dtype=cp.cp.intp) index array which specifies the modes of T to 
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
           'T' - Transpose.

          'fc' - Forward cyclic.  Order the remaining dimensions in the
                       columns by [rdim+1:T.ndim, 1:rdim-1].  This is the
                       ordering defined by Kiers.

           'bc' - Backward cyclic.  Order the remaining dimensions in the
                       columns by [rdim-1:-1:1, T.ndim:-1:rdim+1].  This is the
                       ordering defined by De Lathauwer, De Moor, and Vandewalle.

        TenMat(T, options=Bundle({rdims, cdims, tsize})): Create a tenmat from a matrix
               T along with the mappings of the row (rdims) and column indices
               (cdims) and the size of the original tensor (T.shape).
               
        Example: 
        1.  X  = cp.arange(1, 28).reshape(3,3,3)
            options = dict(rdims=cp.array([2], dtype=cp.intp))
            X_1 = TenMat(X, **options)
            
        2.  X  = cp.arange(1, 28).reshape(3,3,3)
            options = dict(rdims=cp.array([0, 1], dtype=cp.intp))
            X_1 = TenMat(X, **options)

        Author: Lekan Molux, November 3, 2021
        """

        if not isinstance(T, Tensor):
            T = Tensor(T, T.shape)
            
        if not isbundle(options) and isinstance(options, dict):
            options = Bundle(options)
        assert isbundle(options), "options must be of Bundle class."

        self.tsize = cp.asarray(options.__dict__.get("tsize", T.shape))
        self.rindices = options.__dict__.get("rdims", None)
        self.cindices = options.__dict__.get("cdims", None)
        self.data = T.data    

        if self.rindices is None and self.cindices is None:
            return

        tsize = cp.asarray(options.__dict__.get("tsize", T.shape))
        rdims = options.__dict__.get("rdims", None)
        cdims = options.__dict__.get("cdims", None)
        data  = T.data

        n = T.data.ndim
            
        #if len(options)==1:
        if isfield(options, 'rdims') and not isfield(options, 'cdims'):
            tmp = cp.zeros((n), dtype=bool)
            tmp.fill(True)
            tmp[rdims] = False
            cdims = cp.nonzero(tmp)[0]
        elif isfield(options, 'cyclic'):
        #elif len(options)>=2: #isfield(options, 'cyclic'):
            if options.cyclic=='T':
                cdims = options.rdims 
                tmp = cp.zeros((n,1), dtype=bool)
                tmp.fill(True)
                tmp[cdims] = False
                rdims = cp.nonzero(tmp)[0]
            elif options.cyclic=='fc':
                rdims = options.rdims
                if numel(rdims)!=1:
                    raise ValueError(f'Only one row dimension if options.cyclic is ''fc''.')
                cdims = cp.concatenate((cp.arange(rdims, n, dtype=cp.intp), \
                                        cp.arange(rdims-1, dtype=cp.intp)), dtype=cp.intp)
            elif options.cyclic=='bc':
                rdims = options.rdims

                if numel(rdims)!=1:
                    raise ValueError('Only one row dimension if third argument is ''bc''.')

                cdims = cp.concatenate((cp.arange(rdims-1, dtype=cp.intp)[::-1],\
                                        cp.arange(rdims, n, dtype=cp.intp)[::-1]), dtype=cp.intp)
            else:
                raise ValueError('Unrecognized option.')

        else:
            rdims = options.rdims
            cdims = options.cdims

        # Error check
        if not cp.array_equal(cp.arange(n), cp.sort( cp.concatenate((rdims, cdims)))):
            raise ValueError('Incorrect specification of dimensions')

        # Permute T so that the dimensions specified by RDIMS come first
        T_Rot = cp.transpose(T.data, axes=cp.concatenate([rdims, cdims]))
        rprods = cp.prod(tsize[rdims])
        cprods = cp.prod(tsize[cdims]) 
        
        self.data     = T_Rot.reshape(rprods, cprods)
        self.rindices = rdims
        self.cindices = cdims
        self.tsize    = tsize 
        self.T = Tensor(self.data, shape=self.data.shape)
        
    def __call__(self):
        return Bundle(dict(T=self.T, rdims=self.rindices, cdims=self.cindices))
        
        