
class TenMat():
    def __init__(self, T, options):
        """
            This class matricizes a Tensor.

            Signatures
            ----------
            TenMat(T, RDIMS): Create a matrix representation of a tensor
                T.  The dimensions (or modes) specified in RDIMS map to the rows
                of the matrix, and the remaining dimensions (in ascending order)
                map to the columns.

            TENMAT(T, CDIMS, Transpose=True): Similar to RDIMS, but for column
                dimensions are specified, and the remaining dimensions (in
                ascending order) map to the rows.

            TENMAT(T, RDIMS, CDIMS): Create a matrix representation of
               tensor T.  The dimensions specified in RDIMS map to the rows of
               the matrix, and the dimensions specified in CDIMS map to the
               columns, in the order given.

            TENMAT(T, RDIM, STR): Create the same matrix representation as
               above, except only one dimension in RDIM maps to the rows of the
               matrix, and the remaining dimensions span the columns in an order
               specified by the string argument STR as follows:

              'fc' - Forward cyclic.  Order the remaining dimensions in the
                           columns by [RDIM+1:T.ndim, 1:RDIM-1].  This is the
                           ordering defined by Kiers.

               'bc' - Backward cyclic.  Order the remaining dimensions in the
                           columns by [RDIM-1:-1:1, T.ndim:-1:RDIM+1].  This is the
                           ordering defined by De Lathauwer, De Moor, and Vandewalle.

            TENMAT(T, RDIMS, CDIMS, TSIZE): Create a tenmat from a matrix
                   T along with the mappings of the row (RDIMS) and column indices
                   (CDIMS) and the size of the original tensor (TSIZE).

            Author: Lekan Molux, November 3, 2021
        """

        self.tsize = options.__dict__.get(tsize, [])
        self.rindices = options.__dict__.get(rindices, [])
        self.cindices = options.__dict__.get(cindices, [])
        self.data = options.__dict__.get(data, [])

        assert T.ndims !=2, "Input Tensor must be 2D."
