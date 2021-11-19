__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

import numpy as np
from Utilities import *
from ValueFuncs import *
from Grids import processGrid
from BoundaryCondition import addGhostPeriodic
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
import logging

logger = logging.getLogger(__name__)

def proj(g, data, dimsToRemove, xs=None, NOut=None, process=True):
    """
        [gOut, dataOut] = proj(g, data, dims, xs, NOut)
        Projects data corresponding to the grid g in g.dim dimensions, removing
        dimensions specified in dims. If a point is specified, a slice of the
        full-dimensional data at the point xs is taken.
        Inp.ts:
        g            - grid corresponding to inp.t data
        data         - inp.t data
        dimsToRemove - vector of len g.dim specifying dimensions to project
                    For example, if g.dim = 4, then dims = [0 0 1 1] would
                    project the last two dimensions
                    xs           - Type of projection (defaults to 'min')
                    'min':    takes the union across the projected dimensions
                    'max':    takes the intersection across the projected dimensions
                    a vector: takes a slice of the data at the point xs
                    NOut    - number of grid points in output grid (defaults to the same
                    number of grid points of the original grid in the unp.ojected
                    dimensions)
                    process            - specifies whether to call processGrid to generate
                            grid points
                            Outputs:
                            gOut    - grid corresponding to projected data
                            dataOut - projected data
    """
    # print(f'dimsToRemove: {dimsToRemove} {dimsToRemove.shape}')
    # Inp.t checking
    if isinstance(dimsToRemove, list):
        dimsToRemove = np.asarray(dimsToRemove)
    if len(dimsToRemove) != g.dim:
        raise ValueError('Dimensions are inconsistent!')

    if np.count_nonzero(np.logical_not(dimsToRemove)) == g.dim:
        logger.warning('Inp.t and output dimensions are the same!')
        return g, data

    # By default, do a projection
    if not xs:
        xs = 'min'

    # If a slice is requested, make sure the specified point has the correct
    # dimension
    if isnumeric(xs) and len(xs) != np.count_nonzero(dimsToRemove):
        raise ValueError('Dimension of xs and dims do not match!')

    if NOut is None:
        NOut = g.N[np.ix_(np.logical_not(dimsToRemove))]
        # if NOut.ndim < 2:
        #     NOut = expand(NOut, 1)

    dataDims = data.ndim
    if np.any(data) and not (dataDims== g.dim or dataDims== g.dim+1) \
        and not isinstance(data, list):
        raise ValueError('Inconsistent inp.t data dimensions!')

    # Project data
    if dataDims == g.dim:
        gOut, dataOut = projSingle(g, data, dimsToRemove, xs, NOut, process)
    else:
        gOut, _  = projSingle(g, np.array([]), dimsToRemove, xs, NOut, process)

    # Project data
    numTimeSteps = len(data) if iscell(data) else data.shape[dataDims-1]
    dataOut      = cell(numTimeSteps) #np.zeros( NOut.T.shape + (numTimeSteps,) )

    for i in range(numTimeSteps):
        if iscell(data):
            _, dataOut[i] = projSingle(g, data[i], dimsToRemove, xs, NOut, process)
        else:
            _, dataOut[i] = projSingle(g, data[i, ...], dimsToRemove, xs, NOut, process)

    dataOut = np.asarray(dataOut).T

    return gOut, dataOut



def projSingle(g, data, dims, xs, NOut, process):
    """
     [gOut, dataOut] = proj(g, data, dims, xs, NOut)
       Projects data corresponding to the grid g in g.dim dimensions, removing
       dimensions specified in dims. If a point is specified, a slice of the
       full-dimensional data at the point xs is taken.

     Inp.ts:
       g       - grid corresponding to inp.t data
       data    - inp.t data
       dims    - vector of len g.dim specifying dimensions to project
                     For example, if g.dim = 4, then dims = [0 0 1 1] would
                     project the last two dimensions
       xs      - Type of projection (defaults to 'min')
           'min':    takes the union across the projected dimensions
           'max':    takes the intersection across the projected dimensions
           a vector: takes a slice of the data at the point xs
       NOut    - number of grid points in output grid (defaults to the same
                 number of grid points of the original grid in the unp.ojected
                 dimensions)
       process            - specifies whether to call processGrid to generate
                            grid points

     Outputs:
       gOut    - grid corresponding to projected data
       dataOut - projected data

     Original by Sylvia;
     Python by Lekan July 29. 2021
    """

    # Create ouptut grid by keeping dimensions that we are not collapsing
    if not np.any(g):
        if not xs.isalpha() or not strcmp(xs, 'max') and not strcmp(xs, 'min'):
            raise ValueError('Must perform min or max projection when not specifying grid!')
    else:
        dims = dims.astype(bool)
        dim = np.count_nonzero(np.logical_not(dims))

        # take the min/max/bdry along the axes we are keeping
        # and create new grid
        gOut = Bundle(dict(dim = dim,
                min = g.min[np.ix_(np.logical_not(dims))],
                max = g.max[np.ix_(np.logical_not(dims))],
                ))
        bdry = expand(np.asarray(g.bdry)[np.logical_not(dims)], 1)
        gOut.bdry = bdry
        if numel(NOut) == 1:
            gOut.N = NOut*ones(gOut.dim, 1).astype(np.int64)
        else:
            gOut.N = NOut


        # Process the grid to populate the remaining fields if necessary
        if process:
            gOut = processGrid(gOut)

        # Only compute the grid if value function is not requested
        if not np.any(data) or not data.size:
            return gOut, None

    # 'min' or 'max'
    if isinstance(xs, str): #xs.isalpha():
        dimsToProj = np.nonzero(dims)[0]
        # dimsToProj = dimsToProj[0] if isinstance(dimsToProj, np.ndarray) else dimsToProj
        dimsToProjList = list(range(dimsToProj.size))[::-1]

        for i in dimsToProjList: #range(dimsToProj.size):
            if strcmp(xs,'min'):
                dataOut = np.amin(data, axis=dimsToProjList[i])
            elif strcmp(xs, 'max'):
                dataOut = np.amax(data, axis=dimsToProjList[i])
            else:
                error('xs must be a vector, ''min'', or ''max''!')

        return gOut, dataOut

    # Take a slice
    g, data = augmentPeriodicData(g, data)

    eval_pt = cell(g.dim)
    xsi = 0
    for i in range(g.dim):
        if dims[i]:
            # If this dimension is periodic, wrap the inp.t point to the correct period
            if isfield(g, 'bdry') and id(g.bdry[i])==id(addGhostPeriodic):
                period = max(g.vs[i].squeeze()) - min(g.vs[i].squeeze())
                while xs[xsi] > max(g.vs[i]):
                    xs[xsi] -= period
                while xs[xsi] < min(g.vs[i]):
                    xs[xsi] += period
            eval_pt[i] = xs[xsi]
            xsi += 1
        else:
            eval_pt[i] = g.vs[i]

    # https://stackoverflow.com/questions/21836067/interpolate-3d-volume-with-numpy-and-or-scipy
    # data_coords = [x.squeeze() for x in g.vs]
    eval_pt = [x.squeeze() for x in eval_pt if isinstance(x, np.ndarray)] + [np.array([x]) for x in eval_pt if not isinstance(x, np.ndarray)]
    fn = [RegularGridInterpolator(eval_pt[:-1], data[-1,...])]
    # eshape = tuple([x.shape for x in eval_pt])
    # if len(eshape) != data.ndim:
    #     eval_pt = np.tile(eval_pt, [*(data.shape[:-1]), 1])
    temp = fn(eval_pt)
    print(f'temp: {temp.shape}')
    # dataOut = copy.copy(temp)
    #
    # temp = g.vs[np.logical_not(dims)]
    #
    # # dataOut = np.interp(temp, dataOut, gOut.xs[:])
    # dataOut = np.interp(temp, dataOut, gOut.xs.flatten())
    # print(f'dataOut.shape {dataOut.shape}')
    # if len(data.shape) == 2:
    #     f = interpolate.interp2d(g.vs[0], g.vs[1], data, kind='linear')
    #     dataOut = np.concatenate([f(*x) for x in eval_pt])
    # elif len(data.shape) == 3:
    #     fs = [interpolate.interp2d(g.vs[0], g.vs[1], data[:,:,i], kind='linear') for i in range(data.shape[2])]
    #     dataOut= np.stack([np.concatenate([f(*x) for x in eval_pt]) for f in fs])
    # else:
    #     raise ValueError('%dD data not supported' % len(data.shape))

    return gOut, dataOut
