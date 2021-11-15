__all__ = ['upwindFirstENO2']

import copy
import logging
import numpy as np
from Utilities import *
logger = logging.getLogger(__name__)


from .check_eq_approx import checkEquivalentApprox

def upwindFirstENO2(grid, data, dim, generateAll=0):
    """
     upwindFirstENO2: second order upwind approx of first derivative.

       [ derivL, derivR ] = upwindFirstENO2(grid, data, dim, generateAll)

     Computes a second order directional approximation to the first
       derivative, using a oscillation reducing minimum modulus choice
       of second order term.  The result is an order 2 ENO scheme.

     The approximation is constructed by a divided difference table.

     Some details of this scheme can be found in O&F, section 3.3,
       where this scheme is equivalent to including the Q_1 and Q_2
       terms of the ENO approximation.

     The generateAll option is used for debugging, and possibly by
       higher order weighting schemes.  Under normal circumstances
       the default (generateAll = false) should be used.

     parameters:
       grid	Grid structure (see processGrid.m for details).
       data        Data array.
       dim         Which dimension to compute derivative on.
       generateAll Return all possible second order upwind approximations.
                     If this boolean is true, then derivL and derivR will
                     be cell vectors containing all the approximations
                     instead of just the minimum modulus approximation.
                     (optional, default = 0)

       derivL      Left approximation of first derivative (same size as data).
       derivR      Right approximation of first derivative (same size as data).

     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 1/22/04
     Lekan on August 16, 2021
    ---------------------------------------------------------------------------
    """
    if((dim < 0) or (dim > grid.dim)):
        error('Illegal dim parameter')

    dxInv = 1 / grid.dx.item(dim)

    # How big is the stencil?
    stencil = 2

    # Check that approximations that should be equivalent are equivalent
    #   (for debugging purposes, only used if generateAll == 1).
    checkEquivalentApproximations = 1
    small = 100 * eps             # a small number for "equivalence"

    # Add ghost cells.
    gdata = grid.bdry[dim](data, dim, stencil, grid.bdryData[dim])
    #print(f'gdata: {np.linalg.norm(gdata)}, {gdata.shape}')
    #import time; time.sleep(10)
    #---------------------------------------------------------------------------
    # Create cell array with array indices.
    sizeData = size(gdata)
    indices1 = []
    for i in range(grid.dim):
      indices1.append(np.arange(sizeData[i], dtype=np.intp))
    indices2 = copy.copy(indices1)

    #---------------------------------------------------------------------------
    # First divided differences (first entry corresponds to D^1_{-1/2}).
    indices1[dim] = np.arange(1,size(gdata, dim), dtype=np.intp)
    indices2[dim] = copy.copy(indices1[dim]) - 1
    D1 = dxInv*(gdata[np.ix_(*indices1)] - gdata[np.ix_(*indices2)])
    # print(f'[@upFirstENO2] gdata: {gdata.shape} {dxInv}')
    # print(f'gdata[np.ix_(*indices1)] norm: {np.linalg.norm(gdata[np.ix_(*indices1)])}')
    # print(f'gdata[np.ix_(*indices2)] norm: {np.linalg.norm(gdata[np.ix_(*indices2)])}')
    # print(f'D1: {D1.shape} norm: {np.linalg.norm(D1)}')
    # print(f'indices1 in upENO2: {indices1[dim]}, shape: {indices1[dim].shape}')
    # print(f'indices2 in upENO2: {indices2[dim]}, shape: {indices2[dim].shape}')

    # print(f'@UpwindFirstENO2: D1: {np.linalg.norm(D1)}, {D1.shape}, \
    #         dL: {[(np.linalg.norm(dL[i])) for i in range(2)]}, \
    #         dR: {[(np.linalg.norm(dR[i])) for i in range(2)]}')
    # Second divided differences (first entry corresponds to D^2_0).
    indices1[dim] = np.arange(1, size(D1, dim), dtype=np.intp)
    indices2[dim] = copy.copy(indices1[dim]) - 1
    D2 = 0.5 * dxInv*(D1[np.ix_(*indices1)] - D1[np.ix_(*indices2)])

    #---------------------------------------------------------------------------
    # First divided difference array has an extra entry at top and bottom
    #   (from stencil width 2), so strip them off.
    # Now first entry corresponds to D^1_{1/2}.
    indices1[dim] = np.arange(1, size(D1, dim)-1, dtype=np.intp)
    D1 = D1[np.ix_(*indices1)]
    # print(f'indices1[dim]: {indices1[dim]}')
    # import time; time.sleep(10)
    #---------------------------------------------------------------------------
    # First order approx is just the first order divided differences.
    #   Make two copies to build the two approximations

    # Take leftmost grid.N(dim) entries for left approximation.
    indices1[dim] = np.arange(0, size(D1, dim) - 1, dtype=np.intp)
    dL = [D1[np.ix_(*indices1)] for i in range(2)]

    # Take rightmost grid.N(dim) entries for right approximation.
    indices1[dim] = np.arange(1, size(D1, dim), dtype=np.intp)
    # print('indices1[dim]: ', indices1[dim], [indices1[x] for x in [1, 2]])
    dR = [D1[np.ix_(*indices1)] for i in range(2)]

    #---------------------------------------------------------------------------
    # Each copy gets modified by one of the second order terms.
    #   Second order terms are sorted left to right.
    indices1[dim] = np.arange(0, size(D2, dim) - 2, dtype=np.intp)
    indices2[dim] = np.arange(1, size(D2, dim) - 1, dtype=np.intp)
    dL[0] += (grid.dx.item(dim) * D2[np.ix_(*indices1)])
    dL[1] += (grid.dx.item(dim) * D2[np.ix_(*indices2)])

    indices1[dim] += 1
    indices2[dim] += 1

    dR[0] -= (grid.dx.item(dim) * D2[np.ix_(*indices1)])
    dR[1] -= (grid.dx.item(dim) * D2[np.ix_(*indices2)])

    #---------------------------------------------------------------------------
    if(generateAll):
        if(checkEquivalentApproximations):
            # Rightward left and leftward right approximations should be the same
            #   (should be centered approximations, but we don't check for that).
            checkEquivalentApprox(dL[1], dR[0], small)

        # Caller requested both approximations in each direction.
        derivL = copy.copy(dL)
        derivR = copy.copy(dR)
    #---------------------------------------------------------------------------
    else:

        # Need to figure out which approximation has the least oscillation.
        #   Note that L and R in this section refer to neighboring divided
        #   difference entries, not to left and right approximations.

        # Pick out minimum modulus neighboring D2 term.
        D2abs = np.abs(D2)
        indices1[dim] = np.arange(0, size(D2, dim) - 1, dtype=np.intp)
        indices2[dim] = copy.copy(indices1[dim]) + 1
        smallerL = (D2abs[np.ix_(*indices1)] < D2abs[np.ix_(*indices2)])
        smallerR = np.logical_not(smallerL)

        #---------------------------------------------------------------------------
        # Pick out second order approximation that used the minimum modulus D2 term.
        indices1[dim] = np.arange(0, size(smallerL, dim) - 1, dtype=np.intp)
        derivL = dL[0] * smallerL[np.ix_(*indices1)] + dL[1] * smallerR[np.ix_(*indices1)]

        indices1[dim] = np.arange(1,size(smallerL, dim), dtype=np.intp)
        derivR = dR[0] * smallerL[np.ix_(*indices1)] + dR[1] * smallerR[np.ix_(*indices1)]

    return derivL, derivR
