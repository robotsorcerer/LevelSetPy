from utils import *

def truncateGrid(gOld, dataOld=None, xmin=None, xmax=None, process=True):
    """
     [gNew, dataNew] = truncateGrid(gOld, dataOld, xmin, xmax)
        Truncates dataOld to be within the bounds xmin and xmax

     Inputs:
       gOld, gNew - old and new grid structures
       dataOld    - data corresponding to old grid structure
       process    - specifies whether to call processGrid to generate
                    grid points

     Output: dataNew    - equivalent data corresponding to new grid structure

     Mo Chen, 2015-08-27
     Lekan Molux, Aug 21, 2021

     Gather indices of new grid vectors that are within the bounds of the old
     grid
    """
    gNew.dim = gOld.dim
    gNew.vs = cell(gNew.dim, 1)
    gNew.N = zeros(gNew.dim, 1)
    gNew.min = zeros(gNew.dim, 1)
    gNew.max = zeros(gNew.dim, 1)
    gNew.bdry = gOld.bdry
    small = 1e-3

    for i in range(gNew.dim):
        gNew.vs[i] = gOld.vs[i](np.logical_and(gOld.vs[i] > xmin[i], gOld.vs[i] < xmax[i]));
        gNew.N[i]= len(gNew.vs[i])
        gNew.min[i] = min(gNew.vs[i])
    if gNew.N[i] == 1:
        gNew.max[i] = max(gNew.vs[i]) + small
    else:
        gNew.max[i] = max(gNew.vs[i])
    if gNew.N[i] < gOld.N[i]:
        gNew.bdry[i] = addGhostExtrapolate

    if process:
        gNew = processGrid(gNew)

    if not dataOut:
        return gNew

    dataNew = []
    # Truncate everything that's outside of xmin and xmax
    if gOld.dim==1:
        # Data
        if dataOld:
          dataNew = dataOld[np.logical_and(gOld.vs[0]>xmin, gOld.vs[0]<xmax)]

    elif gOld.dim==2:
        if dataOld:
            dataNew = dataOld[np.logical_and(gOld.vs[0]>xmin[0], gOld.vs[0]<xmax[0]), \
                              np.logical_and(gOld.vs[1]>xmin[1], gOld.vs[1]<xmax[1])]

    elif gOld.dim==3
        if dataOld:
            dataNew = dataOld[np.logical_and(gOld.vs[0]>xmin[0], gOld.vs[0]<xmax[0]), \
                              np.logical_and(gOld.vs[1]>xmin[1], gOld.vs[1]<xmax[1]), \
                              np.logical_and(gOld.vs[2]>xmin[2], gOld.vs[2]<xmax[2])]

    elif gOld.dim==4
        if dataOld:
            dataNew = dataOld[np.logical_and(gOld.vs[0]>xmin[0], gOld.vs[0]<xmax[0]), \
                              np.logical_and(gOld.vs[1]>xmin[1], gOld.vs[1]<xmax[1]), \
                              np.logical_and(gOld.vs[2]>xmin[2], gOld.vs[2]<xmax[2]), \
                              np.logical_and(gOld.vs[3]>xmin[3], gOld.vs[3]<xmax[3])]
    else:
        error('truncateGrid has only been implemented up to 4 dimensions!')

    return gNew, dataNew
