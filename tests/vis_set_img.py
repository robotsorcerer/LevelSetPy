__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

from Utilities import *
from Grids import *
from .show_2d import show2D
from .show_3d import show3D
from ValueFuncs import proj
import matplotlib.pyplot as plt
from Visualization.value_set_viz import ValueSetVisualizer


# DEPRECATIONS ================================================================

def visSetIm(data, g=None, color='r', level=0, extraArgs=None): # pragma nocover
    """
     h = visSetIm(g, data, color, level, sliceDim, applyLight)
     Code for quickly visualizing level sets

     Icp.ts: g          - grid structure
             data       - value function corresponding to grid g
             color      - (defaults to red)
             level      - level set to display (defaults to 0)
             sliceDim   - for 4D sets, choose the dimension of the slices (defaults
                          to last dimension)
     Output: h - figure handle

     Adapted from Ian Mitchell's visualizeLevelSet function from the level set
     toolbox

     Lekan Molux, August, 2021
    """
    cp.warnings.warn("visSetIm() has been ValueSetVisualizer()",
                   DeprecationWarning, stacklevel=1)
    return ValueSetVisualizer()
visSetIm.__doc__ = "\nDEPRECATED! use ValueSetVisualizer().\n\n" \
                        + ValueSetVisualizer.__doc__
#     ## Default parameters and icp.t check
#     if g is None:
#         N = cp.asarray(size(data)).T
#         g = createGrid(cp.ones(numDims(data), 1), N, N)
#
#     if g.dim != numDims(data) and g.dim+1 != numDims(data):
#         error('Grid dimension is inconsistent with data dimension!')
#
#     if not extraArgs:
#         extraArgs = Bundle({})
#
#     deleteLastPlot = True
#     if isfield(extraArgs, 'deleteLastPlot'):
#         deleteLastPlot = extraArgs.deleteLastPlot
#
#     save_png = False
#     if isfield(extraArgs, 'fig_filename'):
#         save_png = True
#         fig_filename = extraArgs.fig_filename
#     if g.dim == numDims(data):
#         # Visualize a single set
#         visSetIm_single(g, data, color, level, extraArgs)
#     else:
#       dataSize = size(data)
#       numSets = dataSize[-1]
#
#       fig = plt.figure(figsize=(16,9))
#       fig.tight_layout()
#       ax = self._fig.add_subplot(1, 1, 1)
#       extraArgs.ax = ax
#
#       for i in range(numSets):
#         if deleteLastPlot:
#             if i>1:  plt.clf()
#             visSetIm_single(g, data[i,...], color, level, extraArgs)
#         else:
#             if i == 1:
#                 visSetIm_single(g, data[i,...], color, level, extraArgs)
#
# ## Visualize a single set
# def visSetIm_single(g, data, color, level, extraArgs):
#
#     sliceDim = g.dim # Slice last dimension by default
#     applyLight = True # Add cam light by default
#     LineStyle = '-'
#     LineWidth = 1
#
#     if isfield(extraArgs, 'sliceDim'):
#         sliceDim = extraArgs.sliceDim
#
#     if isfield(extraArgs, 'applyLight'):
#         applyLight = extraArgs.applyLight
#
#     if isfield(extraArgs, 'LineStyle'):
#         LineStyle = extraArgs.LineStyle
#
#     if isfield(extraArgs, 'LineWidth'):
#         LineWidth = extraArgs.LineWidth
#
#     if isfield(extraArgs, 'savedict'):
#         savedict = extraArgs.savedict
#
#     if g.dim==1:
#         extraArgs.ax.plot(g.xs[0], data, linestyle='-', color=color)
#         extraArgs.ax.plot(g.xs[0], cp.zeros(size(g.xs[0])), linestyle=':', color='k')
#
#     elif g.dim==2:
#         show2D(g, data, fc='g', savedict = {"save": False}, disp=extraArgs.disp)
#
#     elif g.dim==3:
#         show3D(g, data, fc='g', savedict = {"save": False}, disp=extraArgs.disp)
#     elif g.dim==4:
#         visSetIm4D(g, data, color, level, sliceDim, applyLight)
#
# ## 3D Visualization
# def visSetIm3D( ax, g, data, color, level=0., disp=True):
#     show3D(g, data, fc=color, level=level, disp=disp)
#
# ## 4D Visualization
# def visSetIm4D(g, data, color, level, sliceDim, disp=True):
#     # Takes 6 slices in the dimension sliceDim and shows the 3D projections
#     N = 6
#     spC = 3
#     spR = 2
#     fig = plt.figure(figsize=(16, 9))
#     for i in range(N):
#         ax = fig.add_subplot(spR, spC, i, projection='3d')
#         xs = g.min[sliceDim] + i/(N+1) * (g.max[sliceDim] - g.min[sliceDim])
#
#         dim = cp.zeros(([4,1]))
#         dim[sliceDim, 0] = 1
#         g3D, data3D = proj(g, data, dim, xs)
#
#         # Visualize 3D slices
#         show3D(g3D, data3D, color, ax=ax, level=level, disp=disp)
#     plt.show()
