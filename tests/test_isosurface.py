__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import numpy as cp.#meshgrid, sin, cos, pi, linspace
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import sys, os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from Visualization.iso_surface import isosurface
from Grids import createGrid
from InitialConditions import shapeCylinder
import pyvista as pv
# def main():
#     dx = 100; dy =  50; dz = 25
#     nx = 200; ny = 100; nz = 100
#     xs = cp.linspace(0,dx,nx)
#     ys = cp.linspace(0,dy,ny)
#     zs = cp.linspace(0,dz,nz)
#     X,Y,Z = cp.meshgrid( xs, ys, zs)
#     my_array = cp.sin(0.3*cp.pi+0.4*cp.pi*X/dx)*cp.sin(0.3*cp.pi+0.4*cp.pi*Y/dy)*(Z/dz)
#
#     fig = plt.figure(figsize=(16,9))
#     ax = fig.add_subplot(1,1,1,projection='3d')
#
#     z = isosurface( my_array, my_value=0.1, zs=zs, interp_order=6 )
#     ax.plot_surface(X[:,:,0], Y[:,:,0], z, cstride=4, rstride=4, color='g')
#
#     z = isosurface( my_array, my_value=0.2, zs=zs, interp_order=6 )
#     ax.plot_surface(X[:,:,0], Y[:,:,0], z, cstride=4, rstride=4, color='y')
#
#     z = isosurface( my_array, my_value=0.3, zs=zs, interp_order=6 )
#     ax.plot_surface(X[:,:,0], Y[:,:,0], z, cstride=4, rstride=4, color='b')
#
#     plt.ioff()
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
g3min = -2*cp.ones((3, 1),dtype=cp.float64)
g3max = +2*cp.ones((3, 1),dtype=cp.float64)
# print(g3min, g3min.shape)
g3N = 51*cp.ones((3, 1),dtype=cp.int64)
g3 = createGrid(g3min, g3max, g3N, process=True)
cylinder = shapeCylinder(g3, 2, .5*cp.ones((3, 1), cp.float64), 0.5);

mesh = pv.ExplicitStructureGrid(g3.xs[0], g3.xs[1], g3.xs[2])
mesh.point_arrays['values'] = cylinder.ravel(order='F')  # also the active scalars

# compute 3 isosurfaces
isos = mesh.contour(isosurfaces=3, rng=[10, 40])
# or: mesh.contour(isosurfaces=cp.linspace(10, 40, 3)) etc.

# plot them interactively if you want to
isos.plot(opacity=0.7)
