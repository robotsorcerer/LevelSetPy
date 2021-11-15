__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "Sylvia Herbert."
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

import copy
import time
import logging
import argparse
import sys, os
import numpy as np
from math import pi
from Grids import createGrid
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Visualization import RCBRTVisualizer

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from os.path import dirname, abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))

from ValueFuncs import *
from Visualization import *
from DynamicalSystems import *
from InitialConditions import shapeCylinder
from Utilities import *
from ExplicitIntegration import *
from SpatialDerivative import *
from Visualization.value_viz import ValueVisualizer

parser = argparse.ArgumentParser(description='Hamilton-Jacobi Analysis')
parser.add_argument('--compute_traj', '-ct', action='store_true', default=False, help='compute trajectory?')
parser.add_argument('--silent', '-si', action='store_false', help='silent debug print outs' )
parser.add_argument('--visualize', '-vz', type=int, default=0, help='visualize level sets?' )
parser.add_argument('--pause_time', '-pz', type=float, default=5e-3, help='pause time between successive updates of plots' )
args = parser.parse_args()

print('args: ', args)

if args.silent:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
else:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
# Turn off pyplot's spurious dumps on screen
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

obj = Bundle({})

# define the target set
def get_target(g):
	cylinder = shapeCylinder(g.grid, g.axis_align, g.center, g.radius);
	return cylinder

def get_hamiltonian_func(t, data, deriv, finite_diff_data):
	global obj
	ham_value = deriv[0] * obj.p1_term + \
				deriv[1] * obj.p2_term - \
				obj.omega_e_bound*np.abs(deriv[0]*obj.grid.xs[1] - \
				deriv[1] * obj.grid.xs[0] - deriv[2])  + \
				obj.omega_p_bound * np.abs(deriv[2])

	return ham_value, finite_diff_data

def get_partial_func(t, data, derivMin, derivMax, \
			  schemeData, dim):
	"""
		Calculate the extrema of the absolute value of the partials of the
		analytic Hamiltonian with respect to the costate (gradient).
	"""
	global obj

	# print('dim: ', dim)
	assert dim>=0 and dim <3, "grid dimension has to be between 0 and 2 inclusive."

	return obj.alpha[dim]

def main(args):
	## Grid
	grid_min = expand(np.array((-.75, -1.25, -pi)), ax = 1) # Lower corner of computation domain
	grid_max = expand(np.array((3.25, 1.25, pi)), ax = 1)   # Upper corner of computation domain
	pdDims = 2                      # 3rd dimension is periodic
	resolution = 100
	N = np.array(([[
					resolution,
					np.ceil(resolution*(grid_max[1, 0] - grid_min[1, 0])/ \
								(grid_max[0, 0] - grid_min[0, 0])),
					resolution-1
					]])).T.astype(int)
	grid_max[2, 0]*= (1-2/N[2])

	# be careful not to create periodic dims for this problem
	obj.grid = createGrid(grid_min, grid_max, N, pdDims)

	# print(f'[RCBRT@] grid.max: {obj.grid.max.T} grid.min: {obj.grid.min.T}')
	# print(f'[RCBRT@] grid.N: {obj.grid.N.T}')
	# print(f'[RCBRT@] grid.dx: {obj.grid.dx.T}')

	# global params
	obj.axis_align, obj.center, obj.radius = 2, np.zeros((3, 1)), 0.5
	data0 = get_target(obj)

	data = copy.copy(data0)

	obj.v_e			  = +1
	obj.v_p			  = +1

	obj.omega_e_bound = +1
	obj.omega_p_bound = +1

	t_range = [0, 2.5]

	obj.p1_term = obj.v_e - obj.v_p * np.cos(obj.grid.xs[2])
	obj.p2_term = -obj.v_p * np.sin(obj.grid.xs[2])
	obj.alpha = [ np.abs(obj.p1_term) + np.abs(obj.omega_e_bound * obj.grid.xs[1]), \
					np.abs(obj.p2_term) + np.abs(obj.omega_e_bound * obj.grid.xs[0]), \
					obj.omega_e_bound + obj.omega_p_bound ]

	small = 100*eps
	level = 0

	finite_diff_data = Bundle({'grid': obj.grid, 'hamFunc': get_hamiltonian_func,
								'partialFunc': get_partial_func,
								'dissFunc': artificialDissipationGLF,
								'derivFunc': upwindFirstENO2,
								})

	options = Bundle(dict(factorCFL=0.95, stats='on', singleStep='off'))
	integratorOptions = odeCFLset(options)

	"""
	---------------------------------------------------------------------------
	 Restrict the Hamiltonian so that reachable set only grows.
	   The Lax-Friedrichs approximation scheme MUST already be completely set up.
	"""
	innerData = copy.copy(finite_diff_data)
	del finite_diff_data

	# Wrap the true Hamiltonian inside the term approximation restriction routine.
	schemeFunc = termRestrictUpdate
	finite_diff_data = Bundle(dict(innerFunc = termLaxFriedrichs,
								   innerData = innerData,
								   positive = 0
								))

	# Period at which intermediate plots should be produced.
	plot_steps = 11
	t_plot = (t_range[1] - t_range[0]) / (plot_steps - 1)
	# Loop through t_range (subject to a little roundoff).
	t_now = t_range[0]
	start_time = cputime()

	# plt.close('all')
	# Visualization paramters
	spacing = tuple(obj.grid.dx.flatten().tolist())
	init_mesh = implicit_mesh(data, level=0, spacing=spacing,  edge_color='orange',
	                         face_color='orange')
	params = Bundle(
			{"grid": obj.grid,
			 'disp': True,
			 'labelsize': 16,
			 'labels': "Initial 0-LevelSet",
			 'linewidth': 2,
			 'data': data,
			 'mesh': init_mesh,
			 'init_conditions': False,
			 'pause_time': args.pause_time,
			 'level': 0, # which level set to visualize
			 'winsize': (12,7),
			 'fontdict': Bundle({'fontsize':12, 'fontweight':'bold'}),
			 "savedict": Bundle({"save": False,
			 				"savename": "rcbrt",
			 				"savepath": "../jpeg_dumps/rcbrt"})
			 })

	if args.visualize:
		rcbrt_viz = RCBRTVisualizer(params=params)

	# print(f'dx: {obj.grid.dx.T}')
	while(t_range[1] - t_now > small * t_range[1]):

		time_step = f"{t_now}/{t_range[-1]}"
		if(strcmp(options.stats, 'on')):
			info(f"Iteration: {time_step}")

		# Reshape data array into column vector for ode solver call.
		y0 = data.flatten()
		#print(f'y0 rcbrt: {np.linalg.norm(y0)}')
		# print(f'y0 from data0 rcbrt: {np.linalg.norm(data0)}')
		# print(f'min y0: {min(y0):.3f}, max y0: {max(y0):.3f}')
		# import time; time.sleep(40)
		# correct up to here

		# How far to step?
		t_span = np.hstack([ t_now, min(t_range[1], t_now + t_plot) ])

		# Take a timestep.
		t, y, _ = odeCFL2(termRestrictUpdate, t_span, y0, integratorOptions, finite_diff_data)
		t_now = t

		print(f't: {t:.3f} min y: {min(y):.3f}, max y: {max(y):.3f} normy: {np.linalg.norm(y)}')

		# Get back the correctly shaped data array
		data = np.reshape(y, obj.grid.shape)

		# TODO: Why doesn't mesh have zero values?
		# mesh=implicit_mesh(data, level=None, spacing=spacing,  edge_color='red',
	    #                      face_color='red')
		verts, faces, normals, values = measure.marching_cubes(data, level=None, spacing=spacing, gradient_direction='descent')
		# Fancy indexing: `verts[faces]` to generate a collection of triangles
		mesh = Poly3DCollection(verts[faces])
		mesh.set_edgecolor('magenta')
		mesh.set_facecolor('magenta')

		if args.visualize:
			rcbrt_viz.update_tube(data, mesh, time_step, delete_last_plot)

	end_time = cputime()
	info(f'Total execution time {end_time - start_time} seconds.')

if __name__ == '__main__':
	main(args)
