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
import cupy as cp
import numpy  as np
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
parser.add_argument('--silent', '-si', action='store_false',  default=0, help='silent debug print outs' )
parser.add_argument('--visualize', '-vz', action='store_false', help='visualize level sets?' )
parser.add_argument('--elevation', '-el', type=float, default=5., help='elevation angle for target set plot.' )
parser.add_argument('--azimuth', '-az', type=float, default=5., help='azimuth angle for target set plot.' )
parser.add_argument('--pause_time', '-pz', type=float, default=4, help='pause time between successive updates of plots' )
args = parser.parse_args()
args.verbose = True if not args.silent else False

print('args: ', args)

if not args.silent:
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
				obj.omega_e_bound*cp.abs(deriv[0]*obj.grid.xs[1] - \
				deriv[1] * obj.grid.xs[0] - deriv[2])  + \
				obj.omega_p_bound * cp.abs(deriv[2])

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
	grid_min = expand(np.array((-.75, -1.25, -pi)), ax = 1)
	grid_max = expand(np.array((3.25, 1.25, pi)), ax = 1)
	pdDims = 2                      # 3rd dimension is periodic
	resolution = 100
	N = np.array(([[
					resolution,
					np.ceil(resolution*(grid_max[1, 0] - grid_min[1, 0])/ \
								(grid_max[0, 0] - grid_min[0, 0])),
					resolution-1
					]])).T.astype(int)
	grid_max[2, 0]*= (1-2/N[2,0])

	obj.grid = createGrid(grid_min, grid_max, N, pdDims)

	# global params
	obj.axis_align, obj.center, obj.radius = 2, np.zeros((3, 1)), 0.5
	data0 = get_target(obj)

	data = copy.copy(data0)

	obj.v_e			  = +1
	obj.v_p			  = +1

	obj.omega_e_bound = +1
	obj.omega_p_bound = +1

	t_range = [0, 2.5]

	obj.p1_term = obj.v_e - obj.v_p * cp.cos(obj.grid.xs[2])
	obj.p2_term = -obj.v_p * cp.sin(obj.grid.xs[2])
	obj.alpha = [ cp.abs(obj.p1_term) + cp.abs(obj.omega_e_bound * obj.grid.xs[1]), \
					cp.abs(obj.p2_term) + cp.abs(obj.omega_e_bound * obj.grid.xs[0]), \
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

	# Visualization paramters
	spacing = tuple(obj.grid.dx.flatten().tolist())
	data_np = data.get()
	init_mesh = implicit_mesh(data_np, level=0, spacing=spacing, edge_color='b', face_color='b')
	params = Bundle(
			{"grid": obj.grid,
			 'disp': True,
			 'labelsize': 16,
			 'labels': "Initial 0-LevelSet",
			 'linewidth': 2,
			 'data': data_np,
			 'elevation': args.elevation,
			 'azimuth': args.azimuth,
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

	while(t_range[1] - t_now > small * t_range[1]):

		time_step = f"{t_now}/{t_range[-1]}"

		# Reshape data array into column vector for ode solver call.
		y0 = data.flatten()

		# How far to step?
		t_span = cp.hstack([ t_now, min(t_range[1], t_now + t_plot) ])

		# Take a timestep.
		t, y, _ = odeCFL2(termRestrictUpdate, t_span, y0, integratorOptions, finite_diff_data)
		t_now = t

		logger.info(f't: {t:.3f}/{t_range[-1]} TargSet Min: {min(y):.3f}, TargSet Max: {max(y):.3f} TargSet Norm: {cp.linalg.norm(y)}')

		# Get back the correctly shaped data array
		data = cp.reshape(y, obj.grid.shape)

		mesh=implicit_mesh(data.get(), level=0, spacing=spacing,  edge_color='None',
	                         face_color='red')

		if args.visualize:
			rcbrt_viz.update_tube(data.get(), mesh.get(), time_step)
	
	if args.visualize:
		rcbrt_viz.update_tube(data.get(), params.mesh, time_step)

	end_time = cputime()
	info(f'Total execution time {end_time - start_time} seconds.')

if __name__ == '__main__':
	main(args)
