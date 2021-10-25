__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Decomposing Level Sets of PDEs"
__credits__  	= "Sylvia Herbert, Ian Abraham"
__license__ 	= "Lekan License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import time
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from Utilities.matlab_utils import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Visualization.mesh_implicit import implicit_mesh


def buffered_axis_limits(amin, amax, buffer_factor=1.0):
    """
    Increases the range (amin, amax) by buffer_factor on each side
    and then rounds to precision of 1/10th min or max.
    Used for generating good plotting limits.
    For example (0, 100) with buffer factor 1.1 is buffered to (-10, 110)
    and then rounded to the nearest 10.
    """
    diff = amax - amin
    amin -= (buffer_factor-1)*diff
    amax += (buffer_factor-1)*diff
    magnitude = np.floor(np.log10(np.amax(np.abs((amin, amax)) + 1e-100)))
    precision = np.power(10, magnitude-1)
    amin = np.floor(amin/precision) * precision
    amax = np.ceil (amax/precision) * precision
    return (amin, amax)

def get_field(field, bundle):
    if isfield(field, bundle):
        return bundle.__dict__['field']
    else:
        return

class InteractiveVisualizer():
    def __init__(fig, params={}, num_plots=2,
                rows=None, cols=None):

        assert num_plots is not None, 'Number of Plots cannot be None.'

        if cols is None:
            cols = int(np.floor(np.sqrt(num_plots)))
        if rows is None:
            rows = int(np.ceil(float(num_plots)/cols))

        assert num_plots <= rows*cols, 'Too many plots to put into gridspec.'

		self._labelsize = params.labelsize
		self._init      = False
		self.value      = value
		self._fontdict  = params.fontdict
		self.pause_time = params.pause_time
        self.savedict   = params.savedict
        self.savepath   = params.savepath

		plt.ion()
        self._fig = plt.figure(figsize=params.winsize)
		self._gs_plots = gridspec.GridSpec(rows, cols, self._fig)
        # self._gs_plot   = self._gs[1:8, 0]
        # self._gs_legend = self._gs[0, 0]

		self._ax = plt.subplot(self._gs[0])
        self._ax_arr = [np.nan for _ in range(num_plots)]
        # self._axarr = [plt.subplot(self._gs_plots[i], projection='3d') for i in range(num_plots)]

        # self._ax_legend = plt.subplot(self._gs_legend)
        # self._ax_legend.get_xaxis().set_visible(False)
        # self._ax_legend.get_yaxis().set_visible(False)

		if self.params.init_conditions::
            assert isinstance(value, np.ndarray), "value function must be a numpy array."
			self.init_projections(value.ndim)

		self._fig.canvas.draw()
		self._fig.canvas.flush_events()

	def init_projections(self, data_len):
		"""
		Initialize plots based off the length of the data array.
		"""
		self._data_len = data_len
        # Show me the value function
        if self._data_len==2:
            # ax = self._fig.add_subplot(1, 1, 1)
            self._ax_arr[0] = plt.subplot(self._gs_plots[0])
    		self._ax_arr[0].plot(self.value[0], self.value[1], 'b*') #,markersize=15,linewidth=3, label='Target')
        elif self._data_len==3:
            self._ax_arr[0] = plt.subplot(self._gs_plots[0], projection='3d')
    		self._ax_arr[0].plot(self.value[0], self.value[1], self.value[2], 'b*') #,markersize=15,linewidth=3, label='Target')

		# self._ax.set_xlabel('$\\xi$', fontdict=self._fontdict)
		# self._ax.set_ylabel('$\\dot{\\xi}$', fontdict=self._fontdict)

		self._ax.xaxis.set_tick_params(labelsize=self._labelsize)
		self._ax.yaxis.set_tick_params(labelsize=self._labelsize)

		self._ax.grid('on')
		self._ax.legend(loc='best')
		self._ax.set_title('Initial Projections', fontdict=self._fontdict)
		self._init = True

    def initial_valueset(self, g, data):
        if g.dim==2:
            self._ax_arr[0] = plt.subplot(self._gs_plots[0], projection='3d')
    		self._ax_arr[0].plot(self.value[0], self.value[1], 'b*') #,markersize=15,linewidth=3, label='Target')
            ax.contourf(g.xs[0], g.xs[1], mesh, colors=fc)
        ax.set_xlabel('X', fontdict=fontdict)
        ax.set_ylabel('Y', fontdict=fontdict)
        ax.set_zlabel('Z', fontdict=fontdict)
        ax.set_title(f'Contours')


    def set_title(self, i, title):
            self._axarr[i].set_title(title)
            self._axarr[i].title.set_fontsize(10)

    def add_legend(self, linestyle, marker, color, label):
        self._ax_legend.plot([], [], linestyle=linestyle, marker=marker,
                color=color, label=label)
        self._ax_legend.legend(ncol=2, mode='expand', fontsize=10)
