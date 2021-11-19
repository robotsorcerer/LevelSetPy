__all__ = ["RCBRTVisualizer"]

import time
import numpy as np
from skimage import measure
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Visualization.mesh_implicit import implicit_mesh


class RCBRTVisualizer(object):
	def __init__(self, params=None):
		"""
			Class RCBRTVisualizer:

			This class expects to be constantly given values to plot in realtime.
			It assumes the values are an array and plots different indices at different
			colors according to the spectral colormap.

			Inputs:
				mesh: Initial mesh
		"""
		plt.ion()
		if params.winsize:
			self._fig = plt.figure(figsize=params.winsize)
			self.winsize=params.winsize
		else:
			self._fig = fig

		self.grid = params.grid
		self._gs  = gridspec.GridSpec(1, 2, self._fig)
		self._ax  = [plt.subplot(self._gs[i], projection='3d') for i in [0, 1]]

		self._init = False
		self.params = params

		if self.params.savedict.save and not os.path.exists(self.savedict["savepath"]):
			os.makedirs(self.params.savedict.savepath)

		if self.params.fontdict is None:
			self._fontdict = {'fontsize':12, 'fontweight':'bold'}

		if np.any(params.mesh):
			self.init(params.mesh)
			self._init = True

		self._fig.canvas.draw()
		self._fig.canvas.flush_events()

	def init(self, mesh=None):
		"""
			Plot the initialize target set mesh.
			Inputs:
				data: marching cubes mesh
		"""
		cm = plt.get_cmap('rainbow')
		self._ax[0].grid('on')
		self._ax[0].add_collection3d(mesh)

		self._ax[0].view_init(elev=self.params.elevation, azim=self.params.azimuth)
		self._ax[1].view_init(elev=self.params.elevation, azim=self.params.azimuth)

		self._ax[0].axes.get_xaxis().set_ticks([])
		self._ax[0].axes.get_yaxis().set_ticks([])

		self._ax[1].axes.get_xaxis().set_ticks([])
		self._ax[1].axes.get_yaxis().set_ticks([])

		if isinstance(mesh, list):
			for m in mesh:
				m = implicit_mesh(m, level=self.params.level, spacing=spacing,  edge_color='k', face_color='red')
				self._ax[0].add_collection3d(m)
		else:
			self._ax[0].add_collection3d(mesh)

		data = self.params.data
		xlim = (min(data[0].ravel()), max(data[0].ravel()))
		ylim = (self.params.grid.min.item(1)-.5, self.params.grid.min.item(1)+4.)
		zlim = (self.params.grid.min.item(2)-1.3, self.params.grid.max.item(2)+4.5)

		self._ax[0].set_xlim3d(*xlim)
		self._ax[0].set_ylim3d(*ylim)
		self._ax[0].set_zlim3d(*zlim)
		# self._ax[0].autoscale()

		self._ax[0].set_xlabel("X", fontdict = self.params.fontdict.__dict__)
		self._ax[0].set_ylabel("Y", fontdict = self.params.fontdict.__dict__)
		self._ax[0].set_zlabel("Z", fontdict = self.params.fontdict.__dict__)
		self._ax[0].set_title(f"Initial {self.params.level}-Level Value Set", \
								fontweight=self.params.fontdict.fontweight)
		self._ax[1].set_title(f'BRT at {0} secs.', fontweight=self.params.fontdict.fontweight)

	def update_tube(self, data, mesh, time_step, delete_last_plot=False):
		self._ax[1].grid('on')
		self._ax[1].add_collection3d(mesh)
		self._ax[1].view_init(elev=self.params.elevation, azim=self.params.azimuth)

		self._ax[1].axes.get_xaxis().set_ticks([])
		self._ax[1].axes.get_yaxis().set_ticks([])

		if delete_last_plot:
			plt.cla()

		if isinstance(mesh, list):
			for m in mesh:
				m = implicit_mesh(m, level=self.params.level, spacing=spacing,  edge_color='orange', face_color='orange')
				self._ax[1].add_collection3d(m)
		else:
			self._ax[1].add_collection3d(mesh)

		xlim = (min(data[0].ravel())-2, max(data[0].ravel())+2)
		ylim = (min(data[1].ravel())-3, max(data[1].ravel())+2.)
		zlim = (min(data[2].ravel())-1.3, max(data[2].ravel())+4.1)

		self._ax[1].set_xlim3d(*xlim)
		self._ax[1].set_ylim3d(*ylim)
		self._ax[1].set_zlim3d(*zlim)

		self._ax[1].set_xlabel("X", fontdict = self.params.fontdict.__dict__)
		self._ax[1].set_title(f'BRT at {time_step}.', fontweight=self.params.fontdict.fontweight)

		self.draw()
		time.sleep(self.params.pause_time)

	def add_legend(self, linestyle, marker, color, label):
		self._ax_legend.plot([], [], linestyle=linestyle, marker=marker,
				color=color, label=label)
		self._ax_legend.legend(ncol=2, mode='expand', fontsize=10)

	def draw(self, ax=None):
		self._fig.canvas.draw()
		self._fig.canvas.flush_events()
