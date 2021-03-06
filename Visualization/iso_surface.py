__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

import numpy as np

def isosurface(my_array, my_value, zs, interp_order=6, power_parameter=0.5):
    #https://stackoverflow.com/questions/13627104/using-numpy-scipy-to-calculate-iso-surface-from-3d-array
    'uses a weighted average to compute the iso-surface:'
    if interp_order < 1: interp_order = 1
    dist = (my_array - my_value)**2
    arg = np.argsort(dist,axis=2)
    dist.sort(axis=2)
    w_total = 0.
    z = np.zeros(my_array.shape[:2], dtype=float)
    for i in range(int(interp_order)):
        zi = np.take(zs, arg[:,:,i])
        valuei = dist[:,:,i]
        wi = 1/valuei
        np.clip(wi, 0, 1.e6, out=wi) # avoiding overflows
        w_total += wi**power_parameter
        z += zi*wi**power_parameter
    z /= w_total
    return z
