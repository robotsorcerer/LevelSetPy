__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

from LevelSetPy.Utilities import size, error


def checkInterpInput(g, x):
    if size(x, 1) != g.dim:
        if size(x, 0) == g.dim:
            # Take transpose if number of inp.t rows is same as grid dimension
            x = x.T;
        else:
            error('Input points must have the same dimension as grid!')

    return x
